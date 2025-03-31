from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset, load_dataset
import torch
import json
import argparse
import os
import time
import psutil
from tqdm import tqdm
import pickle
import zipfile
import tempfile
from pathlib import Path

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def prepare_dataset(jsonl_file, validation_split=0.1):
    """
    Load and prepare the dataset from a JSONL file, splitting into train and validation.
    
    Args:
        jsonl_file: Path to the JSONL file with training data
        validation_split: Fraction of data to use for validation (default: 0.1)
        
    Returns:
        tuple: (train_dataset, validation_dataset) HuggingFace Dataset objects
    """
    initial_memory = get_memory_usage()
    print(f"Memory before loading dataset: {initial_memory:.2f} MB")
    
    print(f"Loading dataset from {jsonl_file}...")
    
    # Load the dataset
    full_dataset = load_dataset('json', data_files=jsonl_file, split='train')
    
    # Split into train and validation
    split_datasets = full_dataset.train_test_split(test_size=validation_split)
    train_dataset = split_datasets['train']
    validation_dataset = split_datasets['test']  # 'test' is the validation set in this context
    
    current_memory = get_memory_usage()
    print(f"Dataset loaded with {len(train_dataset)} training examples and {len(validation_dataset)} validation examples")
    print(f"Memory after loading dataset: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    
    return train_dataset, validation_dataset

# Modify max length for memory efficiency (longer tokens = more memory consumption)
def tokenize_function(examples, tokenizer, max_length=256):
    """
    Tokenize the input and output texts.
    
    Args:
        examples: Batch of examples to tokenize
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        
    Returns:
        dict: Tokenized examples
    """
    inputs = tokenizer(examples["input"], truncation=True, max_length=max_length, padding="max_length")
    outputs = tokenizer(examples["output"], truncation=True, max_length=max_length, padding="max_length")
    
    inputs["labels"] = outputs["input_ids"]
    return inputs

class MemoryCallback(TrainerCallback):
    """Custom callback to track memory usage during training."""
    def __init__(self):
        self.initial_memory = get_memory_usage()
        print(f"Initial memory before training: {self.initial_memory:.2f} MB")
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        current_memory = get_memory_usage()
        print(f"Memory at start of epoch {state.epoch}: {current_memory:.2f} MB (Δ: {current_memory - self.initial_memory:.2f} MB)")
    
    def on_epoch_end(self, args, state, control, **kwargs):
        current_memory = get_memory_usage()
        print(f"Memory at end of epoch {state.epoch}: {current_memory:.2f} MB (Δ: {current_memory - self.initial_memory:.2f} MB)")
    
    def on_train_end(self, args, state, control, **kwargs):
        final_memory = get_memory_usage()
        print(f"Final memory after training: {final_memory:.2f} MB (Δ: {final_memory - self.initial_memory:.2f} MB)")

def train_model(input_file, output_dir, model_name="deepseek-ai/deepseek-coder-1.3b-base", epochs=3, validation_split=0.1):
    """
    Fine-tune a model on the labeled data with validation.
    
    Args:
        input_file: Path to the JSONL file with training data
        output_dir: Directory to save the fine-tuned model
        model_name: Base model to fine-tune
        epochs: Number of training epochs
        validation_split: Fraction of data to use for validation
        
    Returns:
        str: Path to the fine-tuned model
    """
    initial_memory = get_memory_usage()
    print(f"Starting with memory usage: {initial_memory:.2f} MB")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect device type
    if torch.cuda.is_available():
        device_type = "cuda"
        print("CUDA GPU detected")
    elif hasattr(torch, 'mps') and torch.mps.is_available():
        device_type = "mps"
        print("Apple Silicon MPS device detected")
    else:
        device_type = "cpu"
        print("No GPU detected, using CPU")
    
    # Load tokenizer
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with appropriate device configuration
    print(f"Loading model {model_name}...")
    
    try:
        # For MPS devices, we need to be more careful with memory usage
        if device_type == "mps":
            # For MPS, load the model on CPU first, then move specific modules to MPS as needed
            print("Loading model on CPU first due to MPS memory constraints...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,  # Use float32 for MPS compatibility
                device_map="cpu",  # Load on CPU first
                low_cpu_mem_usage=True
            )
            # Move only essential parts to MPS to avoid OOM
            print("Moving model to MPS device...")
            # Only move the model to MPS if it's small enough, otherwise keep on CPU
            try:
                # Try to estimate model size
                model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
                print(f"Estimated model size: {model_size_gb:.2f} GB")
                
                if model_size_gb < 4.0:  # Only move models smaller than 4GB to MPS
                    model = model.to("mps")
                else:
                    print(f"Model is too large ({model_size_gb:.2f} GB) for MPS. Keeping on CPU.")
                    # Force CPU device type for training configuration
                    device_type = "cpu"
            except Exception as e:
                print(f"Error when trying to move model to MPS: {e}")
                print("Keeping model on CPU for stability.")
                device_type = "cpu"
        else:
            # For CUDA or CPU, use standard loading
            model = AutoModelForCausalLM.from_pretrained(model_name)
            if device_type == "cuda":
                model = model.to("cuda")

    except Exception as e:
        print(f"Error loading model: {e}")
        print("Attempting to load model with safetensors disabled...")
        try:
            # Try loading with safetensors disabled as a fallback
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                use_safetensors=False,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            if device_type == "cuda":
                model = model.to("cuda")
            elif device_type == "mps" and sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9 < 4.0:
                model = model.to("mps")
        except Exception as e2:
            print(f"Second attempt to load model failed: {e2}")
            raise RuntimeError(f"Failed to load model {model_name}. Please try a different model.") from e2
    
    # Check if model was successfully loaded
    if model is None:
        raise RuntimeError(f"Failed to load model {model_name}. Please try a different model.")
    
    model_load_memory = get_memory_usage()
    print(f"Memory after loading model: {model_load_memory:.2f} MB (Δ: {model_load_memory - initial_memory:.2f} MB)")

    # Prepare dataset with train/validation split
    train_dataset, validation_dataset = prepare_dataset(input_file, validation_split)
    
    # Tokenize datasets with reduced sequence length for MPS/CPU
    max_length = 128 if device_type in ["mps", "cpu"] else 256
    print(f"Tokenizing datasets with max_length={max_length}...")
    
    tokenize_func = lambda examples: tokenize_function(examples, tokenizer, max_length=max_length)
    
    tokenized_train_dataset = train_dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training dataset"
    )
    
    tokenized_validation_dataset = validation_dataset.map(
        tokenize_func,
        batched=True,
        remove_columns=validation_dataset.column_names,
        desc="Tokenizing validation dataset"
    )
    
    tokenize_memory = get_memory_usage()
    print(f"Memory after tokenizing: {tokenize_memory:.2f} MB (Δ: {tokenize_memory - model_load_memory:.2f} MB)")

    # Set up training arguments based on device type
    print("Setting up training arguments...")
    
    # Common arguments for all device types
    training_args_dict = {
        "output_dir": output_dir,
        "num_train_epochs": epochs,
        "save_strategy": "epoch",  
        "save_total_limit": 2,
        "logging_dir": os.path.join(output_dir, 'logs'),
        "logging_steps": 100,
        "report_to": "none",  # Disable wandb, etc.
        "eval_strategy": "epoch",  # Evaluate at the end of each epoch
        "load_best_model_at_end": True,  # Load the best model at the end of training
        "metric_for_best_model": "eval_loss",  # Use validation loss to determine the best model
        "greater_is_better": False,  # Lower loss is better
    }
    
    # Device-specific optimizations
    if device_type == "cuda":
        # For CUDA GPUs, we can use fp16 and larger batch sizes
        training_args_dict.update({
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "optim": "adamw_torch"
        })
    elif device_type == "mps":
        # For MPS, use very conservative settings to avoid OOM
        training_args_dict.update({
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,  # More accumulation steps for smaller effective batch size
            "optim": "adamw_torch",
            "gradient_checkpointing": True,  # Enable gradient checkpointing to save memory
            "max_grad_norm": 0.5  # Lower gradient clipping for stability
        })
    else:
        # For CPU, use small batch size but more accumulation steps
        training_args_dict.update({
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 16,  # More accumulation for CPU
            "optim": "adamw_torch"
        })
    
    training_args = TrainingArguments(**training_args_dict)

    # Initialize Trainer with memory callback and validation dataset
    print("Initializing trainer with validation...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_validation_dataset, 
        callbacks=[MemoryCallback()]
    )

    # Start fine-tuning with error handling
    print(f"Starting training for {epochs} epochs...")
    try:
        trainer.train()
        # Run evaluation after training
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        print(f"Evaluation results: {eval_results}")
    except RuntimeError as e:
        error_msg = str(e)
        if "MPS backend out of memory" in error_msg or "Invalid buffer size" in error_msg:
            print("Memory error encountered. Trying to reduce memory usage further...")
            # Fall back to CPU if memory issues occur
            print("Falling back to CPU training (this will be slower but more stable)...")
            model = model.to("cpu")
            training_args_dict.update({
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 16,
            })
            training_args = TrainingArguments(**training_args_dict)
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train_dataset,
                eval_dataset=tokenized_validation_dataset,
                callbacks=[MemoryCallback()]
            )
            trainer.train()
        else:
            # Re-raise other errors
            raise

    # Print final evaluation metrics
    print("Performing final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final validation loss: {eval_results['eval_loss']:.4f}")
    
    # Save evaluation results
    with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
        json.dump(eval_results, f)

    # Save the fine-tuned model
    print(f"Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Wait for model files to be completely saved
    print("Waiting for model files to complete saving...")
    
    # Check for model files with more flexible pattern matching
    max_wait_time = 120  # Maximum wait time in seconds
    start_time = time.time()
    files_saved = False

    while not files_saved and (time.time() - start_time) < max_wait_time:
        # List all files in the output directory
        all_files = os.listdir(output_dir)
        
        # Check for different model file patterns
        safetensor_files = [f for f in all_files if f.endswith('.safetensors')]
        pytorch_files = [f for f in all_files if f.endswith('.bin') and 'pytorch_model' in f]
        tokenizer_files = [f for f in all_files if f in ['tokenizer.json', 'tokenizer_config.json']]
        
        # Check if we have either safetensor files or pytorch files, plus tokenizer files
        if (safetensor_files or pytorch_files) and tokenizer_files:
            # Check if files are non-empty
            all_files_valid = True
            for file in safetensor_files + pytorch_files + tokenizer_files:
                file_path = os.path.join(output_dir, file)
                if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
                    all_files_valid = False
                    break
            
            if all_files_valid:
                files_saved = True
                print("All model files have been saved successfully.")
                break
        
        # If files not found or not valid yet, wait and try again
        print("Waiting for model files to be saved...")
        time.sleep(5)

    # Check if we timed out
    if not files_saved:
        print(f"Warning: Timed out after {max_wait_time} seconds waiting for model files.")
        print("Continuing anyway, but please check the model files manually.")

    # Create a compressed pickle file for easier transport
    print(f"Creating compressed pickle file for transportability...")
    try:
        # Create a temporary directory to store files for compression
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a dictionary to store model components
            model_data = {
                "model_name": model_name,
                "config": model.config.to_dict() if hasattr(model, "config") else None,
                "training_args": training_args.to_dict() if hasattr(training_args, "to_dict") else None,
            }
            
            # Save the model state dictionary
            model_state_dict_path = os.path.join(temp_dir, "model_state_dict.pt")
            torch.save(model.state_dict(), model_state_dict_path)
            
            # Save the tokenizer files
            tokenizer_dir = os.path.join(temp_dir, "tokenizer")
            os.makedirs(tokenizer_dir, exist_ok=True)
            tokenizer.save_pretrained(tokenizer_dir)
            
            # Save the model metadata
            with open(os.path.join(temp_dir, "model_data.pkl"), "wb") as f:
                pickle.dump(model_data, f)
            
            # Create a zip file containing all the necessary files
            pkl_path = f"{output_dir}.pkl"
            with zipfile.ZipFile(pkl_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add model state dict
                zipf.write(model_state_dict_path, "model_state_dict.pt")
                
                # Add model metadata
                zipf.write(os.path.join(temp_dir, "model_data.pkl"), "model_data.pkl")
                
                # Add tokenizer files
                for file_path in Path(tokenizer_dir).rglob("*"):
                    if file_path.is_file():
                        zipf.write(
                            file_path, 
                            os.path.join("tokenizer", os.path.relpath(file_path, tokenizer_dir))
                        )
            
            print(f"Compressed model saved to {pkl_path}")
    
    except Exception as e:
        print(f"Warning: Could not create compressed model file: {str(e)}")
        print("The uncompressed model directory is still available and usable.")
    
    # Final memory report
    final_memory = get_memory_usage()
    print(f"Training complete. Model saved to {output_dir}")
    print(f"Final memory usage: {final_memory:.2f} MB (Δ: {final_memory - initial_memory:.2f} MB)")
    
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Fine-tune a model on labeled log data')
    parser.add_argument('--input', required=True, help='Path to training data JSONL file')
    parser.add_argument('--output', default='./fine_tuned_model', help='Directory to save the fine-tuned model')
    parser.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base', help='Base model to fine-tune')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--validation_split', type=float, default=0.1, help='Fraction of data to use for validation')
    args = parser.parse_args()
    
    train_model(args.input, args.output, args.model, args.epochs)

if __name__ == "__main__":
    main()
