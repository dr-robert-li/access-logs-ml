import json
import pandas as pd
import argparse
import os
import psutil
import numpy as np
from tqdm import tqdm

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def prepare_training_data(df_labeled):
    """Prepare the labeled data for fine-tuning an LLM."""
    initial_memory = get_memory_usage()
    print(f"Memory before preparing training data: {initial_memory:.2f} MB")
    
    # Convert features to text format for the LLM
    training_data = []
    
    for _, row in tqdm(df_labeled.iterrows(), total=len(df_labeled), desc="Preparing training data"):
        # Create input text (prompt)
        input_text = f"""
        Analyze this web request:
        IP: {row['ip']} ({row['country']}, {row['asn']})
        Request: {row['method']} {row['path']} {row['protocol']}
        User-Agent: {row['user_agent']}
        Status: {row['status_code']}
        Request frequency: {row['requests_per_minute']:.2f} req/min
        Time between requests: {row['time_diff'] if pd.notna(row['time_diff']) else 'N/A'} seconds
        User agent indicators: {'No user agent' if not row['has_user_agent'] else ('Old browser' if row['is_old_browser'] else ('Known bot' if row['is_known_bot'] else 'Modern browser'))}
        """
        
        # Create output text (completion)
        output_text = f"""
        Classification: {row['classification']}
        Confidence: {row['confidence']:.2f}
        Reasoning: {row['reasoning']}
        """
        
        training_data.append({
            "input": input_text.strip(),
            "output": output_text.strip()
        })
    
    current_memory = get_memory_usage()
    print(f"Memory after preparing training data: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    
    return training_data

def save_for_fine_tuning(training_data, output_file="training_data.jsonl"):
    """Save the training data in JSONL format for fine-tuning."""
    initial_memory = get_memory_usage()
    print(f"Memory before saving training data: {initial_memory:.2f} MB")
    
    with open(output_file, 'w') as f:
        for item in tqdm(training_data, desc="Saving training data"):
            f.write(json.dumps(item) + '\n')
    
    current_memory = get_memory_usage()
    print(f"Memory after saving training data: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    print(f"Saved {len(training_data)} training examples to {output_file}")

def process_dataframe_in_chunks(input_file, chunk_size=50000):
    """
    Generator function to process a large CSV file in chunks.
    
    Args:
        input_file: Path to the input CSV file
        chunk_size: Number of rows to process in each chunk
        
    Yields:
        DataFrame chunks
    """
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        yield chunk

def create_validation_set(input_file, output_file, chunk_size=50000):
    """
    Main function to create a validation set for model training.
    
    Args:
        input_file: Path to the input CSV file with labeled logs
        output_file: Path to save the training data in JSONL format
        chunk_size: Number of rows to process in each chunk
        
    Returns:
        str: Path to the output file
    """
    initial_memory = get_memory_usage()
    print(f"Starting with memory usage: {initial_memory:.2f} MB")
    
    # For very large files, process in chunks
    print(f"Processing {input_file} in chunks of {chunk_size} rows...")
    
    # Get total number of rows for progress tracking
    total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract header row
    print(f"Total rows to process: {total_rows}")
    
    # Process chunks and write directly to output file
    with open(output_file, 'w') as f:
        total_examples = 0
        
        for chunk_num, chunk in enumerate(process_dataframe_in_chunks(input_file, chunk_size)):
            chunk_start_memory = get_memory_usage()
            print(f"\n--- Processing chunk {chunk_num+1} ({len(chunk)} rows) ---")
            print(f"Memory at start of chunk: {chunk_start_memory:.2f} MB")
            
            # Prepare training data for this chunk
            training_data = prepare_training_data(chunk)
            
            # Write directly to file
            for item in tqdm(training_data, desc=f"Writing chunk {chunk_num+1} to file"):
                f.write(json.dumps(item) + '\n')
            
            total_examples += len(training_data)
            chunk_end_memory = get_memory_usage()
            
            print(f"Chunk {chunk_num+1} complete. Memory: {chunk_end_memory:.2f} MB (Δ: {chunk_end_memory - chunk_start_memory:.2f} MB)")
            print(f"Processed {len(training_data)} examples from chunk {chunk_num+1}")
            print(f"Progress: {total_examples}/{total_rows} rows processed ({total_examples/total_rows*100:.1f}%)")
    
    # Final memory report
    final_memory = get_memory_usage()
    print(f"\nSaved {total_examples} training examples to {output_file}")
    print(f"Final memory usage: {final_memory:.2f} MB (Δ: {final_memory - initial_memory:.2f} MB)")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Prepare labeled data for model training')
    parser.add_argument('--input', required=True, help='Path to labeled logs CSV file')
    parser.add_argument('--output', default='training_data.jsonl', help='Path to save the training data')
    parser.add_argument('--chunk-size', type=int, default=50000, help='Number of rows to process in each chunk')
    args = parser.parse_args()
    
    create_validation_set(args.input, args.output, args.chunk_size)

if __name__ == "__main__":
    main()
