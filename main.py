#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import modules from src
from src.parse_logs import parse_logs
from src.feature_engineering import enrich_logs
from src.llm_labeling import label_logs
from src.create_validation_set import create_validation_set
from src.model_training import train_model
from src.inferencing import run_inference
from src.visualizer import visualize_results
from src.package_pipeline import package_pipeline

def setup_directories():
    """Create necessary directories if they don't exist."""
    os.makedirs('logs', exist_ok=True)
    os.makedirs('fine_tuned_model', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('visualizations', exist_ok=True)
    os.makedirs('training_data', exist_ok=True)
    os.makedirs('pipeline_model', exist_ok=True)

def run_full_pipeline(args):
    """Run the complete pipeline from log parsing to inference."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define file paths with timestamp to avoid overwriting
    parsed_logs = f"data/parsed_logs_{timestamp}.csv"
    enriched_logs = f"data/enriched_logs_{timestamp}.csv"
    labeled_logs = f"data/labeled_logs_{timestamp}.csv"
    training_data = f"data/training_data_{timestamp}.jsonl"
    model_dir = f"fine_tuned_model/model_{timestamp}"
    results_file = f"data/results_{timestamp}.csv"
    
    print(f"Starting full pipeline with timestamp: {timestamp}")
    
    # Step 1: Parse logs
    print("\n=== Step 1: Parsing Logs ===")
    parse_logs(args.input, parsed_logs, args.batch_size)
    
    # Step 2: Feature engineering
    print("\n=== Step 2: Feature Engineering ===")
    enrich_logs(parsed_logs, enriched_logs, args.batch_size)
    
    # Step 3: LLM labeling
    print("\n=== Step 3: LLM Labeling ===")
    label_logs(enriched_logs, labeled_logs, args.samples, args.batch_size, args.llm_provider)
    
    # Step 4: Create validation set
    print("\n=== Step 4: Creating Validation Set ===")
    create_validation_set(labeled_logs, training_data, args.batch_size)
    
    # Step 5: Train model
    print("\n=== Step 5: Training Model ===")
    train_model(training_data, model_dir, args.model, args.epochs, args.validation_split)
    
    # Step 6: Run inference
    print("\n=== Step 6: Running Inference ===")
    results_file = run_inference(model_dir, args.inference_input or enriched_logs, results_file, args.batch_size)
    time_based_file = f"{os.path.splitext(results_file)[0]}_time_based.csv"
    
    # Step 7: Generate visualizations (if requested)
    if args.visualize:
        print("\n=== Step 7: Generating Visualizations ===")
        visualization_dir = f"visualizations/report_{timestamp}"
        html_report = visualize_results(time_based_file, visualization_dir)
        print(f"Visualization report generated: {html_report}")
    
    print(f"\nPipeline complete!")
    print(f"- Detailed classification results saved to: {results_file}")
    print(f"- Time-based classification summary saved to: {time_based_file}")
    if args.visualize:
        print(f"- Visualization report: {html_report}")
    
    return results_file

def run_training_pipeline(args):
    """Run only the training part of the pipeline (parsing to model training)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define file paths with timestamp to avoid overwriting
    parsed_logs = f"data/parsed_logs_{timestamp}.csv"
    enriched_logs = f"data/enriched_logs_{timestamp}.csv"
    labeled_logs = f"data/labeled_logs_{timestamp}.csv"
    training_data = f"data/training_data_{timestamp}.jsonl"
    model_dir = f"fine_tuned_model/model_{timestamp}"
    
    print(f"Starting training pipeline with timestamp: {timestamp}")
    
    # Step 1: Parse logs
    print("\n=== Step 1: Parsing Logs ===")
    parse_logs(args.input, parsed_logs, args.batch_size)
    
    # Step 2: Feature engineering
    print("\n=== Step 2: Feature Engineering ===")
    enrich_logs(parsed_logs, enriched_logs, args.batch_size)
    
    # Step 3: LLM labeling
    print("\n=== Step 3: LLM Labeling ===")
    label_logs(enriched_logs, labeled_logs, args.samples, args.batch_size, args.llm_provider)
    
    # Step 4: Create validation set
    print("\n=== Step 4: Creating Validation Set ===")
    create_validation_set(labeled_logs, training_data, args.batch_size)
    
    # Step 5: Train model
    print("\n=== Step 5: Training Model ===")
    model_dir = train_model(training_data, model_dir, args.model, args.epochs, args.validation_split)
    pkl_path = f"{model_dir}.pkl"
    if os.path.exists(pkl_path):
        print(f"- Compressed model saved to: {pkl_path}")
    print(f"\nTraining pipeline complete!")
    print(f"- Trained model saved to: {model_dir}")
    if os.path.exists(pkl_path):
        print(f"- Compressed model saved to: {pkl_path}")
    print(f"- Training data saved to: {training_data}")
    print(f"- To use this model for inference, run:")
    print(f"  python main.py inference --model {model_dir} --input your_logs.csv --output results.csv")
    
    return model_dir

def run_inference_pipeline(args):
    """Run only the inference part of the pipeline (inference and visualization)."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define file paths with timestamp to avoid overwriting
    results_file = args.output or f"data/results_{timestamp}.csv"
    
    print(f"Starting inference pipeline with timestamp: {timestamp}")
    
    # Step 1: Run inference
    print("\n=== Step 1: Running Inference ===")
    results_file = run_inference(args.model, args.input, results_file, args.batch_size)
    time_based_file = f"{os.path.splitext(results_file)[0]}_time_based.csv"
    
    # Step 2: Generate visualizations (if requested)
    if args.visualize:
        print("\n=== Step 2: Generating Visualizations ===")
        visualization_dir = f"visualizations/report_{timestamp}"
        html_report = visualize_results(time_based_file, visualization_dir)
        print(f"Visualization report generated: {html_report}")
    
    print(f"\nInference pipeline complete!")
    print(f"- Detailed classification results saved to: {results_file}")
    print(f"- Time-based classification summary saved to: {time_based_file}")
    if args.visualize:
        print(f"- Visualization report: {html_report}")
    
    return results_file

def package_model_pipeline(args):
    """Package a trained model into a self-contained pipeline."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define output path with timestamp
    output_path = args.output or f"pipeline_model/pipeline_{timestamp}.pkl"
    
    print(f"Packaging model pipeline with timestamp: {timestamp}")
    
    # Package the pipeline
    output_path = package_pipeline(
        args.model,
        args.threat_patterns,
        output_path,
        args.status_code_whitelist,
        args.status_code_blacklist
    )
    
    print(f"\nPackaging complete!")
    print(f"- Packaged pipeline saved to: {output_path}")
    print(f"- To use this pipeline, run:")
    print(f"  python -c \"import pickle; pipeline = pickle.load(open('{output_path}', 'rb')); results = pipeline.analyze_logs('your_access.log')\"")
    
    # If status code filters were applied, show example of how to use them
    if args.status_code_whitelist or args.status_code_blacklist:
        filter_type = "whitelist" if args.status_code_whitelist else "blacklist"
        filter_value = args.status_code_whitelist or args.status_code_blacklist
        print(f"- Note: This pipeline has a status code {filter_type} of '{filter_value}'")
        print(f"  Only requests with status codes matching this pattern will be analyzed.")
    
    return output_path

def show_detailed_help():
    """Display detailed help information about the system."""
    help_text = """
Access Log Classification System v0.2.0
Created by Robert Li (Principal Solutions Engineer - APAC)

OVERVIEW:
This system analyzes web server access logs to classify traffic as legitimate or illegitimate,
distinguishing between human users and bots. It uses machine learning to automate the classification process.

WORKFLOW:
1. Log Parsing: Convert raw access logs into structured data
2. Feature Engineering: Extract and calculate features useful for classification
3. LLM Labeling: Use a large language model to label a training dataset
4. Model Training: Fine-tune a model on the labeled data
5. Inferencing: Use the trained model to classify new log entries
6. Visualization: Generate visual reports of classification results
7. Package: Package the trained model into a self-contained pipeline - README in the /pipeline_model folder

PIPELINES:

full-pipeline:
    Run the complete pipeline from log parsing to inference and visualization
    Example: python main.py pipeline --input /path/to/logs/ --samples 500 --visualize

train-pipeline:
    Run only the training part of the pipeline (parsing to model training)
    Example: python main.py train-pipeline --input /path/to/logs/ --samples 500

inference-pipeline:
    Run only the inference part of the pipeline (inference and visualization)
    Example: python main.py inference-pipeline --model ./fine_tuned_model --input logs.csv --visualize

INDIVIDUAL COMMANDS:

parse:
    Parse log files into structured CSV format
    Example: python main.py parse --input /path/to/logs/ --output parsed_logs.csv

enrich:
    Add features to parsed logs for better classification
    Example: python main.py enrich --input parsed_logs.csv --output enriched_logs.csv

label:
    Use an LLM to label a portion of the data for training
    Example: python main.py label --input enriched_logs.csv --output labeled_logs.csv --samples 1000 --llm-provider openai

validate:
    Prepare labeled data for model training
    Example: python main.py validate --input labeled_logs.csv --output training_data.jsonl

train:
    Fine-tune a model on the labeled data
    Example: python main.py train --input training_data.jsonl --output ./my_model --epochs 3

inference:
    Use the trained model to classify new log entries
    Example: python main.py inference --model ./my_model --input new_logs.csv --output results.csv --visualize

visualize:
    Generate visualizations from time-based classification summary
    Example: python main.py visualize --input results_time_based.csv --output-dir ./visualizations

package:
    Package a trained model into a self-contained pipeline for easy deployment
    Example: python main.py package --model ./fine_tuned_model/model_20240520_123456 --output my_pipeline.pkl
    
    Status Code Filtering:
    You can filter logs by status code using regex patterns:
    --status-code-whitelist: Only include logs with status codes matching this pattern
    Example: python main.py package --model ./my_model --status-code-whitelist "50[0-9]"
    
    --status-code-blacklist: Exclude logs with status codes matching this pattern
    Example: python main.py package --model ./my_model --status-code-blacklist "20[0-9]"

PACKAGED PIPELINE:
The packaged pipeline (.pkl file) is a self-contained model that includes:
- Log parsing functionality
- Feature engineering
- Classification model
- Summary generation

LLM PROVIDERS:
The system supports multiple LLM providers for the labeling step:
- sonar: Perplexity's Sonar model (default)
- openai: OpenAI's GPT-4o-mini model
- gemini: Google's Gemini 1.5 Flash model
- anthropic: Anthropic's Claude 3 Haiku model

ENVIRONMENT SETUP:
Make sure to create a .env file with your API keys:
SONAR_API_KEY=your_sonar_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

For more detailed information, refer to the README.md file.
"""
    print(help_text)

def main():
    parser = argparse.ArgumentParser(description='Access Log Classification Modeling')
    
    # Add a help flag that shows detailed help
    parser.add_argument('--help-detailed', action='store_true', help='Show detailed help information')
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run the full pipeline')
    pipeline_parser.add_argument('--input', nargs='+', default=['training_data/'], 
                        help='Path to log file(s) or directory containing log files (default: training_data/)')
    pipeline_parser.add_argument('--batch-size', type=int, default=100000, help='Batch size for processing')
    pipeline_parser.add_argument('--samples', type=int, default=1000, help='Number of samples to label')
    pipeline_parser.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base', help='Base model to fine-tune')
    pipeline_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    pipeline_parser.add_argument('--validation-split', type=float, default=0.1, help='Fraction of data to use for validation')
    pipeline_parser.add_argument('--inference-input', help='Optional separate file for inference (if not provided, uses the enriched logs)')
    pipeline_parser.add_argument('--visualize', action='store_true', help='Generate visualizations after inference')
    pipeline_parser.add_argument('--llm-provider', choices=['sonar', 'openai', 'gemini', 'anthropic'], 
                        default='sonar', help='LLM provider to use for labeling')
    
    # Training pipeline command (parsing to model training)
    train_pipeline_parser = subparsers.add_parser('train-pipeline', help='Run only the training part of the pipeline')
    train_pipeline_parser.add_argument('--input', nargs='+', default=['training_data/'], 
                        help='Path to log file(s) or directory containing log files (default: training_data/)')
    train_pipeline_parser.add_argument('--batch-size', type=int, default=100000, help='Batch size for processing')
    train_pipeline_parser.add_argument('--samples', type=int, default=1000, help='Number of samples to label')
    train_pipeline_parser.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base', help='Base model to fine-tune')
    train_pipeline_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    train_pipeline_parser.add_argument('--validation-split', type=float, default=0.1, help='Fraction of data to use for validation')
    train_pipeline_parser.add_argument('--llm-provider', choices=['sonar', 'openai', 'gemini', 'anthropic'], 
                        default='sonar', help='LLM provider to use for labeling')
    
    # Inference pipeline command (inference and visualization)
    inference_pipeline_parser = subparsers.add_parser('inference-pipeline', help='Run only the inference part of the pipeline')
    inference_pipeline_parser.add_argument('--model', required=True, help='Path to the fine-tuned model')
    inference_pipeline_parser.add_argument('--input', required=True, help='Path to logs CSV file to classify')
    inference_pipeline_parser.add_argument('--output', help='Path to save the classified logs (default: auto-generated)')
    inference_pipeline_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    inference_pipeline_parser.add_argument('--visualize', action='store_true', help='Generate visualizations after inference')
    
    # Parse logs command
    parse_parser = subparsers.add_parser('parse', help='Parse log files')
    parse_parser.add_argument('--input', nargs='+', default=['training_data/'], 
                        help='Path to log file(s) or directory containing log files (default: training_data/)')
    parse_parser.add_argument('--output', required=True, help='Path to save the parsed logs')
    parse_parser.add_argument('--batch-size', type=int, default=100000, help='Batch size for processing')
    
    # Feature engineering command
    feature_parser = subparsers.add_parser('enrich', help='Enrich logs with features')
    feature_parser.add_argument('--input', required=True, help='Path to parsed logs CSV file')
    feature_parser.add_argument('--output', required=True, help='Path to save the enriched logs')
    feature_parser.add_argument('--batch-size', type=int, default=100000, help='Batch size for processing')
    
    # LLM labeling command
    label_parser = subparsers.add_parser('label', help='Label logs with LLM')
    label_parser.add_argument('--input', required=True, help='Path to enriched logs CSV file')
    label_parser.add_argument('--output', required=True, help='Path to save the labeled logs')
    label_parser.add_argument('--samples', type=int, default=1000, help='Number of samples to label')
    label_parser.add_argument('--batch-size', type=int, default=10, help='Batch size for API calls')
    label_parser.add_argument('--llm-provider', choices=['sonar', 'openai', 'gemini', 'anthropic'], 
                        default='sonar', help='LLM provider to use for labeling')
    
    # Create validation set command
    validation_parser = subparsers.add_parser('validate', help='Create validation set')
    validation_parser.add_argument('--input', required=True, help='Path to labeled logs CSV file')
    validation_parser.add_argument('--output', required=True, help='Path to save the training data')
    validation_parser.add_argument('--batch-size', type=int, default=50000, help='Batch size for processing')
    
    # Train model command
    train_parser = subparsers.add_parser('train', help='Train model')
    train_parser.add_argument('--input', required=True, help='Path to training data JSONL file')
    train_parser.add_argument('--output', required=True, help='Directory to save the fine-tuned model')
    train_parser.add_argument('--model', default='deepseek-ai/deepseek-coder-1.3b-base', help='Base model to fine-tune')
    train_parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    train_parser.add_argument('--validation-split', type=float, default=0.1, help='Fraction of data to use for validation')

    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference and generate time-based summary')
    inference_parser.add_argument('--model', required=True, help='Path to the fine-tuned model')
    inference_parser.add_argument('--input', required=True, help='Path to logs CSV file to classify')
    inference_parser.add_argument('--output', required=True, help='Path to save the classified logs (a time-based summary will also be generated)')
    inference_parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    inference_parser.add_argument('--visualize', action='store_true', help='Generate visualizations after inference')
    
    # Visualization command
    visualize_parser = subparsers.add_parser('visualize', help='Generate visualizations from time-based summary')
    visualize_parser.add_argument('--input', required=True, help='Path to time-based classification summary CSV')
    visualize_parser.add_argument('--output-dir', default='visualizations', help='Directory to save visualizations')

    # Package pipeline command
    package_parser = subparsers.add_parser('package', help='Package a trained model into a self-contained pipeline')
    package_parser.add_argument('--model', required=True, help='Path to the trained model directory')
    package_parser.add_argument('--threat-patterns', default='src/threat_int/malicious_req_patterns.txt', 
                        help='Path to the malicious request patterns file')
    package_parser.add_argument('--output', help='Path to save the packaged pipeline (default: auto-generated)')
    package_parser.add_argument('--status-code-whitelist', help='Regex pattern for status codes to include (only these will be processed)')
    package_parser.add_argument('--status-code-blacklist', help='Regex pattern for status codes to exclude (all except these will be processed)')
    
    args, unknown = parser.parse_known_args()
    
    # Check if detailed help is requested
    if args.help_detailed or (len(sys.argv) == 1 and not hasattr(args, 'command')):
        show_detailed_help()
        return
    
    # Create necessary directories
    setup_directories()
    
    # Execute the appropriate command
    if args.command == 'pipeline':
        run_full_pipeline(args)
    elif args.command == 'train-pipeline':
        run_training_pipeline(args)
    elif args.command == 'inference-pipeline':
        run_inference_pipeline(args)
    elif args.command == 'parse':
        parse_logs(args.input, args.output, args.batch_size)
    elif args.command == 'enrich':
        enrich_logs(args.input, args.output, args.batch_size)
    elif args.command == 'label':
        label_logs(args.input, args.output, args.samples, args.batch_size, args.llm_provider)
    elif args.command == 'validate':
        create_validation_set(args.input, args.output, args.batch_size)
    elif args.command == 'train':
        train_model(args.input, args.output, args.model, args.epochs, args.validation_split)
    elif args.command == 'inference':
        results_file = run_inference(args.model, args.input, args.output, args.batch_size)
        if args.visualize:
            time_based_file = f"{os.path.splitext(results_file)[0]}_time_based.csv"
            visualization_dir = f"visualizations/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            html_report = visualize_results(time_based_file, visualization_dir)
            print(f"Visualization report generated: {html_report}")
    elif args.command == 'visualize':
        html_report = visualize_results(args.input, args.output_dir)
        print(f"Visualization report generated: {html_report}")
    elif args.command == 'package':
        package_model_pipeline(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
