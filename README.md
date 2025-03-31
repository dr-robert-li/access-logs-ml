# Access Log Classification Modeling
## v0.4
### Robert Li (Principal Solutions Engineer - APAC)

This system analyzes web server access logs to classify traffic as legitimate or illegitimate, distinguishing between human users and bots. It uses machine learning to automate the classification process.

## Overview

The system consists of several components:

1. **Log Parsing**: Convert raw access logs into structured data
2. **Feature Engineering**: Extract and calculate features useful for classification
3. **LLM Labeling**: Use a large language model to label a training dataset
4. **Model Training**: Fine-tune a model on the labeled data
5. **Inferencing**: Use the trained model to classify new log entries and generate time-based summaries

## Requirements

Create your python environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## LLM Provider Options

The system now supports multiple LLM providers for the labeling step. You can choose from:

1. **Perplexity Sonar** (default): Uses Perplexity's Sonar model
2. **OpenAI GPT-4o-mini**: Uses OpenAI's GPT-4o-mini model
3. **Google Gemini 2.0 Flash**: Uses Google's Gemini 2.0 Flash model
4. **Anthropic Claude 3.5 Haiku**: Uses Anthropic's Claude 3.5 Haiku model

To use a specific provider, add the `--llm-provider` argument:

```bash
# Label with OpenAI
python main.py label --input enriched_logs.csv --output labeled_logs.csv --llm-provider openai

# Use Gemini in the full pipeline
python main.py pipeline --input /path/to/logs/ --llm-provider gemini

# Use Claude for labeling
python main.py label --input enriched_logs.csv --output labeled_logs.csv --llm-provider anthropic
```

### API Keys

You'll need to set up API keys for the providers you want to use. Add them to your `.env` file - a `.env_sample` has been provided:

```
SONAR_API_KEY=your_sonar_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### Provider Comparison

Each provider has different strengths and rate limits:

| Provider | Model | Default Rate Limit | Strengths |
|----------|-------|------------|-----------|
| Perplexity | Sonar | ~125 req/min | Good at structured outputs |
| OpenAI | GPT-4o-mini | ~300 req/min | Strong reasoning capabilities |
| Google | Gemini 2.0 Flash Lite | ~600 req/min | Fast and cost-effective |
| Anthropic | Claude 3.5 Haiku | ~200 req/min | Excellent at following instructions |

The system automatically adjusts rate limiting based on the provider you choose. These rate limits are based on the assumption that you are a paying user on at least Tier 1 of each vendor's API plans.

You can modify `rate_limit_delay` in the `llm_labeling.py` script to adjust the delay between API calls if required.

## Usage Options

This system can be used in several ways:

1. **Full Pipeline**: Run the entire process from log parsing to inference with a single command
2. **Training Pipeline**: Run only the training part (parsing to model training)
3. **Inference Pipeline**: Run only the inference part (inference and visualization)
4. **Individual Components**: Run each step separately for more control

### Getting Help

For detailed help about all available commands and options:

```bash
python main.py --help-detailed
```

### Option 1: Full Pipeline

Run the entire pipeline with a single command:

```bash
python main.py pipeline --input /path/to/logs/ --samples 1000 --epochs 3
```

Options:
- `--input`: Path to log file(s) or directory containing log files
- `--batch-size`: Batch size for processing (default: 100000)
- `--samples`: Number of samples to label (default: 1000)
- `--model`: Base model to fine-tune (default: deepseek-ai/deepseek-coder-1.3b-base)
- `--epochs`: Number of training epochs (default: 3)
- `--validation-split`: Fraction of data to use for validation (default: 0.1)
- `--inference-input`: Optional separate file for inference (if not provided, uses the enriched logs)

### Option 2: Training Pipeline

Run only the training part of the pipeline (parsing to model training):

```bash
python main.py train-pipeline --input /path/to/logs/ --samples 1000 --epochs 3 --validation-split 0.1
```

Options:
- `--input`: Path to log file(s) or directory containing log files (default: training_data/)
- `--batch-size`: Batch size for processing (default: 100000)
- `--samples`: Number of samples to label (default: 1000)
- `--model`: Base model to fine-tune (default: deepseek-ai/deepseek-coder-1.3b-base)
- `--epochs`: Number of training epochs (default: 3)
- `--validation-split`: Fraction of data to use for validation (default: 0.1)
- `--llm-provider`: LLM provider to use for labeling (default: sonar)

### Option 3: Inference Pipeline

Run only the inference part of the pipeline (inference and visualization):

```bash
python main.py inference-pipeline --model ./fine_tuned_model --input logs.csv --visualize
```

For information on the input `csv` file refer to the inferencing step (Step 6) below.

Options:
- `--model`: Path to the fine-tuned model
- `--input`: Path to logs CSV file to classify
- `--output`: Path to save the classified logs (default: auto-generated)
- `--batch-size`: Batch size for inference (default: 32)
- `--status-code-whitelist`: Regex pattern for status codes to include (only these will be processed)
- `--status-code-blacklist`: Regex pattern for status codes to exclude (all except these will be processed)
- `--visualize`: Generate visualizations after inference
- `--summary`: Generate a comprehensive analysis summary with attack statistics and recommendations

### Option 4: Individual Components

#### 1. Parse Log Files

```bash
python main.py parse --input /path/to/logs/ --output parsed_logs.csv
```

Options:
- `--input`: Path to log file(s) or directory containing log files
- `--output`: Path to save the parsed logs
- `--batch-size`: Number of log entries to process in each batch (default: 100000)

#### 2. Feature Engineering

```bash
python main.py enrich --input parsed_logs.csv --output enriched_logs.csv
```

Options:
- `--input`: Path to parsed logs CSV file
- `--output`: Path to save the enriched logs
- `--batch-size`: Number of rows to process in each chunk (default: 100000)

#### 3. Label Data with LLM

```bash
python main.py label --input enriched_logs.csv --output labeled_logs.csv --samples 1000
```

Options:
- `--input`: Path to enriched logs CSV file
- `--output`: Path to save the labeled logs
- `--samples`: Number of samples to label (default: 1000)
- `--batch-size`: Batch size for API calls (default: 10)

#### 4. Create Validation Set

```bash
python main.py validate --input labeled_logs.csv --output training_data.jsonl
```

Options:
- `--input`: Path to labeled logs CSV file
- `--output`: Path to save the training data
- `--batch-size`: Number of rows to process in each chunk (default: 50000)

#### 5. Train Model

```bash
python main.py train --input training_data.jsonl --output ./fine_tuned_model --epochs 3 --validation-split 0.1
```

Options:
- `--input`: Path to training data JSONL file
- `--output`: Directory to save the fine-tuned model
- `--model`: Base model to fine-tune (default: deepseek-ai/deepseek-coder-1.3b-base)
- `--epochs`: Number of training epochs (default: 3)
- `--validation-split`: Fraction of data to use for validation (default: 0.1)

##### Validation Metrics

The training process now reports validation loss after each epoch, allowing you to monitor how well the model generalizes to unseen data. The system will:

1. Automatically evaluate the model on the validation set after each epoch
2. Save the best model based on validation loss
3. Report final validation metrics after training
4. Save evaluation results to a JSON file in the model directory

##### Interpreting Validation Results

- **Decreasing training and validation loss**: Model is learning effectively
- **Decreasing training loss but increasing validation loss**: Model is overfitting
- **High validation loss**: Model may not generalize well to new data

You can find the validation results in the `eval_results.json` file in your model directory.

#### 6. Run Inference

```bash
python main.py inference --model ./fine_tuned_model --input new_logs.csv --output classified_logs.csv --visualize
```

Options:
- `--model`: Path to the fine-tuned model
- `--input`: Path to logs CSV file to classify
- `--output`: Path to save the classified logs (a time-based summary will also be generated)
- `--batch-size`: Batch size for inference (default: 32)
- `--visualize`: Visualize the classified logs (default: False)

##### Input CSV Format for Inference

When running inference, your input CSV file should contain the following columns:

| Column | Description | Example |
|--------|-------------|---------|
| ip | IP address of the client | 192.168.1.100 |
| timestamp | Date and time of the request | 2023-10-15 08:45:23 |
| method | HTTP method | GET |
| path | Request path | /index.php |
| protocol | HTTP protocol version | HTTP/1.1 |
| status_code | HTTP status code | 200 |
| user_agent | User agent string | Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 |
| country | Country of origin for the IP | US |
| asn | Autonomous System Number | AS15169 |
| requests_per_minute | Request frequency from this IP | 2.5 |
| time_diff | Time in seconds since previous request from this IP | 24.3 |
| has_user_agent | Boolean indicating if user agent is present | TRUE |
| is_old_browser | Boolean indicating if it's an old browser | FALSE |
| is_known_bot | Boolean indicating if it's a known bot | FALSE |

##### Example CSV:

```csv
ip,timestamp,method,path,protocol,status_code,user_agent,country,asn,requests_per_minute,time_diff,has_user_agent,is_old_browser,is_known_bot
192.168.1.100,2023-10-15 08:45:23,GET,/index.php,HTTP/1.1,200,"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",US,AS15169,2.5,24.3,TRUE,FALSE,FALSE
8.8.8.8,2023-10-15 08:46:12,GET,/wp-login.php,HTTP/1.1,401,"Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)",US,AS15169,5.2,12.1,TRUE,FALSE,TRUE
203.0.113.42,2023-10-15 08:47:05,POST,/xmlrpc.php,HTTP/1.1,403,"Mozilla/5.0 (Windows NT 10.0; Win64; x64)",RU,AS9009,45.8,0.5,TRUE,FALSE,FALSE
198.51.100.23,2023-10-15 08:48:32,GET,/wp-admin/install.php,HTTP/1.1,404,"python-requests/2.25.1",CN,AS4134,120.3,0.2,TRUE,FALSE,FALSE
```

Note: If you're starting with raw access logs, you should first run the parsing and feature engineering steps to generate a properly formatted CSV. 

Refer to Step 1 above for more details, but briefly:

```bash
# Parse logs
python main.py parse --input /path/to/logs/ --output parsed_logs.csv

# Enrich with features
python main.py enrich --input parsed_logs.csv --output enriched_logs.csv
```

The enriched logs (indicated above by `enriched_logs.csv`) will contain all the necessary columns for inference.

## Running Scripts Directly

Below are some example workflows:

```bash
# Run the full pipeline with validation
python main.py pipeline --input /var/log/apache2/ --samples 500 --epochs 3 --validation-split 0.15 --visualize

# Or run the training pipeline only with validation
python main.py train-pipeline --input /var/log/apache2/ --samples 500 --epochs 3 --validation-split 0.15

# Then later run inference with the trained model
python main.py inference-pipeline --model ./fine_tuned_model/model_20240615_123456 --input new_logs.csv --visualize

# Basic inference with status code filtering
python main.py inference --model ./fine_tuned_model --input logs.csv --output classified_logs.csv --status-code-whitelist "^(404|500)$"

# Generate comprehensive summary along with classification
python main.py inference --model ./fine_tuned_model --input logs.csv --output classified_logs.csv --summary

# Process in smaller batches for large files
python main.py inference --model ./fine_tuned_model --input large_logs.csv --output classified_logs.csv --batch-size 16 --summary

# Or run each step individually
python main.py parse --input /var/log/apache2/ --output parsed_logs.csv
python main.py enrich --input parsed_logs.csv --output enriched_logs.csv
python main.py label --input enriched_logs.csv --output labeled_logs.csv --samples 500
python main.py validate --input labeled_logs.csv --output training_data.jsonl
python main.py train --input training_data.jsonl --output ./my_model --epochs 3 --validation-split 0.15
python main.py inference --model ./my_model --input new_logs.csv --output results.csv --visualize
```

You can also run each script directly from the `src` directory with the same flags e.g.:

```bash
# Train model
python src/model_training.py --input training_data.jsonl --output ./fine_tuned_model --epochs 3

# Run inference
python src/inferencing.py --model ./fine_tuned_model --input new_logs.csv --output classified_logs.csv
```

## Using Compressed Model Files

The trained models are also exported as compressed pickle (.pkl) files for deployment in other environments. This allows you to train a model once and use it in multiple locations without needing to retrain.

### Importing and Using a Model Elsewhere

To use your exported model in another environment:

1. **Copy the .pkl file** to your target environment
2. **Install required dependencies**:

   ```bash
   pip install pandas scikit-learn joblib
   ```

3. **Load and use the model**:

   ```python
   import joblib
   import pandas as pd
   
   # Load the model
   model = joblib.load('model.pkl')
   
   # Prepare your data (must have the same features as training data)
   data = pd.read_csv('new_logs.csv')
   
   # Make predictions
   predictions = model.predict(data)
   
   # Get prediction probabilities (confidence scores)
   confidence = model.predict_proba(data).max(axis=1)
   ```

### Self-Contained Model

For even easier deployment, you can package your trained model into a self-contained model that includes all necessary components for end-to-end log analysis:

1. **Create a packaged model**:

   ```bash
   python main.py package --model ./fine_tuned_model/model_20240520_123456
   ```

Options:
- `--model`: Path to the trained model directory (required)
- `--threat-patterns`: Path to the malicious request patterns file (default: 'src/threat_int/malicious_req_patterns.txt')
- `--output`: Path to save the packaged pipeline (default: auto-generated)
- `--status-code-whitelist`: Regex pattern for status codes to include (only these will be processed)
- `--status-code-blacklist`: Regex pattern for status codes to exclude (all except these will be processed)


Replace the model folder with an appropriately timestamped one of your own creation.

2. **Use the packaged model**:

```python
import pickle

# Load the pipeline
with open('pipeline_model/pipeline_20240520_123456.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Analyze logs directly from a file
results = pipeline.analyze_logs('path/to/access.log')

# Generate a summary report
summary = pipeline.generate_summary(results)
print(summary)
```

Replace the model folder with an appropriately timestamped one of your own creation.

The packaged model includes:
- Log parsing functionality
- Feature engineering
- Classification model
- Summary generation

For more detailed information on using packaged pipelines, refer to the README.md file in the `/pipeline_model` directory.

### Model Compatibility Considerations

When using exported models in different environments, keep in mind:

1. **Feature consistency**: When inferencing data using the raw model (no pipeline) your input data must have the same features that were used during training i.e. enriched Apache HTTPD style server access logs (use the above log parsing, labeling and enriching functions) - the packaged pipeline model will automatically handle feature engineering for you at a lower fidelity.
2. **Memory requirements**: Sufficient memory required for loading the model
3. **Processing power**: Classification speed depends on the available CPU resources

When using this model for production deployments, consider containerizing your application with Docker to ensure consistent environments.

## Handling Large Datasets

This system is designed to handle very large log files by processing data in chunks. For extremely large datasets, consider the following options:

### Batch Processing

The system will automatically detect whether your system is CUDA enabled, and use the appropriate backend (PyTorch or TensorFlow) for training and inference, or is using MPS for Apple Silicon and appropriately uses the Metal backend and fallback to CPU if memory is insufficient (note that MPS is typically very memory constrained), and if neither are available, will use the CPU by default.

All scripts support processing data in chunks to minimize memory usage:

```bash
# Parse large log files with a smaller batch size
python main.py parse --input /var/log/huge_logs/ --output parsed_logs.csv --batch-size 50000

# Process features in chunks
python main.py enrich --input parsed_logs.csv --output enriched_logs.csv --batch-size 50000

# Sample a subset for labeling
python main.py label --input enriched_logs.csv --output labeled_logs.csv --samples 2000 --batch-size 10

# Create validation set in chunks
python main.py validate --input labeled_logs.csv --output training_data.jsonl --batch-size 50000

# Run inference in batches
python main.py inference --model ./fine_tuned_model --input new_logs.csv --output results.csv --batch-size 32
```

### Suggested Models for Fine-tuning

Depending on your specific needs and available computational resources, you can choose from the following models (it is strongly suggested that you use a model with 3B parameters or less for fine-tuning on a GPU with 12GB of memory or less and a model with 1.5B parameters or less for fine-tuning using MPS on Apple Silicon):

#### Smaller Models (for limited resources)
- `distilgpt2` (~355M parameters)
- `gpt2` (~124M parameters)
- `EleutherAI/pythia-410m` (~410M parameters)
- `bigscience/bloom-560m` (~560M parameters, Multi-lingual)

#### Medium-sized Models
- `facebook/opt-1.3b` (~1.3B parameters)
- `deepseek-ai/deepseek-coder-1.3b-base` (~1.3B parameters, High Performance, default)
- `EleutherAI/pythia-1.4b` (~1.4B parameters)
- `bigscience/bloom-1b7` (~1.7B parameters, Multi-lingual)
- `stabilityai/stablelm-3b-4e1t` (~3B parameters, Memory Efficient)

#### Larger Models (for better performance)
- `deepseek-ai/deepseek-coder-6.7b-base` (~6.7B parameters)
- `meta-llama/Llama-2-7b-hf` (~7B parameters)
- `mistralai/Mistral-7B-v0.1` (~7B parameters)
- `stabilityai/stablelm-base-alpha-7b-v2` (~7B parameters)
- `EleutherAI/pythia-12b` (~12B parameters)

To use a different model, specify it with the `--model` parameter:

```bash
python main.py train --input training_data.jsonl --output ./my_model --model meta-llama/Llama-2-7b-hf
```

Note: Some models may require authentication or special access. Ensure you have the necessary permissions before using them.

## Sample Output

### Detailed Classification Results

After running the inference step, the `classified_logs.csv` file will contain the original log data enriched with classification results. Here's a sample of what the output might look like:

| ip | timestamp | method | path | status_code | user_agent | country | asn | requests_per_minute | time_diff | classification | confidence | reasoning |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 192.168.1.100 | 2023-10-15 08:45:23 | GET | /index.php | 200 | Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 | US | AS15169 | 2.5 | 24.3 | legitimate_human | 0.95 | Normal browsing pattern with standard browser, reasonable request rate, and no suspicious indicators. |
| 8.8.8.8 | 2023-10-15 08:46:12 | GET | /wp-login.php | 401 | Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html) | US | AS15169 | 5.2 | 12.1 | legitimate_bot | 0.87 | Identified as Googlebot with appropriate user agent and reasonable request pattern. |
| 203.0.113.42 | 2023-10-15 08:47:05 | POST | /xmlrpc.php | 403 | Mozilla/5.0 (Windows NT 10.0; Win64; x64) | RU | AS9009 | 45.8 | 0.5 | illegitimate_human | 0.78 | Attempting to access xmlrpc.php with high request frequency from a high-risk country. |
| 198.51.100.23 | 2023-10-15 08:48:32 | GET | /wp-admin/install.php | 404 | python-requests/2.25.1 | CN | AS4134 | 120.3 | 0.2 | illegitimate_bot | 0.92 | Extremely high request rate targeting WordPress installation page from a known bad IP with suspicious user agent. |

The classification results include:
- **classification**: The category assigned to the log entry
- **confidence**: A value between 0 and 1 indicating the model's confidence in the classification
- **reasoning**: A brief explanation of why the entry was classified as it was

### Time-Based Classification Summary

Additionally, a time-based summary file (`classified_logs_time_based.csv`) will be automatically generated, which aggregates the classifications by hour. Here's a sample of what this summary might look like:

| hour | total_requests | legitimate_requests | illegitimate_requests | human_requests | bot_requests | legitimate_human_requests | legitimate_bot_requests | illegitimate_human_requests | illegitimate_bot_requests | avg_confidence | legitimate_pct | illegitimate_pct | human_pct | bot_pct | legitimate_human_pct | legitimate_bot_pct | illegitimate_human_pct | illegitimate_bot_pct |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2023-10-15 08:00:00 | 1245 | 876 | 369 | 723 | 522 | 612 | 264 | 111 | 258 | 0.87 | 70.36 | 29.64 | 58.07 | 41.93 | 49.16 | 21.20 | 8.92 | 20.72 |
| 2023-10-15 09:00:00 | 1532 | 1102 | 430 | 892 | 640 | 782 | 320 | 110 | 320 | 0.89 | 71.93 | 28.07 | 58.22 | 41.78 | 51.04 | 20.89 | 7.18 | 20.89 |
| 2023-10-15 10:00:00 | 1876 | 1345 | 531 | 1125 | 751 | 945 | 400 | 180 | 351 | 0.86 | 71.70 | 28.30 | 59.97 | 40.03 | 50.37 | 21.32 | 9.59 | 18.71 |
| 2023-10-15 11:00:00 | 2134 | 1452 | 682 | 1280 | 854 | 1050 | 402 | 230 | 452 | 0.88 | 68.04 | 31.96 | 60.00 | 40.00 | 49.20 | 18.84 | 10.78 | 21.18 |

This time-based summary provides several useful metrics:
- **Total requests per hour**: Shows traffic volume patterns throughout the day
- **Legitimate vs. Illegitimate**: Shows the breakdown of traffic by legitimacy
- **Human vs. Bot**: Shows the breakdown of traffic by agent type
- **Detailed categories**: Shows all four classification categories
- **Average confidence**: Shows the model's confidence in its classifications
- **Percentage breakdowns**: Shows the proportion of each traffic type

This summary is particularly useful for:
- Identifying traffic patterns and trends over time
- Detecting unusual spikes in illegitimate traffic
- Understanding the composition of your web traffic
- Planning security measures based on traffic patterns
- Reporting on traffic quality metrics

## Project Structure

```
access-log-classification/
├── main.py                          # Main entry point and command handler
├── requirements.txt                 # Dependencies
├── .env                             # Environment variables for API keys
├── .env_sample                      # Sample environment file
├── README.md                        # Documentation
├── threat_int/                      # Threat data for pattern matching
│   └── malicious_req_patterns.txt   # Patterns for identifying malicious requests
├── training_data/                   # Default directory for input Apache HTTPD style server access log files
├── training_data_source/            # Directory for storing training data not being processed immediately
├── src/
│   ├── __init__.py                
│   ├── parse_logs.py                # Log parsing module - converts raw logs to structured data
│   ├── feature_engineering.py       # Feature engineering module - extracts and calculates features
│   ├── llm_labeling.py              # LLM labeling module - uses LLMs to label training data
│   ├── create_validation_set.py     # Validation set creation - prepares data for model training
│   ├── model_training.py            # Model training module - fine-tunes models on labeled data
│   ├── inferencing.py               # Inference module - uses trained model to classify new logs
│   ├── visualizer.py                # Visualization module - generates reports and charts
│   ├── pipeline.py                  # Orchestrates the full processing pipeline
│   ├── train_pipeline.py            # Orchestrates the training pipeline
│   ├── inference_pipeline.py        # Orchestrates the inference pipeline
│   └── model_packaging.py           # Creates self-contained model packages for deployment
├── logs/                            # Training logs directory
├── data/                            # Directory for processed data
├── fine_tuned_model/                # Directory for saved models
│   └── model_YYYYMMDD_HHMMSS/       # Timestamped model directories
│       ├── config.json              # Model configuration
│       ├── pytorch_model.bin        # Model weights
│       ├── tokenizer/               # Tokenizer files
│       └── eval_results.json        # Validation metrics
├── pipeline_model/                  # Directory for packaged pipeline models
│   └── pipeline_YYYYMMDD_HHMMSS.pkl # Self-contained model packages
└── visualizations/                  # Directory for visualization reports
    ├── report_YYYYMMDD_HHMMSS.html  # HTML visualization reports
    └── figures/                     # Individual chart images
```

## Interpreting Results

The classification output includes:
- **Classification**: legitimate_human, legitimate_bot, illegitimate_human, or illegitimate_bot
- **Confidence**: A score between 0.0 and 1.0 indicating confidence in the classification
- **Reasoning**: A brief explanation of why the entry was classified as it was

The time-based summary provides an overview of traffic patterns by hour, showing the distribution of different traffic types over time.

## Advanced Usage

### Custom Feature Engineering

You can modify `src/feature_engineering.py` to add custom features specific to your environment.

### Model Selection

Change the base model by using the `--model` parameter in the training step.

## Memory Optimization

The system includes memory usage tracking and optimization for handling large datasets:

- Processes data in configurable chunks
- Reports memory usage during processing
- Cleans up resources after each processing step
- Uses efficient data structures for large datasets

## Visualizing Results

The system can automatically generate visualizations from the time-based classification summary:

```bash
# Generate visualizations from a time-based summary
python main.py visualize --input results_time_based.csv --output-dir ./my_visualizations
```

You can also generate visualizations as part of the inference step:

```bash
# Run inference and generate visualizations
python main.py inference --model ./fine_tuned_model --input new_logs.csv --output results.csv --visualize
```

Or include visualizations in the full pipeline:

```bash
# Run the full pipeline with visualization
python main.py pipeline --input /var/log/apache2/ --samples 500 --epochs 3 --visualize
```

The visualizations include:
- Traffic composition over time (line chart)
- Total traffic volume (bar chart)
- Stacked traffic composition (area chart)
- Model confidence over time (line chart)

All visualizations are compiled into an HTML report for easy viewing and sharing.

## Distributed Processing

For extremely large datasets (multiple GB), consider:

1. Splitting input files by date ranges
2. Processing each split in parallel on different machines
3. Combining results afterward

### Database Integration

For production environments with continuous log analysis:

1. Store parsed logs in a database (PostgreSQL, MongoDB)
2. Process new logs incrementally
3. Use database queries for feature engineering
4. Sample from the database for model training

## License

MIT License

Copyright (c) 2025 Robert Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

## Disclaimer

This software is provided "as is" without warranty of any kind, either express or implied, including but not limited to the implied warranties of merchantability, fitness for a particular purpose, or non-infringement. The authors or copyright holders shall not be liable for any claim, damages, or other liability, whether in an action of contract, tort, or otherwise, arising from, out of, or in connection with the software or the use or other dealings in the software.

The classification results provided by this system should be used as guidance only and not as definitive security decisions. Always verify results and use in conjunction with other security measures.