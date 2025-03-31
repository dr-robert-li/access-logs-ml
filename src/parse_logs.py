import pandas as pd
import re
import argparse
import os
import psutil
from datetime import datetime
from glob import glob
from tqdm import tqdm

# Define status codes to exclude
EXCLUDED_RESP_CODES = {301, 302, 303, 304, 305, 307, 308}
# EXCLUDED_RESP_CODES = set(range(100, 500)) | set(range(510, 600))  # Option to exclude all response codes from training except 500-509

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def parse_phpfpm_logs(log_file, batch_size=100000):
    """
    Parse PHP-FPM access logs into structured data, processing in batches.
    
    Args:
        log_file: Path to the log file
        batch_size: Number of lines to process in each batch
        
    Returns:
        List of dictionaries containing parsed log entries
    """
    data = []
    total_lines = 0
    skipped_requests = 0
    
    # Count total lines for progress bar (optional, can be skipped for very large files)
    try:
        with open(log_file, 'r') as f:
            for _ in f:
                total_lines += 1
    except:
        # If file is too large to count lines, use a default
        total_lines = None
    
    # Pattern for standard Apache/Nginx access logs
    # This pattern matches: IP - - [timestamp] "METHOD /path HTTP/1.x" status size "referer" "user-agent"
    pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[([^\]]+)\] "([^"]*)" (\d+) ([^ ]+) "([^"]*)" "([^"]*)"'
    
    # Process file in batches
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    with open(log_file, 'r') as f:
        batch = []
        batch_exclusions = 0
        for i, line in enumerate(tqdm(f, total=total_lines, desc=f"Parsing {os.path.basename(log_file)}")):
            match = re.match(pattern, line.strip())
            if match:
                ip, timestamp, request, status_code, response_size, referer, user_agent = match.groups()
                
                # Parse request components
                request_parts = request.split()
                method = request_parts[0] if request_parts else ""
                path = request_parts[1] if len(request_parts) > 1 else ""
                protocol = request_parts[2] if len(request_parts) > 2 else ""
                
                # Convert timestamp to datetime
                try:
                    timestamp_dt = datetime.strptime(timestamp, "%d/%b/%Y:%H:%M:%S %z")
                except ValueError:
                    timestamp_dt = None
                
                # Handle case where response_size is "-"
                try:
                    response_size_int = int(response_size)
                except ValueError:
                    response_size_int = 0
                
                # Skip redirection status codes
                status_code_int = int(status_code)
                if status_code_int not in EXCLUDED_RESP_CODES:
                    batch.append({
                        'ip': ip,
                        'timestamp': timestamp_dt,
                        'method': method,
                        'path': path,
                        'protocol': protocol,
                        'status_code': status_code_int,
                        'response_size': response_size_int,
                        'referer': referer,
                        'user_agent': user_agent,
                        'raw_log': line.strip()
                    })
                else:
                    batch_exclusions += 1
            else:
                print(f"Warning: Could not parse line {i+1}: {line.strip()}")
            
            # Process batch when it reaches the batch size
            if len(batch) >= batch_size:
                data.extend(batch)
                
                # Report memory usage and redirection stats every batch
                current_memory = get_memory_usage()
                skipped_requests += batch_exclusions
                print(f"Processed {len(data)} entries. Skipped {batch_exclusions} redirections in this batch (total skipped: {skipped_requests}).")
                print(f"Memory usage: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
                
                batch = []
                batch_exclusions = 0
        
        # Add any remaining entries
        if batch:
            data.extend(batch)
            skipped_requests += batch_exclusions
    
    # Final memory report
    final_memory = get_memory_usage()
    print(f"Parsing complete. Final memory usage: {final_memory:.2f} MB (Δ: {final_memory - initial_memory:.2f} MB)")
    print(f"Total entries processed: {len(data)}. Total entries skipped: {skipped_requests}.")

    return data

def save_batch_to_csv(data, output_file, mode='w'):
    """
    Save a batch of data to CSV file.
    
    Args:
        data: List of dictionaries to save
        output_file: Path to the output CSV file
        mode: File open mode ('w' for write, 'a' for append)
    """
    if not data:
        return
        
    df = pd.DataFrame(data)
    
    if mode == 'w':
        # Write with headers for the first batch
        df.to_csv(output_file, index=False)
    else:
        # Append without headers for subsequent batches
        df.to_csv(output_file, mode='a', header=False, index=False)

def parse_logs(input_paths, output_file, batch_size=100000):
    """
    Main function to parse logs from multiple input paths.
    
    Args:
        input_paths: List of paths to log files or directories
        output_file: Path to save the parsed logs
        batch_size: Number of log entries to process in each batch
        
    Returns:
        int: Total number of entries processed
    """
    initial_memory = get_memory_usage()
    print(f"Starting with memory usage: {initial_memory:.2f} MB")
    
    # Process each input path
    first_batch = True
    total_entries = 0
    
    for input_path in input_paths:
        if os.path.isdir(input_path):
            # If directory, process all .log files
            log_files = glob(os.path.join(input_path, '*.log'))
        elif os.path.isfile(input_path):
            # If file, process it directly
            log_files = [input_path]
        else:
            print(f"Warning: {input_path} is not a valid file or directory. Skipping.")
            continue
        
        for log_file in log_files:
            print(f"Processing {log_file}...")
            log_data = parse_phpfpm_logs(log_file, batch_size=batch_size)
            
            # Save this batch
            print(f"Saving {len(log_data)} entries to {output_file}...")
            mode = 'w' if first_batch else 'a'
            save_batch_to_csv(log_data, output_file, mode=mode)
            
            first_batch = False
            total_entries += len(log_data)
            print(f"Processed {len(log_data)} entries from {log_file}")
            
            # Report memory usage
            current_memory = get_memory_usage()
            print(f"Memory usage after processing {log_file}: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    
    # Final memory report
    final_memory = get_memory_usage()
    print(f"Total: Parsed {total_entries} log entries and saved to {output_file}")
    print(f"Final memory usage: {final_memory:.2f} MB (Δ: {final_memory - initial_memory:.2f} MB)")
    
    return total_entries

def main():
    parser = argparse.ArgumentParser(description='Parse web server access logs')
    parser.add_argument('--input', nargs='+', default=['training_data/'], 
                        help='Path to log file(s) or directory containing log files (default: training_data/)')
    parser.add_argument('--output', default='parsed_logs.csv', help='Path to save the parsed logs')
    parser.add_argument('--batch-size', type=int, default=100000, help='Number of log entries to process in each batch')
    args = parser.parse_args()
    
    parse_logs(args.input, args.output, args.batch_size)

if __name__ == "__main__":
    main()