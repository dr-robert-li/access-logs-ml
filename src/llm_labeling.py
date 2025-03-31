import requests
import json
import time
import os
import re
import numpy as np  
import argparse 
import pandas as pd
import psutil
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def create_labeling_prompt(log_entry):
    """Create a prompt for the LLM to classify a log entry."""
    prompt = f"""
    Analyze this web server log entry and classify it as:
    1. Legitimate human (normal user browsing)
    2. Legitimate bot (search engine crawler, monitoring tool)
    3. Illegitimate human (manual hacking attempt)
    4. Illegitimate bot (automated attack, malicious crawler)
    
    Log details:
    - IP: {log_entry['ip']}
    - Timestamp: {log_entry['timestamp']}
    - Request: {log_entry['method']} {log_entry['path']} {log_entry['protocol']}
    - Status Code: {log_entry['status_code']}
    - User Agent: {log_entry['user_agent']}
    - Country: {log_entry['country']}
    - ASN: {log_entry['asn']}
    - Organization: {log_entry['organization']}
    - Request frequency: {log_entry['requests_per_minute']:.2f} requests/minute
    - Time since previous request: {log_entry['time_diff'] if pd.notna(log_entry['time_diff']) else 'N/A'} seconds
    
    Additional indicators:
    - IP reputation score: {log_entry['ip_reputation_score']}
    - ASN reputation score: {log_entry['asn_reputation_score']}
    - Combined reputation score: {log_entry['combined_reputation_score']}
    - Is known bad IP: {log_entry['ip_is_known_bad']}
    - Is from high-risk country: {log_entry['is_high_risk_country']}
    - Is from high-risk ASN: {log_entry['is_high_risk_asn']}
    - Is part of potential DoS: {log_entry['potential_dos']}
    - Is potential scanner: {log_entry['potential_scanner']}
    - Is during traffic spike: {log_entry['during_traffic_spike']}
    - Is from coordinated ASN: {log_entry['from_coordinated_asn']}
    - Targets high-error path: {log_entry['to_high_error_path']}
    - Has suspicious path patterns: {log_entry['has_suspicious_patterns']}
    - Has user agent: {log_entry['has_user_agent']}
    - Is known bot: {log_entry['is_known_bot']}
    - Is using old browser: {log_entry['is_old_browser']}
    - Has suspicious user agent: {log_entry['is_suspicious_user_agent']}
    - Has known malicious user agent: {log_entry['is_malicious_user_agent']}
    
    Malicious pattern matches:
    - Has any malicious pattern: {log_entry['has_malicious_pattern']}
    - Path matches malicious pattern: {log_entry['path_matches']}
    - Query string matches malicious pattern: {log_entry['query_matches']}
    - User agent matches malicious pattern: {log_entry['ua_matches']}
    - Referrer matches malicious pattern: {log_entry['referrer_matches']}
    - SQL injection pattern detected: {log_entry['sql_injection_matches']}
    - XSS pattern detected: {log_entry['xss_matches']}
    - Command injection pattern detected: {log_entry['cmd_injection_matches']}
    - File inclusion pattern detected: {log_entry['file_inclusion_matches']}
    - WordPress attack pattern detected: {log_entry['wordpress_matches']}
    - Shell upload pattern detected: {log_entry['shell_upload_matches']}
    - Malicious file extension detected: {log_entry['malicious_ext_matches']}
    - Likely malicious human activity: {log_entry['likely_malicious_human']}
    
    Note: 
    - Blank or very old user agents are often signs of automated tools or attacks.
    - High ASN reputation scores (>50) indicate the IP comes from a network known to host malicious actors.
    - Combined reputation scores consider both the individual IP and its network origin.
    - Known malicious user agents are patterns that have been observed in malware, botnets, or attack tools.
    - Malicious pattern matches indicate the request contains patterns associated with common attack techniques.
    - "Likely malicious human activity" combines multiple indicators to identify manual attack attempts.
    
    Provide your classification as a JSON object with the following structure:
    {{"classification": "legitimate_human|legitimate_bot|illegitimate_human|illegitimate_bot", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}
    """
    return prompt

def classify_with_sonar(log_entry, api_key, max_retries=3):
    """Classify a log entry using Sonar's API."""
    prompt = create_labeling_prompt(log_entry)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "sonar",
        "messages": [{
            "role": "user",
            "content": prompt
        }],
        "temperature": 0.2,
        "max_tokens": 200
    }

    retries = 0
    while retries < max_retries:
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            response_json = response.json()
            content = response_json['choices'][0]['message']['content']
            
            # Extract JSON from the content using regex
            json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result
            else:
                # Fallback: try to find any JSON object in the content
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    return result
                else:
                    # If no JSON found, try to extract components
                    classification_match = re.search(r'"classification":\s*"([^"]+)"', content)
                    confidence_match = re.search(r'"confidence":\s*([\d.]+)', content)
                    reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', content)
                    
                    if classification_match and confidence_match:
                        classification = classification_match.group(1)
                        confidence = float(confidence_match.group(1))
                        reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
                        
                        result = {
                            "classification": classification,
                            "confidence": confidence,
                            "reasoning": reasoning
                        }
                        return result
                    else:
                        # Unexpected response format, print and retry
                        print(f"Unexpected API response format (attempt {retries+1}/{max_retries}):")
                        print(f"API Response content: {content}")
                        retries += 1
                        if retries < max_retries:
                            print(f"Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            return {
                                "classification": "unknown",
                                "confidence": 0.0,
                                "reasoning": "Failed to parse API response after multiple attempts"
                            }

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error (attempt {retries+1}/{max_retries}): {str(e)}")
            print(f"Raw API Response: {response.text if 'response' in locals() else 'No response'}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: JSON parsing failed - {str(e)}"
                }
        except Exception as e:
            print(f"API Error (attempt {retries+1}/{max_retries}): {str(e)}")
            if 'response' in locals():
                print(f"Response status code: {response.status_code}")
                print(f"Response text: {response.text}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }

def classify_with_openai(log_entry, api_key, max_retries=3):
    """Classify a log entry using OpenAI's API."""
    from openai import OpenAI
    
    prompt = create_labeling_prompt(log_entry)
    
    client = OpenAI(api_key=api_key)
    retries = 0
    
    while retries < max_retries:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=200
            )
            
            content = response.choices[0].message.content
            
            # Extract JSON from the content using regex
            json_match = re.search(r'```json\s*({.*?})\s*```', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result
            else:
                # Fallback: try to find any JSON object in the content
                json_match = re.search(r'({.*})', content, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    return result
                else:
                    # If no JSON found, try to extract components
                    classification_match = re.search(r'"classification":\s*"([^"]+)"', content)
                    confidence_match = re.search(r'"confidence":\s*([\d.]+)', content)
                    reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', content)
                    
                    if classification_match and confidence_match:
                        classification = classification_match.group(1)
                        confidence = float(confidence_match.group(1))
                        reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
                        
                        result = {
                            "classification": classification,
                            "confidence": confidence,
                            "reasoning": reasoning
                        }
                        return result
                    else:
                        # Unexpected response format, print and retry
                        print(f"Unexpected API response format (attempt {retries+1}/{max_retries}):")
                        print(f"API Response content: {content}")
                        retries += 1
                        if retries < max_retries:
                            print(f"Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            return {
                                "classification": "unknown",
                                "confidence": 0.0,
                                "reasoning": "Failed to parse API response after multiple attempts"
                            }
                            
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error (attempt {retries+1}/{max_retries}): {str(e)}")
            print(f"Raw API Response content: {response.choices[0].message.content if 'response' in locals() else 'No response'}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: JSON parsing failed - {str(e)}"
                }
        except Exception as e:
            print(f"API Error (attempt {retries+1}/{max_retries}): {str(e)}")
            if 'response' in locals():
                print(f"Response: {response}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }

def classify_with_gemini(log_entry, api_key, max_retries=3):
    """Classify a log entry using Google's Gemini API."""
    import google.generativeai as genai
    
    prompt = create_labeling_prompt(log_entry)
    
    genai.configure(api_key=api_key)
    retries = 0
    
    while retries < max_retries:
        try:
            model = genai.GenerativeModel('gemini-2.0-flash-lite')
            response = model.generate_content(prompt)
            
            # Extract JSON from response
            result_text = response.text
            
            # Find JSON object in the response
            json_match = re.search(r'```json\s*({.*?})\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result
            else:
                # Fallback: try to find any JSON object in the content
                json_match = re.search(r'({.*})', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    return result
                else:
                    # If no JSON found, try to extract components
                    classification_match = re.search(r'"classification":\s*"([^"]+)"', result_text)
                    confidence_match = re.search(r'"confidence":\s*([\d.]+)', result_text)
                    reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', result_text)
                    
                    if classification_match and confidence_match:
                        classification = classification_match.group(1)
                        confidence = float(confidence_match.group(1))
                        reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
                        
                        result = {
                            "classification": classification,
                            "confidence": confidence,
                            "reasoning": reasoning
                        }
                        return result
                    else:
                        # Unexpected response format, print and retry
                        print(f"Unexpected API response format (attempt {retries+1}/{max_retries}):")
                        print(f"API Response content: {result_text}")
                        retries += 1
                        if retries < max_retries:
                            print(f"Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            return {
                                "classification": "unknown",
                                "confidence": 0.0,
                                "reasoning": "Failed to parse API response after multiple attempts"
                            }
                
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error (attempt {retries+1}/{max_retries}): {str(e)}")
            print(f"Raw API Response: {response.text if 'response' in locals() else 'No response'}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: JSON parsing failed - {str(e)}"
                }
        except Exception as e:
            print(f"API Error (attempt {retries+1}/{max_retries}): {str(e)}")
            if 'response' in locals():
                print(f"Response: {response}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }

def classify_with_anthropic(log_entry, api_key, max_retries=3):
    """Classify a log entry using Anthropic's Claude API."""
    from anthropic import Anthropic
    
    prompt = create_labeling_prompt(log_entry)
    
    client = Anthropic(api_key=api_key)
    retries = 0
    
    while retries < max_retries:
        try:
            response = client.messages.create(
                model="claude-3-5-haiku-latest",
                max_tokens=200,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response
            result_text = response.content[0].text
            
            # Find JSON object in the response
            json_match = re.search(r'```json\s*({.*?})\s*```', result_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                result = json.loads(json_str)
                return result
            else:
                # Fallback: try to find any JSON object in the content
                json_match = re.search(r'({.*})', result_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                    result = json.loads(json_str)
                    return result
                else:
                    # If no JSON found, try to extract components
                    classification_match = re.search(r'"classification":\s*"([^"]+)"', result_text)
                    confidence_match = re.search(r'"confidence":\s*([\d.]+)', result_text)
                    reasoning_match = re.search(r'"reasoning":\s*"([^"]+)"', result_text)
                    
                    if classification_match and confidence_match:
                        classification = classification_match.group(1)
                        confidence = float(confidence_match.group(1))
                        reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
                        
                        result = {
                            "classification": classification,
                            "confidence": confidence,
                            "reasoning": reasoning
                        }
                        return result
                    else:
                        # Unexpected response format, print and retry
                        print(f"Unexpected API response format (attempt {retries+1}/{max_retries}):")
                        print(f"API Response content: {result_text}")
                        retries += 1
                        if retries < max_retries:
                            print(f"Retrying in 2 seconds...")
                            time.sleep(2)
                        else:
                            return {
                                "classification": "unknown",
                                "confidence": 0.0,
                                "reasoning": "Failed to parse API response after multiple attempts"
                            }
                
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error (attempt {retries+1}/{max_retries}): {str(e)}")
            print(f"Raw API Response: {result_text if 'result_text' in locals() else 'No response text'}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: JSON parsing failed - {str(e)}"
                }
        except Exception as e:
            print(f"API Error (attempt {retries+1}/{max_retries}): {str(e)}")
            if 'response' in locals():
                print(f"Response: {response}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in 2 seconds...")
                time.sleep(2)
            else:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                }

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

def label_logs(input_file, output_file, max_samples=1000, batch_size=10, llm_provider='sonar', max_retries=3):
    """
    Main function to label logs with an LLM.
    
    Args:
        input_file: Path to the input CSV file with enriched logs
        output_file: Path to save the labeled logs
        max_samples: Maximum number of samples to label
        batch_size: Number of samples to process in each API batch
        llm_provider: LLM provider to use ('sonar', 'openai', 'gemini', 'anthropic')
        max_retries: Maximum number of retries for API calls
        
    Returns:
        str: Path to the output file
    """
    initial_memory = get_memory_usage()
    print(f"Starting with memory usage: {initial_memory:.2f} MB")
    
    # Get appropriate API key based on provider
    if llm_provider == 'sonar':
        api_key = os.getenv("SONAR_API_KEY")
        if not api_key:
            raise ValueError("SONAR_API_KEY not found in environment variables")
        classify_function = classify_with_sonar
        rate_limit_delay = 0.48  # 125 requests per minute
    elif llm_provider == 'openai':
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        classify_function = classify_with_openai
        rate_limit_delay = 0.2  # 300 requests per minute for GPT-4o-mini
    elif llm_provider == 'gemini':
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        classify_function = classify_with_gemini
        rate_limit_delay = 0.1  # 600 requests per minute for Gemini Flash
    elif llm_provider == 'anthropic':
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        classify_function = classify_with_anthropic
        rate_limit_delay = 0.3  # 200 requests per minute for Claude Haiku
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")
    
    print(f"Using {llm_provider} as the LLM provider")
    
    # Load the first chunk to get total rows
    first_chunk = next(process_dataframe_in_chunks(input_file, chunk_size=1))
    total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract header row
    
    print(f"Input file has {total_rows} rows")
    
    # Determine how many rows to sample
    if max_samples and max_samples < total_rows:
        # Sample randomly across the entire file
        print(f"Sampling {max_samples} rows from {total_rows} total rows")
        
        # Generate random row indices to sample
        sample_indices = sorted(np.random.choice(range(1, total_rows + 1), size=max_samples, replace=False))
        
        # Group indices by chunks for efficient processing
        chunk_size = 50000
        chunk_indices = {}
        for idx in sample_indices:
            chunk_num = (idx - 1) // chunk_size
            if chunk_num not in chunk_indices:
                chunk_indices[chunk_num] = []
            chunk_indices[chunk_num].append(idx - chunk_num * chunk_size - 1)  # Adjust index for within-chunk position
        
        # Process each chunk and extract only the sampled rows
        df_to_label = pd.DataFrame()
        
        for chunk_num, indices in tqdm(chunk_indices.items(), desc="Loading sample chunks"):
            # Skip to the right chunk
            chunk_reader = process_dataframe_in_chunks(input_file, chunk_size)
            for _ in range(chunk_num + 1):
                try:
                    chunk = next(chunk_reader)
                except StopIteration:
                    print(f"Warning: Chunk {chunk_num} not found")
                    continue
            
            # Extract the sampled rows from this chunk
            sampled_rows = chunk.iloc[indices]
            df_to_label = pd.concat([df_to_label, sampled_rows])
            print(f"Successfully loaded batch {chunk_num+1} with {len(sampled_rows)} rows")
    else:
        # Use all rows
        print(f"Using all {total_rows} rows for labeling")
        df_to_label = pd.read_csv(input_file)
        print(f"Successfully loaded all {len(df_to_label)} rows")
    
    print(f"Labeling {len(df_to_label)} rows with batch size {batch_size}")
    
    # Process in batches to avoid rate limits
    labels = []
    
    for i in tqdm(range(0, len(df_to_label), batch_size), desc="Labeling batches"):
        batch = df_to_label.iloc[i:i+batch_size]
        batch_labels = []
        
        print(f"Processing batch {i//batch_size + 1} with {len(batch)} rows")
        
        for _, row in batch.iterrows():
            try:
                result = classify_function(row, api_key, max_retries)
                batch_labels.append(result)
            except Exception as e:
                print(f"Error processing entry: {e}")
                batch_labels.append({
                    "classification": "unknown",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}"
                })
            
            # Rate limiting based on provider
            time.sleep(rate_limit_delay)
        
        labels.extend(batch_labels)
        
        # Report progress and memory usage
        current_memory = get_memory_usage()
        print(f"Completed batch {i//batch_size + 1}: Processed {i+len(batch)}/{len(df_to_label)} entries. "
              f"Memory: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    
    # Add labels to dataframe
    df_labeled = df_to_label.copy()
    df_labeled['classification'] = [label.get('classification', 'unknown') for label in labels]
    df_labeled['confidence'] = [label.get('confidence', 0.0) for label in labels]
    df_labeled['reasoning'] = [label.get('reasoning', '') for label in labels]
    
    # Save the labeled data
    print(f"Saving {len(df_labeled)} labeled entries to {output_file}")
    df_labeled.to_csv(output_file, index=False)
    
    # Final memory report
    final_memory = get_memory_usage()
    print(f"Labeling complete. Final memory usage: {final_memory:.2f} MB (Δ: {final_memory - initial_memory:.2f} MB)")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Label log entries using an LLM')
    parser.add_argument('--input', required=True, help='Path to enriched logs CSV file')
    parser.add_argument('--output', default='labeled_logs.csv', help='Path to save the labeled logs')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to label')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for API calls')
    parser.add_argument('--llm-provider', choices=['sonar', 'openai', 'gemini', 'anthropic'], 
                        default='sonar', help='LLM provider to use for labeling')
    parser.add_argument('--max-retries', type=int, default=3, help='Maximum number of retries for API calls')
    args = parser.parse_args()
    
    label_logs(args.input, args.output, args.samples, args.batch_size, args.llm_provider, args.max_retries)

if __name__ == "__main__":
    main()
