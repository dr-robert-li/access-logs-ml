import torch
import pandas as pd
import numpy as np
import argparse
import os
import psutil
import json
import re
import requests
from datetime import datetime
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def load_fine_tuned_model(model_path):
    """
    Load the fine-tuned model and tokenizer.
    
    Args:
        model_path: Path to the fine-tuned model directory
        
    Returns:
        tuple: (tokenizer, model)
    """
    initial_memory = get_memory_usage()
    print(f"Memory before loading model: {initial_memory:.2f} MB")
    
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    current_memory = get_memory_usage()
    print(f"Model loaded. Memory usage: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    
    return tokenizer, model

def classify_log_entry(log_entry, tokenizer, model):
    """
    Classify a single log entry using the fine-tuned model.
    
    Args:
        log_entry: Dictionary containing log entry data
        tokenizer: Tokenizer for the model
        model: Fine-tuned model
        
    Returns:
        str: Classification result
    """
    # Enhanced input text with more features from feature_engineering.py
    input_text = f"""
    Analyze this web request:
    IP: {log_entry.get('ip', 'Unknown')} ({log_entry.get('country', 'Unknown')}, {log_entry.get('asn', 'Unknown')})
    Request: {log_entry.get('method', 'Unknown')} {log_entry.get('path', 'Unknown')} {log_entry.get('protocol', 'Unknown')}
    User-Agent: {log_entry.get('user_agent', 'Unknown')}
    Status: {log_entry.get('status_code', 'Unknown')}
    Request frequency: {log_entry.get('requests_per_minute', 0):.2f} req/min
    Time between requests: {log_entry.get('time_diff', 'N/A') if pd.notna(log_entry.get('time_diff')) else 'N/A'} seconds
    
    IP Reputation:
    - IP Reputation Score: {log_entry.get('ip_reputation_score', 0)}
    - ASN Reputation Score: {log_entry.get('asn_reputation_score', 0)}
    - Known Bad IP: {log_entry.get('ip_is_known_bad', False)}
    - High Risk Country: {log_entry.get('is_high_risk_country', False)}
    - High Risk ASN: {log_entry.get('is_high_risk_asn', False)}
    
    User Agent Indicators:
    - Has User Agent: {log_entry.get('has_user_agent', False)}
    - Is Known Bot: {log_entry.get('is_known_bot', False)}
    - Is Old Browser: {log_entry.get('is_old_browser', False)}
    - Is Suspicious User Agent: {log_entry.get('is_suspicious_user_agent', False)}
    - Is Malicious User Agent: {log_entry.get('is_malicious_user_agent', False)}
    
    Attack Indicators:
    - Has Malicious Pattern: {log_entry.get('has_malicious_pattern', False)}
    - SQL Injection: {log_entry.get('sql_injection_matches', False)}
    - XSS: {log_entry.get('xss_matches', False)}
    - Command Injection: {log_entry.get('cmd_injection_matches', False)}
    - File Inclusion: {log_entry.get('file_inclusion_matches', False)}
    - WordPress Attack: {log_entry.get('wordpress_matches', False)}
    - Shell Upload: {log_entry.get('shell_upload_matches', False)}
    
    Traffic Patterns:
    - Potential DoS: {log_entry.get('potential_dos', False)}
    - Potential Scanner: {log_entry.get('potential_scanner', False)}
    - During Traffic Spike: {log_entry.get('during_traffic_spike', False)}
    - From Coordinated ASN: {log_entry.get('from_coordinated_asn', False)}
    """
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.1, top_p=0.9, do_sample=True)
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

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

def parse_classification_result(result):
    """
    Parse the classification result from the model output.
    
    Args:
        result: Raw model output string
        
    Returns:
        dict: Parsed classification data
    """
    # Try to parse as structured data
    try:
        # Extract classification
        classification_match = re.search(r'Classification:\s*(\w+)', result)
        classification = classification_match.group(1) if classification_match else "unknown"
        
        # Extract confidence
        confidence_match = re.search(r'Confidence:\s*([\d.]+)', result)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=$|\n\n)', result, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
        
        return {
            "classification": classification,
            "confidence": confidence,
            "reasoning": reasoning,
            "is_legitimate": "legitimate" in classification.lower(),
            "is_human": "human" in classification.lower()
        }
    except Exception as e:
        print(f"Error parsing result: {e}")
        return {
            "classification": "unknown",
            "confidence": 0.0,
            "reasoning": "Error parsing model output",
            "is_legitimate": False,
            "is_human": False
        }

def load_malicious_user_agents():
    """Load known malicious user agents from the CSV file."""
    print("Loading known malicious user agent patterns...")
    malicious_ua_url = "https://raw.githubusercontent.com/mthcht/awesome-lists/refs/heads/main/Lists/suspicious_http_user_agents_list.csv"
    
    try:
        # Fetch the CSV file
        response = requests.get(malicious_ua_url)
        response.raise_for_status()
        
        # Parse the CSV content
        ua_patterns = []
        lines = response.text.strip().split('\n')
        
        # Skip header
        for line in tqdm(lines[1:], desc="Parsing malicious user agents"):
            if line:
                # Extract the user agent pattern (first column)
                parts = line.split(',', 1)
                if parts:
                    ua_pattern = parts[0].strip()
                    # Replace wildcard * with regex .*
                    ua_pattern = ua_pattern.replace('*', '.*')
                    ua_patterns.append(ua_pattern)
        
        print(f"Loaded {len(ua_patterns)} malicious user agent patterns")
        return ua_patterns
        
    except Exception as e:
        print(f"Error loading malicious user agents: {e}")
        # Return a small set of common malicious patterns as fallback
        fallback_patterns = [
            "python-requests", "Go-http-client", "curl", "wget",
            "Raccoon", "Stealer", "Bot", "RAT", "Malware",
            "axios", "aiohttp", "stratus-red-team"
        ]
        print(f"Using {len(fallback_patterns)} fallback malicious user agent patterns")
        return fallback_patterns

def load_ip_blocklist():
    """Load IP reputation data from a static AbuseIPDB blocklist."""
    print("Loading IP blocklist data...")
    
    # URL to the raw blocklist file
    blocklist_url = "https://raw.githubusercontent.com/borestad/blocklist-abuseipdb/main/abuseipdb-s100-180d.ipv4"
    
    try:
        # Fetch the blocklist
        print("Fetching blocklist...")
        response = requests.get(blocklist_url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the blocklist
        print("Parsing blocklist...")
        blocklist_data = {}
        asn_data = {}  # Track ASN reputation data
        pattern = r'^(\d+\.\d+\.\d+\.\d+)\s+#\s+(\w+)\s+(\w+\d+)\s+(.+)'
        
        for line in tqdm(response.text.strip().split('\n'), desc="Parsing blocklist"):
            if line.startswith('#') or not line.strip():
                continue
                
            match = re.match(pattern, line)
            if match:
                ip, country, asn, org = match.groups()
                blocklist_data[ip] = {
                    'country': country,
                    'asn': asn,
                    'organization': org.strip()
                }
                
                # Track ASNs in the blocklist
                if asn not in asn_data:
                    asn_data[asn] = {
                        'count': 0,
                        'country': country,
                        'organization': org.strip()
                    }
                asn_data[asn]['count'] += 1
        
        print(f"Blocklist contains {len(blocklist_data)} IPs and {len(asn_data)} unique ASNs")
        
        # Calculate ASN reputation scores based on the number of bad IPs
        max_asn_count = max(asn_data[asn]['count'] for asn in asn_data) if asn_data else 1
        for asn in asn_data:
            # Normalize the count to a score between 0 and 100
            asn_data[asn]['reputation_score'] = min(100, (asn_data[asn]['count'] / max_asn_count) * 100)
        
        return blocklist_data, asn_data
        
    except Exception as e:
        print(f"Error fetching or parsing blocklist: {e}")
        # Return empty data as fallback
        print("Using empty blocklist data as fallback")
        return {}, {}

def enrich_logs(df, ip_blocklist, asn_data, malicious_user_agents):
    """
    Enrich logs with additional features for classification.
    
    Args:
        df: DataFrame with parsed log entries
        ip_blocklist: Dictionary of known bad IPs
        asn_data: Dictionary of ASN reputation data
        malicious_user_agents: List of malicious user agent patterns
        
    Returns:
        DataFrame with enriched features
    """
    # Basic enrichment
    df_enriched = df.copy()
    
    # Add timestamp-based features
    if 'timestamp' in df_enriched.columns and df_enriched['timestamp'].dtype != 'object':
        df_enriched['timestamp'] = pd.to_datetime(df_enriched['timestamp'])
        df_enriched['hour_of_day'] = df_enriched['timestamp'].dt.hour
        df_enriched['day_of_week'] = df_enriched['timestamp'].dt.dayofweek
        df_enriched['is_weekend'] = df_enriched['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
        df_enriched['is_business_hours'] = df_enriched['hour_of_day'].apply(lambda x: 1 if 9 <= x <= 17 else 0)
        
        # Sort by timestamp and IP to calculate time differences
        df_enriched = df_enriched.sort_values(['ip', 'timestamp'])
        
        # Calculate time difference between consecutive requests from the same IP
        df_enriched['time_diff'] = df_enriched.groupby('ip')['timestamp'].diff().dt.total_seconds()
        
        # Calculate requests per minute for each IP
        # Group by IP and count requests in 1-minute windows
        df_enriched['minute_bucket'] = df_enriched['timestamp'].dt.floor('1min')
        requests_per_minute = df_enriched.groupby(['ip', 'minute_bucket']).size().reset_index(name='count')
        requests_per_minute = requests_per_minute.groupby('ip')['count'].mean().reset_index(name='requests_per_minute')
        df_enriched = pd.merge(df_enriched, requests_per_minute, on='ip', how='left')
        
        # Identify traffic spikes
        overall_rpm = df_enriched['requests_per_minute'].mean()
        rpm_std = df_enriched['requests_per_minute'].std()
        df_enriched['during_traffic_spike'] = df_enriched['requests_per_minute'] > (overall_rpm + 2 * rpm_std)
    
    # Add IP reputation features
    df_enriched['country'] = 'Unknown'
    df_enriched['asn'] = 'Unknown'
    df_enriched['organization'] = 'Unknown'
    df_enriched['ip_reputation_score'] = 0
    df_enriched['asn_reputation_score'] = 0
    df_enriched['combined_reputation_score'] = 0
    df_enriched['ip_is_known_bad'] = False
    
    # Check if IP is in blocklist
    for idx, row in df_enriched.iterrows():
        ip = row['ip']
        if ip in ip_blocklist:
            df_enriched.at[idx, 'country'] = ip_blocklist[ip]['country']
            df_enriched.at[idx, 'asn'] = ip_blocklist[ip]['asn']
            df_enriched.at[idx, 'organization'] = ip_blocklist[ip]['organization']
            df_enriched.at[idx, 'ip_reputation_score'] = 100  # Max score for known bad IPs
            df_enriched.at[idx, 'ip_is_known_bad'] = True
            
            # Add ASN reputation score if available
            asn = ip_blocklist[ip]['asn']
            if asn in asn_data:
                df_enriched.at[idx, 'asn_reputation_score'] = asn_data[asn]['reputation_score']
    
    # Calculate combined reputation score (weighted average)
    df_enriched['combined_reputation_score'] = (
        0.7 * df_enriched['ip_reputation_score'] + 
        0.3 * df_enriched['asn_reputation_score']
    )
    
    # Add high-risk country and ASN flags
    high_risk_countries = ['CN', 'RU', 'IR', 'KP', 'SY', 'VN']
    df_enriched['is_high_risk_country'] = df_enriched['country'].isin(high_risk_countries)
    df_enriched['is_high_risk_asn'] = df_enriched['asn_reputation_score'] > 50
    
    # Add coordinated ASN flag (multiple IPs from same ASN with high request rates)
    asn_request_counts = df_enriched.groupby('asn')['requests_per_minute'].mean()
    high_traffic_asns = asn_request_counts[asn_request_counts > asn_request_counts.mean() + asn_request_counts.std()].index
    df_enriched['from_coordinated_asn'] = df_enriched['asn'].isin(high_traffic_asns)
    
    # Add potential DoS flag
    df_enriched['potential_dos'] = (
        (df_enriched['requests_per_minute'] > 30) & 
        (df_enriched['time_diff'] < 1)
    )
    
    # Add potential scanner flag
    df_enriched['potential_scanner'] = (
        (df_enriched['requests_per_minute'] > 10) & 
        (df_enriched['status_code'] >= 400) & 
        (df_enriched['status_code'] < 500)
    )
    
    # Add path-based features
    # Identify paths with high error rates
    path_error_rates = df_enriched[df_enriched['status_code'] >= 400].groupby('path').size()
    high_error_paths = path_error_rates[path_error_rates > 5].index
    df_enriched['to_high_error_path'] = df_enriched['path'].isin(high_error_paths)
    
    # Check for suspicious path patterns
    suspicious_path_patterns = [
        r'/(wp-admin|administrator|admin|phpmyadmin|myadmin|mysql|wp-login|xmlrpc)',
        r'\.(git|svn|htaccess|env|config)',
        r'/(shell|backdoor|hack|exploit|cmd|command|exec|eval|system)',
        r'/(passwd|shadow|etc/passwd|etc/shadow)',
        r'/(config|configuration|setup|install|backup|dump)',
        r'/(sql|database|db|mysql|mysqli|oracle|postgresql)',
        r'/(php|asp|aspx|jsp|cgi|pl|py|rb|sh|bash)',
        r'/(login|logon|signin|signup|register|auth|authenticate)',
        r'/(upload|file|download|attachment)',
        r'/(test|demo|sample|example|temp|tmp)'
    ]
    
    df_enriched['has_suspicious_patterns'] = df_enriched['path'].apply(
        lambda p: any(re.search(pattern, str(p), re.IGNORECASE) for pattern in suspicious_path_patterns) if pd.notna(p) else False
    )
    
    # Add user agent features
    df_enriched['has_user_agent'] = df_enriched['user_agent'].notna() & (df_enriched['user_agent'] != '-') & (df_enriched['user_agent'] != '')
    
    # Identify old browsers
    old_browser_patterns = [r'MSIE [1-9]\.', r'Firefox/[1-9]\.', r'Chrome/[1-9]\.', r'Safari/[0-9]']
    df_enriched['is_old_browser'] = df_enriched['user_agent'].apply(
        lambda ua: any(re.search(pattern, str(ua)) for pattern in old_browser_patterns) if isinstance(ua, str) else False
    )
    
    # Identify known bots
    known_bot_patterns = [
        r'bot', r'crawler', r'spider', r'slurp', r'googlebot', r'bingbot', r'yandex', 
        r'baidu', r'duckduck', r'yahoo', r'archive', r'facebook', r'twitter', 
        r'semrush', r'ahrefs', r'moz', r'majestic', r'screaming'
    ]
    df_enriched['is_known_bot'] = df_enriched['user_agent'].apply(
        lambda ua: any(re.search(pattern, str(ua), re.IGNORECASE) for pattern in known_bot_patterns) if isinstance(ua, str) else False
    )
    
    # Check for suspicious user agents
    malicious_ua_patterns = [
        r'(acunetix|appscan|burp|dirbuster|nessus|netsparker|nikto|nmap|paros|qualys|scanner|sqlmap|webinspect|whatweb|zmeu)',
        r'(curl|wget|python-requests|go-http-client|ruby|perl|bash|powershell)',
        r'(masscan|zgrab|gobuster|wfuzz|ffuf|feroxbuster)',
        r'(nuclei|subfinder|amass|httpx|gau|waybackurls)',
        r'(xss|sqli|rce|lfi|rfi|ssrf|csrf|xxe|ssti|deserialization)'
    ]
    
    df_enriched['is_suspicious_user_agent'] = df_enriched['user_agent'].apply(
        lambda ua: any(re.search(pattern, str(ua), re.IGNORECASE) for pattern in malicious_ua_patterns) if isinstance(ua, str) else False
    )
    
    # Check for known malicious user agents
    df_enriched['is_malicious_user_agent'] = df_enriched['user_agent'].apply(
        lambda ua: any(re.search(pattern, str(ua), re.IGNORECASE) for pattern in malicious_user_agents) if isinstance(ua, str) else False
    )
    
    # Add malicious pattern matching features
    df_enriched['has_malicious_pattern'] = False
    df_enriched['path_matches'] = False
    df_enriched['query_matches'] = False
    df_enriched['ua_matches'] = False
    df_enriched['referrer_matches'] = False
    df_enriched['sql_injection_matches'] = False
    df_enriched['xss_matches'] = False
    df_enriched['cmd_injection_matches'] = False
    df_enriched['file_inclusion_matches'] = False
    df_enriched['wordpress_matches'] = False
    df_enriched['shell_upload_matches'] = False
    df_enriched['malicious_ext_matches'] = False
    
    # SQL injection patterns
    sql_patterns = [
        r'(\%27)|(\')|(\-\-)|(\%23)|(#)',
        r'((\%3D)|(=))[^\n]*((\%27)|(\')|(\-\-)|(\%3B)|(;))',
        r'((\%27)|(\'))((\%6F)|o|(\%4F))((\%72)|r|(\%52))',
        r'((\%27)|(\'))union',
        r'exec(\s|\+)+(s|x)p\w+',
        r'UNION\s+ALL\s+SELECT',
        r'SELECT\s+FROM',
        r'INSERT\s+INTO',
        r'UPDATE\s+SET',
        r'DELETE\s+FROM'
    ]
    
    # XSS patterns
    xss_patterns = [
        r'<script>',
        r'<img[^>]+src[^>]+onerror',
        r'javascript:',
        r'onload=',
        r'onclick=',
        r'onmouseover=',
        r'eval\(',
        r'document\.cookie',
        r'document\.location',
        r'document\.write',
        r'alert\(',
        r'prompt\(',
        r'confirm\('
    ]
    
    # Command injection patterns
    cmd_patterns = [
        r';.*(bash|sh|ksh|csh|cat|pwd|ls|cp|mv|rm|ps|wget|curl|lynx|python|perl|ruby)',
        r'\|(bash|sh|ksh|csh|cat|pwd|ls|cp|mv|rm|ps|wget|curl|lynx|python|perl|ruby)',
        r'`(bash|sh|ksh|csh|cat|pwd|ls|cp|mv|rm|ps|wget|curl|lynx|python|perl|ruby)`',
        r'\$\((bash|sh|ksh|csh|cat|pwd|ls|cp|mv|rm|ps|wget|curl|lynx|python|perl|ruby)\)',
        r'system\(',
        r'exec\(',
        r'shell_exec\(',
        r'passthru\(',
        r'popen\('
    ]
    
    # File inclusion patterns
    file_patterns = [
        r'(\.\.\/)+',
        r'\/etc\/passwd',
        r'\/etc\/shadow',
        r'\/etc\/hosts',
        r'\/proc\/self\/environ',
        r'\/var\/log',
        r'\/var\/www',
        r'\/bin\/sh',
        r'c:\\windows',
        r'cmd\.exe',
        r'php:\/\/input',
        r'data:\/\/',
        r'expect:\/\/',
        r'file:\/\/'
    ]
    
    # WordPress specific patterns
    wp_patterns = [
        r'\/wp-admin',
        r'\/wp-content',
        r'\/wp-includes',
        r'\/wp-json',
        r'\/xmlrpc\.php',
        r'\/wp-login\.php',
        r'\/wp-config\.php',
        r'\/wp-cron\.php',
        r'\/wp-load\.php',
        r'\/wp-mail\.php',
        r'\/wp-settings\.php',
        r'\/wp-signup\.php',
        r'\/wp-activate\.php',
        r'\/wp-trackback\.php'
    ]
    
    # Shell upload patterns
    shell_patterns = [
        r'(c99|r57|shell|b374k|weevely)\.php',
        r'(shell|cmd|command|exec|system|passthru)\.php',
        r'(backdoor|rootkit|webshell)\.php',
        r'(upload|uploader|upload_file|upload-file)\.php',
        r'(adminer|phpmy|myadmin|mysql)\.php'
    ]
    
    # Malicious file extensions
    malicious_ext = [
        r'\.php[3-7]',
        r'\.phtml',
        r'\.pht',
        r'\.phps',
        r'\.phar',
        r'\.php_',
        r'\.php-',
        r'\.php\.',
        r'\.php~',
        r'\.php#',
        r'\.php:'
    ]
    
    # Compile all patterns
    all_patterns = {
        'sql_injection_matches': [re.compile(p, re.IGNORECASE) for p in sql_patterns],
        'xss_matches': [re.compile(p, re.IGNORECASE) for p in xss_patterns],
        'cmd_injection_matches': [re.compile(p, re.IGNORECASE) for p in cmd_patterns],
        'file_inclusion_matches': [re.compile(p, re.IGNORECASE) for p in file_patterns],
        'wordpress_matches': [re.compile(p, re.IGNORECASE) for p in wp_patterns],
        'shell_upload_matches': [re.compile(p, re.IGNORECASE) for p in shell_patterns],
        'malicious_ext_matches': [re.compile(p, re.IGNORECASE) for p in malicious_ext]
    }
    
    # Check each request for malicious patterns
    for idx, row in df_enriched.iterrows():
        path = str(row['path']) if pd.notna(row['path']) else ''
        ua = str(row['user_agent']) if pd.notna(row['user_agent']) else ''
        ref = str(row.get('referer', '')) if pd.notna(row.get('referer', '')) else ''
        
        # Split path into path and query string
        path_parts = path.split('?', 1)
        base_path = path_parts[0]
        query = path_parts[1] if len(path_parts) > 1 else ''
        
        # Check path
        for pattern_type, patterns in all_patterns.items():
            for pattern in patterns:
                try:
                    # Check path
                    if re.search(pattern, base_path):
                        df_enriched.at[idx, 'path_matches'] = True
                        df_enriched.at[idx, pattern_type] = True
                        df_enriched.at[idx, 'has_malicious_pattern'] = True
                    
                    # Check query string
                    if query and re.search(pattern, query):
                        df_enriched.at[idx, 'query_matches'] = True
                        df_enriched.at[idx, pattern_type] = True
                        df_enriched.at[idx, 'has_malicious_pattern'] = True
                    
                    # Check user agent
                    if ua and re.search(pattern, ua):
                        df_enriched.at[idx, 'ua_matches'] = True
                        df_enriched.at[idx, pattern_type] = True
                        df_enriched.at[idx, 'has_malicious_pattern'] = True
                    
                    # Check referrer
                    if ref and re.search(pattern, ref):
                        df_enriched.at[idx, 'referrer_matches'] = True
                        df_enriched.at[idx, pattern_type] = True
                        df_enriched.at[idx, 'has_malicious_pattern'] = True
                except:
                    # Skip patterns that can't be compiled
                    continue
    
    # Add likely malicious human flag (combination of indicators)
    df_enriched['likely_malicious_human'] = (
        (df_enriched['has_malicious_pattern']) &
        (~df_enriched['is_known_bot']) &
        (df_enriched['requests_per_minute'] < 30) &  # Not too fast to be a bot
        (
            (df_enriched['sql_injection_matches']) |
            (df_enriched['xss_matches']) |
            (df_enriched['cmd_injection_matches']) |
            (df_enriched['file_inclusion_matches']) |
            (df_enriched['shell_upload_matches'])
        )
    )
    
    return df_enriched

def generate_time_based_summary(df_classified, output_file):
    """
    Generate a time-based summary of classifications, aggregated by hour.
    
    Args:
        df_classified: DataFrame with classified log entries
        output_file: Path to save the time-based summary
        
    Returns:
        str: Path to the output file
    """
    print(f"Generating time-based classification summary...")
    
    # Ensure timestamp is in datetime format
    if 'timestamp' in df_classified.columns:
        if df_classified['timestamp'].dtype == 'object':
            df_classified['timestamp'] = pd.to_datetime(df_classified['timestamp'])
        
        # Create hour bucket
        df_classified['hour'] = df_classified['timestamp'].dt.floor('H')
        
        # Group by hour and calculate summaries
        hourly_summary = df_classified.groupby('hour').agg(
            total_requests=('ip', 'count'),
            legitimate_requests=('is_legitimate', 'sum'),
            illegitimate_requests=('is_legitimate', lambda x: (~x).sum()),
            human_requests=('is_human', 'sum'),
            bot_requests=('is_human', lambda x: (~x).sum()),
            legitimate_human_requests=('classification', lambda x: (x == 'legitimate_human').sum()),
            legitimate_bot_requests=('classification', lambda x: (x == 'legitimate_bot').sum()),
            illegitimate_human_requests=('classification', lambda x: (x == 'illegitimate_human').sum()),
            illegitimate_bot_requests=('classification', lambda x: (x == 'illegitimate_bot').sum()),
            avg_confidence=('confidence', 'mean')
        )
        
        # Calculate percentages
        hourly_summary['legitimate_pct'] = (hourly_summary['legitimate_requests'] / hourly_summary['total_requests'] * 100).round(2)
        hourly_summary['illegitimate_pct'] = (hourly_summary['illegitimate_requests'] / hourly_summary['total_requests'] * 100).round(2)
        hourly_summary['human_pct'] = (hourly_summary['human_requests'] / hourly_summary['total_requests'] * 100).round(2)
        hourly_summary['bot_pct'] = (hourly_summary['bot_requests'] / hourly_summary['total_requests'] * 100).round(2)
        
        # Calculate detailed percentages
        hourly_summary['legitimate_human_pct'] = (hourly_summary['legitimate_human_requests'] / hourly_summary['total_requests'] * 100).round(2)
        hourly_summary['legitimate_bot_pct'] = (hourly_summary['legitimate_bot_requests'] / hourly_summary['total_requests'] * 100).round(2)
        hourly_summary['illegitimate_human_pct'] = (hourly_summary['illegitimate_human_requests'] / hourly_summary['total_requests'] * 100).round(2)
        hourly_summary['illegitimate_bot_pct'] = (hourly_summary['illegitimate_bot_requests'] / hourly_summary['total_requests'] * 100).round(2)
        
        # Reset index to make hour a column
        hourly_summary = hourly_summary.reset_index()
        
        # Save to CSV
        hourly_summary.to_csv(output_file, index=False)
        print(f"Time-based classification summary saved to {output_file}")
        
        return output_file
    else:
        print("Warning: No timestamp column found, skipping time-based summary")
        return None

def generate_summary(df_classified, output_file, tokenizer, model):
    """
    Generate a comprehensive analysis summary.
    
    Args:
        df_classified: DataFrame with classified log entries
        output_file: Path to save the summary
        tokenizer: Model tokenizer
        model: Fine-tuned model
        
    Returns:
        str: Path to the output file
    """
    print("Generating comprehensive analysis summary...")
    
    # Extract basic statistics
    total_requests = len(df_classified)
    
    # Classification breakdown
    classification_counts = df_classified['classification'].value_counts()
    legitimate_human = classification_counts.get('legitimate_human', 0)
    legitimate_bot = classification_counts.get('legitimate_bot', 0)
    illegitimate_human = classification_counts.get('illegitimate_human', 0)
    illegitimate_bot = classification_counts.get('illegitimate_bot', 0)
    unknown = classification_counts.get('unknown', 0)
    
    # Calculate percentages
    legitimate_human_pct = legitimate_human / total_requests * 100 if total_requests > 0 else 0
    legitimate_bot_pct = legitimate_bot / total_requests * 100 if total_requests > 0 else 0
    illegitimate_human_pct = illegitimate_human / total_requests * 100 if total_requests > 0 else 0
    illegitimate_bot_pct = illegitimate_bot / total_requests * 100 if total_requests > 0 else 0
    unknown_pct = unknown / total_requests * 100 if total_requests > 0 else 0
    
    # Time range
    if 'timestamp' in df_classified.columns and df_classified['timestamp'].dtype != 'object':
        start_time = df_classified['timestamp'].min()
        end_time = df_classified['timestamp'].max()
        time_range = f"{start_time} to {end_time}"
    else:
        time_range = "Unknown (timestamp data not available)"
    
    # Potential attacks
    potential_attacks = df_classified[
        (df_classified['classification'].isin(['illegitimate_human', 'illegitimate_bot'])) &
        (df_classified['has_malicious_pattern'] == True)
    ]
    
    # Attack types
    sql_injection = potential_attacks[potential_attacks['sql_injection_matches'] == True]
    xss = potential_attacks[potential_attacks['xss_matches'] == True]
    cmd_injection = potential_attacks[potential_attacks['cmd_injection_matches'] == True]
    file_inclusion = potential_attacks[potential_attacks['file_inclusion_matches'] == True]
    wordpress = potential_attacks[potential_attacks['wordpress_matches'] == True]
    shell_upload = potential_attacks[potential_attacks['shell_upload_matches'] == True]
    
    # Path traversal attacks
    path_traversal = potential_attacks[potential_attacks['path'].str.contains(r'\.\.|%2e%2e|/etc/|/var/|/bin/', case=False, na=False)]
    
    # Top malicious IPs
    top_malicious_ips = df_classified[
        df_classified['classification'].isin(['illegitimate_human', 'illegitimate_bot'])
    ]['ip'].value_counts().head(10)
    
    # Top malicious ASNs
    top_malicious_asns = df_classified[
        df_classified['classification'].isin(['illegitimate_human', 'illegitimate_bot'])
    ]['asn'].value_counts().head(5)
    
    # Top malicious countries
    top_malicious_countries = df_classified[
        df_classified['classification'].isin(['illegitimate_human', 'illegitimate_bot'])
    ]['country'].value_counts().head(5)
    
    # Top targeted paths
    top_targeted_paths = df_classified[
        df_classified['classification'].isin(['illegitimate_human', 'illegitimate_bot'])
    ]['path'].value_counts().head(10)
    
    # Create the summary
    summary = f"""
# Access Log Analysis Summary

## Overview

- **Total Requests**: {total_requests}
- **Time Period**: {time_range}

## Traffic Classification

| Category | Count | Percentage |
|----------|-------|------------|
| Legitimate Human | {legitimate_human} | {legitimate_human_pct:.2f}% |
| Legitimate Bot | {legitimate_bot} | {legitimate_bot_pct:.2f}% |
| Illegitimate Human | {illegitimate_human} | {illegitimate_human_pct:.2f}% |
| Illegitimate Bot | {illegitimate_bot} | {illegitimate_bot_pct:.2f}% |
| Unknown | {unknown} | {unknown_pct:.2f}% |

## Potential Attacks

- **Total Potential Attacks**: {len(potential_attacks)}
- **SQL Injection Attempts**: {len(sql_injection)}
- **Cross-Site Scripting (XSS) Attempts**: {len(xss)}
- **Command Injection Attempts**: {len(cmd_injection)}
- **File Inclusion/Path Traversal Attempts**: {len(file_inclusion) + len(path_traversal)}
- **WordPress-Specific Attacks**: {len(wordpress)}
- **Shell Upload Attempts**: {len(shell_upload)}

## Malicious Traffic Sources

### Top Malicious IPs

{top_malicious_ips.to_string() if not top_malicious_ips.empty else "None detected"}

### Top Malicious ASNs

{top_malicious_asns.to_string() if not top_malicious_asns.empty else "None detected"}

### Top Malicious Countries

{top_malicious_countries.to_string() if not top_malicious_countries.empty else "None detected"}

## Top Targeted Paths

{top_targeted_paths.to_string() if not top_targeted_paths.empty else "None detected"}

## Examples

### Legitimate Human Traffic Example

{df_classified[df_classified['classification'] == 'legitimate_human'].head(1)['raw_log'].values[0] if legitimate_human > 0 and 'raw_log' in df_classified.columns else "No examples available"}

### Legitimate Bot Traffic Example

{df_classified[df_classified['classification'] == 'legitimate_bot'].head(1)['raw_log'].values[0] if legitimate_bot > 0 and 'raw_log' in df_classified.columns else "No examples available"}

### Illegitimate Human Traffic Example

{df_classified[df_classified['classification'] == 'illegitimate_human'].head(1)['raw_log'].values[0] if illegitimate_human > 0 and 'raw_log' in df_classified.columns else "No examples available"}

### Illegitimate Bot Traffic Example

{df_classified[df_classified['classification'] == 'illegitimate_bot'].head(1)['raw_log'].values[0] if illegitimate_bot > 0 and 'raw_log' in df_classified.columns else "No examples available"}

## Recommendations
"""
    
    # Generate recommendations using the model
    recommendations_prompt = f"""
Based on the following web server log analysis, provide specific security recommendations:

- Total Requests: {total_requests}
- Legitimate Human Traffic: {legitimate_human_pct:.2f}%
- Legitimate Bot Traffic: {legitimate_bot_pct:.2f}%
- Illegitimate Human Traffic: {illegitimate_human_pct:.2f}%
- Illegitimate Bot Traffic: {illegitimate_bot_pct:.2f}%

Attack types detected:
- SQL Injection Attempts: {len(sql_injection)}
- XSS Attempts: {len(xss)}
- Command Injection Attempts: {len(cmd_injection)}
- File Inclusion/Path Traversal: {len(file_inclusion) + len(path_traversal)}
- WordPress Attacks: {len(wordpress)}
- Shell Upload Attempts: {len(shell_upload)}

Provide 5-10 specific, actionable security recommendations based on this data. Format as a numbered list.
"""
    
    # Generate recommendations using the model
    input_ids = tokenizer(recommendations_prompt, return_tensors="pt").input_ids
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=500,
            temperature=0.3,
            top_p=0.9,
            do_sample=True
        )
    
    recommendations_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract just the model's response (after the prompt)
    recommendations_response = recommendations_text[len(recommendations_prompt):].strip()
    
    # Add the recommendations to the summary
    summary += f"\n{recommendations_response}\n\n"
    
    # Add time-based summary if available
    if 'timestamp' in df_classified.columns and df_classified['timestamp'].dtype != 'object':
        # Create a time-based summary
        df_classified['hour'] = df_classified['timestamp'].dt.floor('H')
        time_summary = df_classified.groupby('hour').agg({
            'classification': 'count',
            'is_legitimate': 'sum',
            'is_human': 'sum'
        }).reset_index()
        
        # Calculate percentages
        time_summary['legitimate_pct'] = time_summary['is_legitimate'] / time_summary['classification'] * 100
        time_summary['human_pct'] = time_summary['is_human'] / time_summary['classification'] * 100
        time_summary['bot_pct'] = 100 - time_summary['human_pct']
        time_summary['illegitimate_pct'] = 100 - time_summary['legitimate_pct']
        
        # Rename columns for clarity
        time_summary = time_summary.rename(columns={
            'classification': 'total_requests',
            'is_legitimate': 'legitimate_requests',
            'is_human': 'human_requests'
        })
        
        # Calculate derived counts
        time_summary['illegitimate_requests'] = time_summary['total_requests'] - time_summary['legitimate_requests']
        time_summary['bot_requests'] = time_summary['total_requests'] - time_summary['human_requests']
        
        # Add detailed breakdown
        time_summary_text = "## Time-Based Analysis\n\n"
        time_summary_text += "### Traffic by Hour\n\n"
        time_summary_text += "| Hour | Total | Legitimate | Illegitimate | Human | Bot |\n"
        time_summary_text += "|------|-------|------------|--------------|-------|-----|\n"
        
        for _, row in time_summary.iterrows():
            hour_str = row['hour'].strftime("%Y-%m-%d %H:00")
            time_summary_text += f"| {hour_str} | {row['total_requests']} | "
            time_summary_text += f"{row['legitimate_requests']} ({row['legitimate_pct']:.1f}%) | "
            time_summary_text += f"{row['illegitimate_requests']} ({row['illegitimate_pct']:.1f}%) | "
            time_summary_text += f"{row['human_requests']} ({row['human_pct']:.1f}%) | "
            time_summary_text += f"{row['bot_requests']} ({row['bot_pct']:.1f}%) |\n"
        
        summary += f"\n{time_summary_text}\n"
    
    summary += f"""
## Additional Notes

This analysis was generated automatically based on machine learning classification of access logs. While the system aims for high accuracy, manual review is recommended for critical security decisions.

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    # Save the summary to a file
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"Comprehensive analysis summary saved to {output_file}")
    return output_file

def filter_by_status_code(df, status_code_whitelist=None, status_code_blacklist=None):
    """
    Filter DataFrame based on status code whitelist or blacklist.
    
    Args:
        df: DataFrame to filter
        status_code_whitelist: Regex pattern for status codes to include (only these will be processed)
        status_code_blacklist: Regex pattern for status codes to exclude (all except these will be processed)
        
    Returns:
        DataFrame: Filtered DataFrame
    """
    if status_code_whitelist and status_code_blacklist:
        print("Warning: Both status code whitelist and blacklist provided. Whitelist will take precedence.")
    
    if status_code_whitelist:
        # Compile regex pattern
        whitelist_regex = re.compile(status_code_whitelist)
        # Keep only entries with status codes matching the whitelist pattern
        filtered_df = df[df['status_code'].astype(str).str.match(whitelist_regex)]
        print(f"Applied status code whitelist filter: {status_code_whitelist}, {len(filtered_df)} entries remaining")
        return filtered_df
    
    elif status_code_blacklist:
        # Compile regex pattern
        blacklist_regex = re.compile(status_code_blacklist)
        # Remove entries with status codes matching the blacklist pattern
        filtered_df = df[~df['status_code'].astype(str).str.match(blacklist_regex)]
        print(f"Applied status code blacklist filter: {status_code_blacklist}, {len(filtered_df)} entries remaining")
        return filtered_df
    
    # No filtering if neither whitelist nor blacklist is provided
    return df

def analyze_logs(input_file, model_path, output_file, summary_file=None, batch_size=32, 
                status_code_whitelist=None, status_code_blacklist=None):
    """
    End-to-end analysis of logs.
    
    Args:
        input_file: Path to the input CSV file with log entries
        model_path: Path to the fine-tuned model directory
        output_file: Path to save the classified logs
        summary_file: Path to save the analysis summary
        batch_size: Number of entries to process in each batch
        status_code_whitelist: Regex pattern for status codes to include
        status_code_blacklist: Regex pattern for status codes to exclude
        
    Returns:
        tuple: (Path to classified logs, Path to summary file)
    """
    initial_memory = get_memory_usage()
    print(f"Starting with memory usage: {initial_memory:.2f} MB")
    
    # Load model
    tokenizer, model = load_fine_tuned_model(model_path)
    
    # Load external data sources
    malicious_user_agents = load_malicious_user_agents()
    ip_blocklist, asn_data = load_ip_blocklist()
    
    # Get total number of rows for progress tracking
    total_rows = sum(1 for _ in open(input_file)) - 1  # Subtract header row
    print(f"Total rows to process: {total_rows}")
    
    # Process in batches
    first_chunk = True
    processed_rows = 0
    all_classified_chunks = []
    
    for chunk_num, chunk in enumerate(process_dataframe_in_chunks(input_file, batch_size)):
        chunk_start_memory = get_memory_usage()
        print(f"\n--- Processing chunk {chunk_num+1} ({len(chunk)} rows) ---")
        print(f"Memory at start of chunk: {chunk_start_memory:.2f} MB")
        
        # Apply status code filtering
        filtered_chunk = filter_by_status_code(chunk, status_code_whitelist, status_code_blacklist)
        
        # Skip empty chunks after filtering
        if len(filtered_chunk) == 0:
            print("Chunk is empty after filtering, skipping...")
            continue
        
        # Enrich logs with features
        print("Enriching logs with features...")
        enriched_chunk = enrich_logs(filtered_chunk, ip_blocklist, asn_data, malicious_user_agents)
        
        # Classify each entry in the chunk
        classifications = []
        
        for _, row in tqdm(enriched_chunk.iterrows(), total=len(enriched_chunk), desc=f"Classifying chunk {chunk_num+1}"):
            try:
                result = classify_log_entry(row, tokenizer, model)
                parsed_result = parse_classification_result(result)
                
                # Add the classification data to the row
                row_data = row.to_dict()
                row_data.update(parsed_result)
                classifications.append(row_data)
            except Exception as e:
                print(f"Error processing entry: {e}")
                
                # Add a default classification
                row_data = row.to_dict()
                row_data.update({
                    "classification": "error",
                    "confidence": 0.0,
                    "reasoning": f"Error: {str(e)}",
                    "is_legitimate": False,
                    "is_human": False
                })
                classifications.append(row_data)
        
        # Convert to DataFrame
        df_classified = pd.DataFrame(classifications)
        all_classified_chunks.append(df_classified)
        
        # Save this chunk
        if first_chunk:
            df_classified.to_csv(output_file, index=False)
            first_chunk = False
        else:
            df_classified.to_csv(output_file, mode='a', header=False, index=False)
        
        processed_rows += len(filtered_chunk)
        chunk_end_memory = get_memory_usage()
        
        print(f"Chunk {chunk_num+1} complete. Memory: {chunk_end_memory:.2f} MB (Δ: {chunk_end_memory - chunk_start_memory:.2f} MB)")
        print(f"Progress: {processed_rows}/{total_rows} rows processed ({processed_rows/total_rows*100:.1f}%)")
    
    # Combine all classified chunks for summary generation
    if all_classified_chunks:
        combined_df = pd.concat(all_classified_chunks, ignore_index=True)
        
        # Generate time-based summary
        time_based_output = os.path.splitext(output_file)[0] + "_time_based.csv"
        generate_time_based_summary(combined_df, time_based_output)
        
        # Generate comprehensive summary if requested
        if summary_file:
            generate_summary(combined_df, summary_file, tokenizer, model)
    else:
        print("Warning: No data remained after filtering. No summaries generated.")
    
    # Final memory report
    final_memory = get_memory_usage()
    print(f"\nAnalysis complete. Results saved to {output_file}")
    print(f"Time-based summary saved to {os.path.splitext(output_file)[0]}_time_based.csv")
    if summary_file:
        print(f"Comprehensive analysis summary saved to {summary_file}")
    print(f"Final memory usage: {final_memory:.2f} MB (Δ: {final_memory - initial_memory:.2f} MB)")
    
    return output_file, summary_file if summary_file else None

def run_inference(model_path, input_file, output_file, batch_size=32, 
                 status_code_whitelist=None, status_code_blacklist=None):
    """
    Run inference on log entries using the fine-tuned model.
    
    Args:
        model_path: Path to the fine-tuned model directory
        input_file: Path to the input CSV file with log entries
        output_file: Path to save the classified logs
        batch_size: Number of entries to process in each batch
        status_code_whitelist: Regex pattern for status codes to include
        status_code_blacklist: Regex pattern for status codes to exclude
        
    Returns:
        str: Path to the output file
    """
    # Generate summary file path
    summary_file = os.path.splitext(output_file)[0] + "_summary.md"
    
    # Run the full analysis
    classified_logs, summary = analyze_logs(
        input_file, 
        model_path, 
        output_file, 
        summary_file, 
        batch_size,
        status_code_whitelist,
        status_code_blacklist
    )
    
    return classified_logs

def main():
    parser = argparse.ArgumentParser(description='Classify log entries using a fine-tuned model')
    parser.add_argument('--model', required=True, help='Path to the fine-tuned model')
    parser.add_argument('--input', required=True, help='Path to logs CSV file to classify')
    parser.add_argument('--output', default='classified_logs.csv', help='Path to save the classified logs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--status-code-whitelist', help='Regex pattern for status codes to include (only these will be processed)')
    parser.add_argument('--status-code-blacklist', help='Regex pattern for status codes to exclude (all except these will be processed)')
    parser.add_argument('--summary', action='store_true', help='Generate a comprehensive analysis summary')
    args = parser.parse_args()
    
    # Set summary file path if requested
    summary_file = os.path.splitext(args.output)[0] + "_summary.md" if args.summary else None
    
    # Run the analysis
    classified_logs, summary = analyze_logs(
        args.input, 
        args.model, 
        args.output, 
        summary_file, 
        args.batch_size,
        args.status_code_whitelist,
        args.status_code_blacklist
    )
    
    print(f"Classification complete. Results saved to {classified_logs}")
    if summary:
        print(f"Comprehensive analysis summary saved to {summary}")
    print(f"A time-based summary has also been generated at {os.path.splitext(args.output)[0]}_time_based.csv")

if __name__ == "__main__":
    main()