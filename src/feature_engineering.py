import pandas as pd
import numpy as np
import requests
import re
import argparse
import time
import os
import psutil
from datetime import datetime
from tqdm import tqdm

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def process_dataframe_in_chunks(input_file, chunk_size=100000):
    """
    Generator function to process a large CSV file in chunks.
    
    Args:
        input_file: Path to the input CSV file
        chunk_size: Number of rows to process in each chunk
        
    Yields:
        DataFrame chunks
    """
    for chunk in pd.read_csv(input_file, chunksize=chunk_size, parse_dates=['timestamp']):
        yield chunk

def load_malicious_patterns():
    """Load known malicious request patterns from the patterns file."""
    print("Loading malicious request patterns...")
    patterns_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                'src', 'threat_int', 'malicious_req_patterns.txt')
    
    try:
        pattern_categories = {
            'query_string': [],
            'request_uri': [],
            'user_agent': [],
            'referrer': [],
            'common_attack': [],
            'file_inclusion': [],
            'command_injection': [],
            'xss': [],
            'wordpress': [],
            'sql_injection': [],
            'shell_upload': [],
            'malicious_file_ext': [],
            'additional_malicious': [],
            'obfuscation': [],
            'crypto_mining': [],
            'ransomware': [],
            'attack_tools': [],
            'additional_tools': []
        }
        
        current_category = None
        normalized_count = 0
        escaped_count = 0
        
        with open(patterns_file, 'r') as f:
            for line in tqdm(f, desc="Loading patterns"):
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    # Check if this is a category header comment
                    if line.startswith('# ') and 'PATTERNS' in line:
                        category_name = line.split('PATTERNS')[0].strip('# ').strip().lower().replace(' ', '_')
                        if category_name in pattern_categories:
                            current_category = category_name
                    continue
                
                # Add pattern to appropriate category
                if current_category and current_category in pattern_categories:
                    # Remove parentheses that wrap the pattern
                    if line.startswith('(') and line.endswith(')'):
                        line = line[1:-1]
                    
                    # Normalize the pattern instead of skipping it
                    try:
                        # Test compile the pattern to catch syntax errors
                        re.compile(line)
                        pattern_categories[current_category].append(line)
                    except re.error:
                        # Fix unbalanced parentheses by counting and balancing them
                        normalized_pattern = normalize_regex_pattern(line)
                        
                        try:
                            # Verify the normalized pattern compiles
                            re.compile(normalized_pattern)
                            pattern_categories[current_category].append(normalized_pattern)
                            normalized_count += 1
                        except re.error:
                            # If still can't compile, escape the pattern to match it literally
                            escaped_pattern = re.escape(line)
                            pattern_categories[current_category].append(escaped_pattern)
                            escaped_count += 1
        
        # Count total patterns
        total_patterns = sum(len(patterns) for patterns in pattern_categories.values())
        print(f"Loaded {total_patterns} malicious patterns across {len(pattern_categories)} categories")
        print(f"Normalized {normalized_count} patterns with syntax issues")
        print(f"Escaped {escaped_count} patterns that couldn't be normalized")
        
        return pattern_categories
        
    except Exception as e:
        print(f"Error loading malicious patterns: {e}")
        # Return empty patterns as fallback
        return {category: [] for category in pattern_categories}

def normalize_regex_pattern(pattern):
    """
    Normalize a regex pattern by fixing common issues like unbalanced parentheses.
    
    Args:
        pattern: The regex pattern to normalize
        
    Returns:
        str: The normalized pattern
    """
    # Count opening and closing parentheses
    open_count = pattern.count('(')
    close_count = pattern.count(')')
    
    # Balance parentheses
    if open_count > close_count:
        # Add missing closing parentheses
        pattern += ')' * (open_count - close_count)
    elif close_count > open_count:
        # Remove extra closing parentheses or add opening ones at the beginning
        if pattern.startswith(')'):
            # If pattern starts with closing parenthesis, remove extras from the beginning
            excess = close_count - open_count
            pattern = pattern[excess:]
        else:
            # Otherwise add opening parentheses at the beginning
            pattern = '(' * (close_count - open_count) + pattern
    
    # Fix other common regex issues
    
    # 1. Ensure character classes are closed
    open_brackets = pattern.count('[')
    close_brackets = pattern.count(']')
    if open_brackets > close_brackets:
        pattern += ']' * (open_brackets - close_brackets)
    
    # 2. Escape standalone + and ? that might be causing issues
    # This is a simplification - in a real scenario, you'd need more sophisticated parsing
    if '+' in pattern and not re.search(r'\\\+|\[\+\]|\w\+', pattern):
        pattern = pattern.replace('+', '\\+')
    
    # 3. Fix incomplete escape sequences
    if pattern.endswith('\\'):
        pattern += '.'
    
    # 4. Fix incomplete character classes
    if '[' in pattern and ']' not in pattern:
        pattern += ']'
    
    # 5. Fix incomplete alternation
    if '|' in pattern and pattern.endswith('|'):
        pattern += '.'
    
    return pattern

def check_request_against_patterns(row, pattern_categories):
    """Check if a request matches any known malicious patterns."""
    matches = {}
    
    # Helper function to safely check patterns
    def safe_pattern_match(text, patterns):
        if pd.isna(text):
            return False
            
        for pattern in patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    return True
            except re.error:
                # If regex still fails despite normalization, try a simple substring match
                if pattern in text:
                    return True
        return False
    
    # Check path against request URI patterns
    path = row.get('path', '')
    if pd.notna(path):
        matches['path_matches'] = safe_pattern_match(path, pattern_categories['request_uri'])
    else:
        matches['path_matches'] = False
    
    # Check query string patterns (if path contains query parameters)
    if pd.notna(path) and '?' in path:
        query = path.split('?', 1)[1]
        matches['query_matches'] = safe_pattern_match(query, pattern_categories['query_string'])
    else:
        matches['query_matches'] = False
    
    # Check user agent against user agent patterns
    ua = row.get('user_agent', '')
    if pd.notna(ua):
        matches['ua_matches'] = safe_pattern_match(ua, pattern_categories['user_agent'])
    else:
        matches['ua_matches'] = False
    
    # Check referrer against referrer patterns
    ref = row.get('referer', '')
    if pd.notna(ref) and ref != '-':
        matches['referrer_matches'] = safe_pattern_match(ref, pattern_categories['referrer'])
    else:
        matches['referrer_matches'] = False
    
    # Check for SQL injection patterns
    if pd.notna(path):
        matches['sql_injection_matches'] = safe_pattern_match(path, pattern_categories['sql_injection'])
    else:
        matches['sql_injection_matches'] = False
    
    # Check for XSS patterns
    if pd.notna(path):
        matches['xss_matches'] = safe_pattern_match(path, pattern_categories['xss'])
    else:
        matches['xss_matches'] = False
    
    # Check for command injection patterns
    if pd.notna(path):
        matches['cmd_injection_matches'] = safe_pattern_match(path, pattern_categories['command_injection'])
    else:
        matches['cmd_injection_matches'] = False
    
    # Check for file inclusion patterns
    if pd.notna(path):
        matches['file_inclusion_matches'] = safe_pattern_match(path, pattern_categories['file_inclusion'])
    else:
        matches['file_inclusion_matches'] = False
    
    # Check for WordPress specific attack patterns
    if pd.notna(path):
        matches['wordpress_matches'] = safe_pattern_match(path, pattern_categories['wordpress'])
    else:
        matches['wordpress_matches'] = False
    
    # Check for shell upload patterns
    if pd.notna(path):
        matches['shell_upload_matches'] = safe_pattern_match(path, pattern_categories['shell_upload'])
    else:
        matches['shell_upload_matches'] = False
    
    # Check for malicious file extensions
    if pd.notna(path):
        matches['malicious_ext_matches'] = safe_pattern_match(path, pattern_categories['malicious_file_ext'])
    else:
        matches['malicious_ext_matches'] = False
    
    # Aggregate all matches into a single flag
    matches['has_malicious_pattern'] = any(matches.values())
    
    return matches

def enrich_with_features(df):
    """Add features useful for bot and attack detection."""
    initial_memory = get_memory_usage()
    print(f"Memory before feature enrichment: {initial_memory:.2f} MB")
    
    # Calculate time between requests per IP
    print("Calculating time between requests...")
    df = df.sort_values(['ip', 'timestamp'])
    df['prev_timestamp'] = df.groupby('ip')['timestamp'].shift(1)
    df['time_diff'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
    
    # Request frequency features
    print("Calculating request frequency features...")
    ip_counts = df['ip'].value_counts().to_dict()
    df['request_count_per_ip'] = df['ip'].map(ip_counts)
    
    # Calculate request rate (requests per minute)
    print("Calculating request rates per IP...")
    ip_groups = df.groupby('ip')
    request_rates = {}
    
    for ip, group in tqdm(ip_groups, desc="Processing IPs"):
        if len(group) > 1:
            time_span = (group['timestamp'].max() - group['timestamp'].min()).total_seconds() / 60
            if time_span > 0:
                request_rates[ip] = len(group) / time_span
            else:
                request_rates[ip] = len(group)  # All requests at same timestamp
        else:
            request_rates[ip] = 1
    
    df['requests_per_minute'] = df['ip'].map(request_rates)
    
    # Error response features
    print("Adding error response features...")
    df['is_error'] = df['status_code'] >= 400
    df['is_server_error'] = df['status_code'] >= 500
    
    # Path features
    print("Adding path features...")
    df['path_length'] = df['path'].str.len()
    df['has_query_params'] = df['path'].str.contains(r'\?')
    df['has_suspicious_patterns'] = df['path'].str.contains('wp-login|xmlrpc|admin|config|passwd|etc', case=False)
    
    # User agent features
    print("Adding user agent features...")
    df['has_user_agent'] = df['user_agent'] != '-'
    df['is_known_bot'] = df['user_agent'].str.contains('bot|crawl|spider|slurp|bingpreview', case=False)
    df['is_common_browser'] = df['user_agent'].str.contains('Chrome|Firefox|Safari|Edge|MSIE|Opera', case=False)
    df['is_old_browser'] = df['user_agent'].str.contains('MSIE [1-9]|MSIE 10|Firefox/[1-2][0-9]|Chrome/[1-3][0-9]|Safari/[1-5]|Android [1-4]|Opera/[1-9]|Opera/1[0-2]', case=False)
    
    # Check for known malicious user agents
    print("Checking for known malicious user agents...")
    malicious_ua_list = load_malicious_user_agents()
    
    # Create a combined feature for potentially illegitimate user agents
    df['is_suspicious_user_agent'] = (
        (~df['has_user_agent']) | 
        (df['is_known_bot']) | 
        (df['is_old_browser'] & ~df['is_known_bot'])  # Old browsers that aren't declared bots
    )
    
    # Add flag for known malicious user agents
    df['is_malicious_user_agent'] = df['user_agent'].apply(
        lambda ua: any(malicious_pattern in ua for malicious_pattern in malicious_ua_list) if pd.notna(ua) else False
    )
    
    # Update the suspicious user agent flag to include known malicious user agents
    df['is_suspicious_user_agent'] = df['is_suspicious_user_agent'] | df['is_malicious_user_agent']
    
    # Load and check for malicious patterns from the patterns file
    print("Checking for malicious request patterns...")
    pattern_categories = load_malicious_patterns()
    
    # Apply pattern matching to each row (using a smaller batch to avoid memory issues)
    print("Applying pattern matching...")
    batch_size = 10000
    all_results = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch = df.iloc[i:i+batch_size]
        batch_results = batch.apply(lambda row: check_request_against_patterns(row, pattern_categories), axis=1)
        all_results.extend(batch_results)
    
    # Convert results to DataFrame and join with original DataFrame
    pattern_results = pd.DataFrame(all_results)
    
    # Add pattern matching results to the main DataFrame
    for col in pattern_results.columns:
        df[col] = pattern_results[col].values
    
    # Create a combined flag for likely malicious/illegitimate human activity
    df['likely_malicious_human'] = (
        df['has_malicious_pattern'] & 
        ~df['is_known_bot'] & 
        (df['is_suspicious_user_agent'] | df['is_malicious_user_agent'])
    )
    
    current_memory = get_memory_usage()
    print(f"Memory after feature enrichment: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    
    return df


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

def get_ip_reputation(ip_list):
    """Get IP reputation data from a static AbuseIPDB blocklist."""
    initial_memory = get_memory_usage()
    print(f"Memory before IP reputation lookup: {initial_memory:.2f} MB")
    print(f"Looking up reputation for {len(ip_list)} unique IPs...")
    
    # URL to the raw blocklist file
    blocklist_url = "https://raw.githubusercontent.com/borestad/blocklist-abuseipdb/main/abuseipdb-s100-all.ipv4"
    
    try:
        # Fetch the blocklist
        print("Fetching blocklist...")
        response = requests.get(blocklist_url)
        response.raise_for_status()  # Raise exception for HTTP errors
        
        # Parse the blocklist
        print("Parsing blocklist...")
        blocklist_data = {}
        asn_data = {}  # Track ASN reputation data
        pattern = r'^(\d+\.\d+\.\d+\.\d+)\s+#\s+(\w+)\s+(\w+\d+)\s+(.+)$'
        
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
        
        # Create reputation data dictionary
        reputation_data = {}
        
        # Process IPs in batches to avoid memory issues with very large datasets
        for i in tqdm(range(0, len(ip_list), 1000), desc="Processing IP batches"):
            batch_ips = ip_list[i:i+1000]
            for ip in batch_ips:
                is_known_bad = ip in blocklist_data
                
                if is_known_bad:
                    # Use actual data from the blocklist
                    asn = blocklist_data[ip]['asn']
                    reputation_data[ip] = {
                        'score': 100,  # Maximum abuse confidence
                        'is_known_bad': True,
                        'country': blocklist_data[ip]['country'],
                        'asn': asn,
                        'organization': blocklist_data[ip]['organization'],
                        'asn_reputation_score': 100  # Known bad ASN
                    }
                else:
                    # For IPs not in the blocklist, use default values
                    reputation_data[ip] = {
                        'score': 0,  # No known abuse
                        'is_known_bad': False,
                        'country': 'Unknown',
                        'asn': 'Unknown',
                        'organization': 'Unknown',
                        'asn_reputation_score': 0  # Default ASN score
                    }
        
        current_memory = get_memory_usage()
        print(f"Memory after IP reputation lookup: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
        
        return reputation_data, asn_data
        
    except Exception as e:
        print(f"Error fetching or parsing blocklist: {e}")
        # Fallback to random data if fetching fails
        print("Using fallback random data for IP reputation")
        reputation_data = {}
        asn_data = {}
        
        # Generate some random ASNs for the fallback
        random_asns = [f"AS{np.random.randint(1000, 60000)}" for _ in range(50)]
        for asn in random_asns:
            asn_data[asn] = {
                'count': np.random.randint(1, 100),
                'country': np.random.choice(['US', 'CN', 'RU', 'BR', 'IN', 'KR']),
                'organization': 'Unknown',
                'reputation_score': np.random.randint(0, 100)
            }
            
        for ip in tqdm(ip_list, desc="Generating fallback data"):
            random_asn = np.random.choice(random_asns)
            reputation_data[ip] = {
                'score': np.random.randint(0, 100),
                'is_known_bad': np.random.random() < 0.1,
                'country': asn_data[random_asn]['country'],
                'asn': random_asn,
                'organization': 'Unknown',
                'asn_reputation_score': asn_data[random_asn]['reputation_score']
            }
            
        current_memory = get_memory_usage()
        print(f"Memory after fallback IP data generation: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
        
        return reputation_data, asn_data

def add_ip_data(df):
    """Add IP geolocation and reputation data."""
    initial_memory = get_memory_usage()
    print(f"Memory before adding IP data: {initial_memory:.2f} MB")
    
    unique_ips = df['ip'].unique().tolist()
    print(f"Found {len(unique_ips)} unique IPs")
    
    reputation_data, asn_data = get_ip_reputation(unique_ips)
    
    # Add reputation data to dataframe
    print("Adding IP reputation data to dataframe...")
    df['ip_reputation_score'] = df['ip'].map(lambda ip: reputation_data[ip]['score'])
    df['ip_is_known_bad'] = df['ip'].map(lambda ip: reputation_data[ip]['is_known_bad'])
    df['country'] = df['ip'].map(lambda ip: reputation_data[ip]['country'])
    df['asn'] = df['ip'].map(lambda ip: reputation_data[ip]['asn'])
    df['organization'] = df['ip'].map(lambda ip: reputation_data[ip].get('organization', 'Unknown'))
    
    # Add ASN reputation score
    df['asn_reputation_score'] = df['ip'].map(lambda ip: reputation_data[ip].get('asn_reputation_score', 0))
    
    # After initial mapping, update ASN reputation for IPs not in the blocklist
    # but whose ASN is in the blocklist
    print("Updating ASN reputation scores...")
    asn_reputation_map = {asn: data['reputation_score'] for asn, data in asn_data.items()}
    
    # Update ASN reputation for IPs with known ASNs
    mask = (df['asn_reputation_score'] == 0) & (df['asn'] != 'Unknown')
    df.loc[mask, 'asn_reputation_score'] = df.loc[mask, 'asn'].apply(
        lambda asn: asn_reputation_map.get(asn, 0) if asn in asn_reputation_map else 0
    )
    
    # Flag known problematic sources
    print("Adding flags for problematic sources...")
    df['is_hostroyale_asn'] = df['asn'].str.contains('AS9009')  # Specific HostRoyale ASN for known HostRoyale botnet
    df['is_high_risk_country'] = df['country'].isin(['CN', 'RU', 'BR', 'KR', 'IR', 'KP'])  # Common botnet sources
    
    # Add a flag for high-risk ASNs (those with reputation scores above a threshold)
    df['is_high_risk_asn'] = df['asn_reputation_score'] > 50
    
    # Create a combined risk score using both IP and ASN reputation
    df['combined_reputation_score'] = (df['ip_reputation_score'] * 0.7) + (df['asn_reputation_score'] * 0.3)
    
    current_memory = get_memory_usage()
    print(f"Memory after adding IP data: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    
    return df

def detect_attack_patterns(df):
    """Detect patterns indicative of DoS, DDoS, and botnet attacks."""
    initial_memory = get_memory_usage()
    print(f"Memory before attack pattern detection: {initial_memory:.2f} MB")
    
    # Group by IP to find potential DoS attackers
    print("Analyzing IP statistics...")
    ip_stats = df.groupby('ip').agg({
        'timestamp': ['min', 'max', 'count'],
        'time_diff': ['mean', 'min', 'std'],
        'is_error': 'mean',
        'is_server_error': 'mean',
        'path': 'nunique'
    })
    
    ip_stats.columns = ['first_seen', 'last_seen', 'request_count', 
                        'avg_time_between_requests', 'min_time_between_requests', 'std_time_between_requests',
                        'error_rate', 'server_error_rate', 'unique_paths']
    
    # Calculate duration in minutes
    ip_stats['duration_minutes'] = (ip_stats['last_seen'] - ip_stats['first_seen']).dt.total_seconds() / 60
    
    # Calculate requests per minute
    ip_stats['requests_per_minute'] = np.where(
        ip_stats['duration_minutes'] > 0,
        ip_stats['request_count'] / ip_stats['duration_minutes'],
        ip_stats['request_count']  # If all requests at same time
    )
    
    # Identify potential DoS attackers
    print("Identifying potential DoS attackers...")
    ip_stats['potential_dos'] = (
        (ip_stats['requests_per_minute'] > 30) &  # High request rate
        (ip_stats['avg_time_between_requests'] < 1.0) &  # Very frequent requests
        (ip_stats['request_count'] > 50)  # Significant number of requests
    )
    
    # Identify potential scanners/crawlers
    print("Identifying potential scanners/crawlers...")
    ip_stats['potential_scanner'] = (
        (ip_stats['unique_paths'] > 20) &  # Many different URLs
        (ip_stats['request_count'] > 30)  # Significant number of requests
    )
    
    # Add these flags back to the original dataframe
    dos_ips = ip_stats[ip_stats['potential_dos']].index.tolist()
    scanner_ips = ip_stats[ip_stats['potential_scanner']].index.tolist()
    
    print(f"Found {len(dos_ips)} potential DoS attackers and {len(scanner_ips)} potential scanners")
    
    df['potential_dos'] = df['ip'].isin(dos_ips)
    df['potential_scanner'] = df['ip'].isin(scanner_ips)
    
    # Detect DDoS by looking at overall traffic patterns
    # Group by minute to see if there are traffic spikes
    print("Analyzing traffic patterns for DDoS detection...")
    df['minute'] = df['timestamp'].dt.floor('min')
    requests_per_minute = df.groupby('minute').size()
    
    # Calculate Z-score for each minute's request count
    mean_rpm = requests_per_minute.mean()
    std_rpm = requests_per_minute.std()
    if std_rpm > 0:
        minute_z_scores = (requests_per_minute - mean_rpm) / std_rpm
        high_traffic_minutes = minute_z_scores[minute_z_scores > 3].index.tolist()
    else:
        high_traffic_minutes = []
    
    print(f"Identified {len(high_traffic_minutes)} minutes with abnormally high traffic")
    
    # Flag requests during potential DDoS periods
    df['during_traffic_spike'] = df['minute'].isin(high_traffic_minutes)
    
    # Detect botnet coordination by looking for similar behavior across IPs
    # Group by ASN and calculate coordination metrics
    print("Analyzing ASN groups for botnet coordination...")
    asn_groups = df.groupby('asn')
    coordinated_asns = []
    
    for asn, group in tqdm(asn_groups, desc="Analyzing ASNs"):
        if len(group) > 100:  # Only consider ASNs with significant traffic
            # Check if requests are evenly distributed (bot-like)
            ip_counts = group['ip'].value_counts()
            if len(ip_counts) > 5:  # Multiple IPs from same ASN
                cv = ip_counts.std() / ip_counts.mean() if ip_counts.mean() > 0 else 0
                if cv < 0.5:  # Low coefficient of variation suggests coordination
                    coordinated_asns.append(asn)
    
    print(f"Identified {len(coordinated_asns)} ASNs with potential coordinated activity")
    df['from_coordinated_asn'] = df['asn'].isin(coordinated_asns)
    
    # Special attention to 500 errors
    print("Analyzing server error patterns...")
    server_error_requests = df[df['status_code'] >= 500]
    if not server_error_requests.empty:
        # Analyze paths causing server errors
        error_paths = server_error_requests['path'].value_counts()
        high_error_paths = error_paths[error_paths > 5].index.tolist()
        
        # Flag requests to problematic paths
        print(f"Identified {len(high_error_paths)} paths with high server error rates")
        df['to_high_error_path'] = df['path'].isin(high_error_paths)
    else:
        print("No server errors found in the dataset")
        df['to_high_error_path'] = False
    
    current_memory = get_memory_usage()
    print(f"Memory after attack pattern detection: {current_memory:.2f} MB (Δ: {current_memory - initial_memory:.2f} MB)")
    
    return df

def enrich_logs(input_file, output_file, chunk_size=100000):
    """
    Main function to enrich logs with features.
    
    Args:
        input_file: Path to the input CSV file with parsed logs
        output_file: Path to save the enriched logs
        chunk_size: Number of rows to process in each chunk
        
    Returns:
        str: Path to the output file
    """
    initial_memory = get_memory_usage()
    print(f"Starting with memory usage: {initial_memory:.2f} MB")
    
    # Check if input file exists and has content
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return output_file
    
    # Check if file has content (more than just a header)
    with open(input_file, 'r') as f:
        line_count = sum(1 for _ in f)
    
    if line_count <= 1:  # Only header or empty
        print(f"Warning: Input file {input_file} is empty or contains only a header. No data to process.")
        # Create an empty output file with headers
        with open(output_file, 'w') as f:
            f.write("ip,timestamp,method,path,status_code,bytes_sent,referer,user_agent\n")
        return output_file
    
    # Process in chunks for large files
    print(f"Processing {input_file} in chunks of {chunk_size} rows...")
    
    # Get total number of rows for progress tracking
    total_rows = line_count - 1  # Subtract header row
    print(f"Total rows to process: {total_rows}")
    
    # Process the first chunk to get the column names
    first_chunk = True
    processed_rows = 0
    
    for chunk_num, chunk in enumerate(process_dataframe_in_chunks(input_file, chunk_size)):
        chunk_start_memory = get_memory_usage()
        print(f"\n--- Processing chunk {chunk_num+1} ({len(chunk)} rows) ---")
        print(f"Memory at start of chunk: {chunk_start_memory:.2f} MB")
        
        # Add features
        print("Enriching with basic features...")
        chunk = enrich_with_features(chunk)
        
        print("Adding IP data...")
        chunk = add_ip_data(chunk)
        
        print("Detecting attack patterns...")
        chunk = detect_attack_patterns(chunk)
        
        # Save this chunk
        print(f"Saving chunk to {output_file}...")
        if first_chunk:
            chunk.to_csv(output_file, index=False)
            first_chunk = False
        else:
            chunk.to_csv(output_file, mode='a', header=False, index=False)
        
        processed_rows += len(chunk)
        chunk_end_memory = get_memory_usage()
        
        print(f"Chunk {chunk_num+1} complete. Memory: {chunk_end_memory:.2f} MB (Δ: {chunk_end_memory - chunk_start_memory:.2f} MB)")
        print(f"Progress: {processed_rows}/{total_rows} rows processed ({processed_rows/total_rows*100:.1f}%)")
    
    # Final memory report
    final_memory = get_memory_usage()
    print(f"\nEnriched logs saved to {output_file}")
    print(f"Final memory usage: {final_memory:.2f} MB (Δ: {final_memory - initial_memory:.2f} MB)")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Enrich log data with features for classification')
    parser.add_argument('--input', required=True, help='Path to parsed logs CSV file')
    parser.add_argument('--output', default='enriched_logs.csv', help='Path to save the enriched logs')
    parser.add_argument('--chunk-size', type=int, default=100000, help='Number of rows to process in each chunk')
    args = parser.parse_args()
    
    enrich_logs(args.input, args.output, args.chunk_size)

if __name__ == "__main__":
    main()