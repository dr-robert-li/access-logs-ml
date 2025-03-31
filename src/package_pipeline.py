import os
import sys
import pickle
import shutil
import argparse
import requests
from datetime import datetime
import pandas as pd
import re
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import necessary modules
from src.parse_logs import parse_phpfpm_logs
from src.feature_engineering import enrich_logs, normalize_regex_pattern
from src.create_validation_set import prepare_training_data

class LogClassificationPipeline:
    """
    A self-contained pipeline for classifying web server access logs.
    This class encapsulates all the necessary components for parsing,
    feature engineering, and classification of access logs.
    """
    
    def __init__(self, model_path, threat_patterns_path=None, status_code_whitelist=None, status_code_blacklist=None):
        """
        Initialize the pipeline with a trained model and threat patterns.
        
        Args:
            model_path: Path to the trained model directory
            threat_patterns_path: Path to the malicious request patterns file
            status_code_whitelist: Regex pattern for status codes to include (only these will be processed)
            status_code_blacklist: Regex pattern for status codes to exclude (all except these will be processed)
        """
        self.model_path = model_path
        
        # Set status code filters
        self.status_code_whitelist = status_code_whitelist
        self.status_code_blacklist = status_code_blacklist
        
        if self.status_code_whitelist and self.status_code_blacklist:
            print("Warning: Both status code whitelist and blacklist provided. Whitelist will take precedence.")
        
        # Compile regex patterns for status code filtering if provided
        self.status_whitelist_regex = re.compile(self.status_code_whitelist) if self.status_code_whitelist else None
        self.status_blacklist_regex = re.compile(self.status_code_blacklist) if self.status_code_blacklist else None
        
        # Load the model
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # Load threat patterns
        self.threat_patterns = []
        if threat_patterns_path and os.path.exists(threat_patterns_path):
            with open(threat_patterns_path, 'r') as f:
                self.threat_patterns = f.read()
            print(f"Loaded threat patterns from {threat_patterns_path}")
        else:
            print("No threat patterns file provided or file not found")
        
        # Load external data sources
        self.malicious_user_agents = self._load_malicious_user_agents()
        self.ip_blocklist, self.asn_data = self._load_ip_blocklist()
        
        # Compile regex patterns for feature engineering
        self._compile_patterns()
    
    def _normalize_regex_pattern(self, pattern):
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
    
    def _load_malicious_user_agents(self):
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
    
    def _load_ip_blocklist(self):
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
    
    def _compile_patterns(self):
        """Compile regex patterns used in feature engineering."""
        # Common suspicious path patterns
        self.suspicious_path_patterns = [
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
        
        # Known malicious user agents
        self.malicious_ua_patterns = [
            r'(acunetix|appscan|burp|dirbuster|nessus|netsparker|nikto|nmap|paros|qualys|scanner|sqlmap|webinspect|whatweb|zmeu)',
            r'(curl|wget|python-requests|go-http-client|ruby|perl|bash|powershell)',
            r'(masscan|zgrab|gobuster|wfuzz|ffuf|feroxbuster)',
            r'(nuclei|subfinder|amass|httpx|gau|waybackurls)',
            r'(xss|sqli|rce|lfi|rfi|ssrf|csrf|xxe|ssti|deserialization)'
        ]
        
        # Normalize and compile patterns
        self.suspicious_path_regex = []
        for pattern in self.suspicious_path_patterns:
            try:
                normalized_pattern = self._normalize_regex_pattern(pattern)
                self.suspicious_path_regex.append(re.compile(normalized_pattern, re.IGNORECASE))
            except re.error as e:
                print(f"Error compiling pattern '{pattern}': {e}")
                # Use a safe version of the pattern
                safe_pattern = re.escape(pattern)
                self.suspicious_path_regex.append(re.compile(safe_pattern, re.IGNORECASE))
        
        self.malicious_ua_regex = []
        for pattern in self.malicious_ua_patterns:
            try:
                normalized_pattern = self._normalize_regex_pattern(pattern)
                self.malicious_ua_regex.append(re.compile(normalized_pattern, re.IGNORECASE))
            except re.error as e:
                print(f"Error compiling pattern '{pattern}': {e}")
                # Use a safe version of the pattern
                safe_pattern = re.escape(pattern)
                self.malicious_ua_regex.append(re.compile(safe_pattern, re.IGNORECASE))
    
    def parse_logs(self, log_file_or_content, is_file=True):
        """
        Parse raw access logs into structured data.
        
        Args:
            log_file_or_content: Path to log file or log content as string
            is_file: Whether the input is a file path or log content
            
        Returns:
            DataFrame with parsed log entries
        """
        if is_file:
            # Parse from file
            log_data = parse_phpfpm_logs(log_file_or_content)
            return pd.DataFrame(log_data)
        else:
            # Parse from string content
            log_lines = log_file_or_content.strip().split('\n')
            log_data = []
            
            # Pattern for standard Apache/Nginx access logs
            pattern = r'(\d+\.\d+\.\d+\.\d+) - - \[([^\]]+)\] "([^"]*)" (\d+) ([^ ]+) "([^"]*)" "([^"]*)"'
            
            for line in log_lines:
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
                    
                    log_data.append({
                        'ip': ip,
                        'timestamp': timestamp_dt,
                        'method': method,
                        'path': path,
                        'protocol': protocol,
                        'status_code': int(status_code),
                        'response_size': response_size_int,
                        'referer': referer,
                        'user_agent': user_agent,
                        'raw_log': line.strip()
                    })
            
            return pd.DataFrame(log_data)
    
    def enrich_logs(self, df):
        """
        Enrich logs with additional features for classification.
        
        Args:
            df: DataFrame with parsed log entries
            
        Returns:
            DataFrame with enriched features
        """
        # Apply status code filtering if specified
        if self.status_whitelist_regex:
            # Keep only entries with status codes matching the whitelist pattern
            df = df[df['status_code'].astype(str).str.match(self.status_whitelist_regex)]
            print(f"Applied status code whitelist filter: {self.status_code_whitelist}, {len(df)} entries remaining")
        elif self.status_blacklist_regex:
            # Remove entries with status codes matching the blacklist pattern
            df = df[~df['status_code'].astype(str).str.match(self.status_blacklist_regex)]
            print(f"Applied status code blacklist filter: {self.status_code_blacklist}, {len(df)} entries remaining")
        
        # If no entries remain after filtering, return the empty dataframe
        if len(df) == 0:
            print("Warning: No entries remain after status code filtering")
            return df
            
        # Basic enrichment
        df_enriched = df.copy()
        
        # Add timestamp-based features
        if 'timestamp' in df_enriched.columns and df_enriched['timestamp'].dtype != 'object':
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
            if ip in self.ip_blocklist:
                df_enriched.at[idx, 'country'] = self.ip_blocklist[ip]['country']
                df_enriched.at[idx, 'asn'] = self.ip_blocklist[ip]['asn']
                df_enriched.at[idx, 'organization'] = self.ip_blocklist[ip]['organization']
                df_enriched.at[idx, 'ip_reputation_score'] = 100  # Max score for known bad IPs
                df_enriched.at[idx, 'ip_is_known_bad'] = True
                
                # Add ASN reputation score if available
                asn = self.ip_blocklist[ip]['asn']
                if asn in self.asn_data:
                    df_enriched.at[idx, 'asn_reputation_score'] = self.asn_data[asn]['reputation_score']
        
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
        df_enriched['has_suspicious_patterns'] = df_enriched['path'].apply(
            lambda p: any(pattern.search(p) for pattern in self.suspicious_path_regex)
        )
        
        # Add user agent features
        df_enriched['has_user_agent'] = df_enriched['user_agent'].notna() & (df_enriched['user_agent'] != '-') & (df_enriched['user_agent'] != '')
        
        # Identify old browsers
        old_browser_patterns = [r'MSIE [1-9]\.', r'Firefox/[1-9]\.', r'Chrome/[1-9]\.', r'Safari/[0-9]']
        df_enriched['is_old_browser'] = df_enriched['user_agent'].apply(
            lambda ua: any(re.search(pattern, ua) for pattern in old_browser_patterns) if isinstance(ua, str) else False
        )
        
        # Identify known bots
        known_bot_patterns = [
            r'bot', r'crawler', r'spider', r'slurp', r'googlebot', r'bingbot', r'yandex', 
            r'baidu', r'duckduck', r'yahoo', r'archive', r'facebook', r'twitter', 
            r'semrush', r'ahrefs', r'moz', r'majestic', r'screaming'
        ]
        df_enriched['is_known_bot'] = df_enriched['user_agent'].apply(
            lambda ua: any(re.search(pattern, ua, re.IGNORECASE) for pattern in known_bot_patterns) if isinstance(ua, str) else False
        )
        
        # Check for suspicious user agents
        df_enriched['is_suspicious_user_agent'] = df_enriched['user_agent'].apply(
            lambda ua: any(pattern.search(ua) for pattern in self.malicious_ua_regex) if isinstance(ua, str) else False
        )
        
        # Check for known malicious user agents
        df_enriched['is_malicious_user_agent'] = df_enriched['user_agent'].apply(
            lambda ua: any(re.search(pattern, ua, re.IGNORECASE) for pattern in self.malicious_user_agents) if isinstance(ua, str) else False
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
        
        # Normalize and compile all patterns
        all_patterns = {
            'sql_injection_matches': [self._normalize_regex_pattern(p) for p in sql_patterns],
            'xss_matches': [self._normalize_regex_pattern(p) for p in xss_patterns],
            'cmd_injection_matches': [self._normalize_regex_pattern(p) for p in cmd_patterns],
            'file_inclusion_matches': [self._normalize_regex_pattern(p) for p in file_patterns],
            'wordpress_matches': [self._normalize_regex_pattern(p) for p in wp_patterns],
            'shell_upload_matches': [self._normalize_regex_pattern(p) for p in shell_patterns],
            'malicious_ext_matches': [self._normalize_regex_pattern(p) for p in malicious_ext]
        }
        
        # Check each request for malicious patterns
        for idx, row in df_enriched.iterrows():
            path = row['path'] if pd.notna(row['path']) else ''
            ua = row['user_agent'] if pd.notna(row['user_agent']) else ''
            ref = row['referer'] if pd.notna(row['referer']) else ''
            
            # Split path into path and query string
            path_parts = path.split('?', 1)
            base_path = path_parts[0]
            query = path_parts[1] if len(path_parts) > 1 else ''
            
            # Check path
            for pattern_type, patterns in all_patterns.items():
                for pattern in patterns:
                    try:
                        # Check path
                        if re.search(pattern, base_path, re.IGNORECASE):
                            df_enriched.at[idx, 'path_matches'] = True
                            df_enriched.at[idx, pattern_type] = True
                            df_enriched.at[idx, 'has_malicious_pattern'] = True
                        
                        # Check query string
                        if query and re.search(pattern, query, re.IGNORECASE):
                            df_enriched.at[idx, 'query_matches'] = True
                            df_enriched.at[idx, pattern_type] = True
                            df_enriched.at[idx, 'has_malicious_pattern'] = True
                        
                        # Check user agent
                        if ua and re.search(pattern, ua, re.IGNORECASE):
                            df_enriched.at[idx, 'ua_matches'] = True
                            df_enriched.at[idx, pattern_type] = True
                            df_enriched.at[idx, 'has_malicious_pattern'] = True
                        
                        # Check referrer
                        if ref and re.search(pattern, ref, re.IGNORECASE):
                            df_enriched.at[idx, 'referrer_matches'] = True
                            df_enriched.at[idx, pattern_type] = True
                            df_enriched.at[idx, 'has_malicious_pattern'] = True
                    except re.error:
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
    
    def classify_logs(self, df_enriched):
        """
        Classify log entries using the trained model.
        
        Args:
            df_enriched: DataFrame with enriched log entries
            
        Returns:
            DataFrame with classification results
        """
        # Create a copy to avoid modifying the original
        df_classified = df_enriched.copy()
        
        # Prepare input for the model
        inputs = []
        for _, row in df_classified.iterrows():
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
            inputs.append(input_text.strip())
        
        # Batch processing for efficiency
        batch_size = 16
        all_classifications = []
        all_confidences = []
        all_reasonings = []
        
        for i in range(0, len(inputs), batch_size):
            batch_inputs = inputs[i:i+batch_size]
            
            # Generate classifications
            batch_outputs = []
            for input_text in batch_inputs:
                # Tokenize input
                input_ids = self.tokenizer(input_text, return_tensors="pt").input_ids
                
                # Generate output
                with torch.no_grad():
                    output_ids = self.model.generate(
                        input_ids,
                        max_new_tokens=200,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=True
                    )
                
                # Decode output
                output_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # Extract the model's response (after the input)
                response = output_text[len(input_text):].strip()
                batch_outputs.append(response)
            
            # Parse the outputs
            for output in batch_outputs:
                # Extract classification
                classification_match = re.search(r'Classification:\s*(\w+)', output)
                classification = classification_match.group(1) if classification_match else "unknown"
                
                # Extract confidence
                confidence_match = re.search(r'Confidence:\s*([\d.]+)', output)
                confidence = float(confidence_match.group(1)) if confidence_match else 0.0
                
                # Extract reasoning
                reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=$|\n\n)', output, re.DOTALL)
                reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning provided"
                
                all_classifications.append(classification)
                all_confidences.append(confidence)
                all_reasonings.append(reasoning)
        
        # Add classifications to the dataframe
        df_classified['classification'] = all_classifications
        df_classified['confidence'] = all_confidences
        df_classified['reasoning'] = all_reasonings
        
        # Add simplified classification
        df_classified['is_legitimate'] = df_classified['classification'].str.contains('legitimate')
        df_classified['is_human'] = df_classified['classification'].str.contains('human')
        
        return df_classified
    
    def analyze_logs(self, log_file_or_content, is_file=True):
        """
        End-to-end analysis of logs.
        
        Args:
            log_file_or_content: Path to log file or log content as string
            is_file: Whether the input is a file path or log content
            
        Returns:
            DataFrame with parsed, enriched, and classified log entries
        """
        print(f"Analyzing {'log file' if is_file else 'log content'}...")
        
        # Step 1: Parse logs
        print("Step 1: Parsing logs...")
        df_parsed = self.parse_logs(log_file_or_content, is_file)
        print(f"Parsed {len(df_parsed)} log entries")
        
        # Step 2: Enrich logs
        print("Step 2: Enriching logs with features...")
        df_enriched = self.enrich_logs(df_parsed)
        print(f"Enriched {len(df_enriched)} log entries")
        
        # Step 3: Classify logs
        print("Step 3: Classifying logs...")
        df_classified = self.classify_logs(df_enriched)
        print(f"Classified {len(df_classified)} log entries")
        
        # Step 4: Generate time-based summary
        print("Step 4: Generating time-based summary...")
        if 'timestamp' in df_classified.columns and df_classified['timestamp'].dtype != 'object':
            # Group by hour
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
            time_summary['legitimate_human_requests'] = df_classified[
                (df_classified['classification'] == 'legitimate_human') & 
                (df_classified['hour'].isin(time_summary['hour']))
            ].groupby('hour').size().reindex(time_summary['hour']).fillna(0).values
            time_summary['legitimate_bot_requests'] = df_classified[
                (df_classified['classification'] == 'legitimate_bot') & 
                (df_classified['hour'].isin(time_summary['hour']))
            ].groupby('hour').size().reindex(time_summary['hour']).fillna(0).values
            time_summary['illegitimate_human_requests'] = df_classified[
                (df_classified['classification'] == 'illegitimate_human') & 
                (df_classified['hour'].isin(time_summary['hour']))
            ].groupby('hour').size().reindex(time_summary['hour']).fillna(0).values
            time_summary['illegitimate_bot_requests'] = df_classified[
                (df_classified['classification'] == 'illegitimate_bot') & 
                (df_classified['hour'].isin(time_summary['hour']))
            ].groupby('hour').size().reindex(time_summary['hour']).fillna(0).values
            
            # Add time summary to the result
            df_classified.attrs['time_summary'] = time_summary
        
        return df_classified
    
    def generate_summary(self, df_classified, prompt=None):
        """
        Generate a comprehensive analysis summary.
        
        Args:
            df_classified: DataFrame with classified log entries
            prompt: Optional custom prompt to guide the summary
            
        Returns:
            str: Markdown-formatted summary
        """
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

    {df_classified[df_classified['classification'] == 'legitimate_human'].head(1)['raw_log'].values[0] if legitimate_human > 0 else "No examples available"}

    ### Legitimate Bot Traffic Example

    {df_classified[df_classified['classification'] == 'legitimate_bot'].head(1)['raw_log'].values[0] if legitimate_bot > 0 else "No examples available"}

    ### Illegitimate Human Traffic Example

    {df_classified[df_classified['classification'] == 'illegitimate_human'].head(1)['raw_log'].values[0] if illegitimate_human > 0 else "No examples available"}

    ### Illegitimate Bot Traffic Example

    {df_classified[df_classified['classification'] == 'illegitimate_bot'].head(1)['raw_log'].values[0] if illegitimate_bot > 0 else "No examples available"}

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
        input_ids = self.tokenizer(recommendations_prompt, return_tensors="pt").input_ids
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=500,
                temperature=0.3,
                top_p=0.9,
                do_sample=True
            )
        
        recommendations_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract just the model's response (after the prompt)
        recommendations_response = recommendations_text[len(recommendations_prompt):].strip()
        
        # Add the recommendations to the summary
        summary += f"\n{recommendations_response}\n\n"
        
        # Add time-based summary if available
        time_summary_text = ""
        if hasattr(df_classified, 'attrs') and 'time_summary' in df_classified.attrs:
            time_summary = df_classified.attrs['time_summary']
            
            # Create a simple text representation of the time summary
            time_summary_text = "### Traffic by Hour\n\n"
            time_summary_text += "| Hour | Total | Legitimate | Illegitimate | Human | Bot |\n"
            time_summary_text += "|------|-------|------------|--------------|-------|-----|\n"
            
            for _, row in time_summary.iterrows():
                hour_str = row['hour'].strftime("%Y-%m-%d %H:00")
                time_summary_text += f"| {hour_str} | {row['total_requests']} | "
                time_summary_text += f"{row['legitimate_requests']} ({row['legitimate_pct']:.1f}%) | "
                time_summary_text += f"{row['illegitimate_requests']} ({row['illegitimate_pct']:.1f}%) | "
                time_summary_text += f"{row['human_requests']} ({row['human_pct']:.1f}%) | "
                time_summary_text += f"{row['bot_requests']} ({row['bot_pct']:.1f}%) |\n"
            
            # Add a breakdown of the four categories
            time_summary_text += "\n### Detailed Classification by Hour\n\n"
            time_summary_text += "| Hour | Legitimate Human | Legitimate Bot | Illegitimate Human | Illegitimate Bot |\n"
            time_summary_text += "|------|-----------------|----------------|-------------------|------------------|\n"
            
            for _, row in time_summary.iterrows():
                hour_str = row['hour'].strftime("%Y-%m-%d %H:00")
                time_summary_text += f"| {hour_str} | {row['legitimate_human_requests']} | "
                time_summary_text += f"{row['legitimate_bot_requests']} | "
                time_summary_text += f"{row['illegitimate_human_requests']} | "
                time_summary_text += f"{row['illegitimate_bot_requests']} |\n"
        
        summary += f"\n## Time-Based Analysis\n\n{time_summary_text}\n"
        
        # Add custom prompt-based analysis if provided
        if prompt:
            summary += f"\n## Custom Analysis\n\n"
            
            # Generate custom analysis using the model
            custom_prompt = f"{prompt}\n\nAnalyze the following web server log data:\n\n{df_classified.head(10).to_string()}\n\nProvide your analysis:"
            
            input_ids = self.tokenizer(custom_prompt, return_tensors="pt").input_ids
            
            with torch.no_grad():
                output_ids = self.model.generate(
                    input_ids,
                    max_new_tokens=500,
                    temperature=0.3,
                    top_p=0.9,
                    do_sample=True
                )
            
            custom_analysis = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # Extract just the model's response (after the prompt)
            custom_response = custom_analysis[len(custom_prompt):].strip()
            
            summary += f"\n{custom_response}\n"
        
        summary += f"""
    ## Additional Notes

    This analysis was generated automatically based on machine learning classification of access logs. While the system aims for high accuracy, manual review is recommended for critical security decisions.

    Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    """
        
        return summary

def package_pipeline(model_path, threat_patterns_path=None, output_path=None, status_code_whitelist=None, status_code_blacklist=None):
    """
    Package a trained model into a self-contained pipeline.
    
    Args:
        model_path: Path to the trained model directory
        threat_patterns_path: Path to the malicious request patterns file
        output_path: Path to save the packaged pipeline
        status_code_whitelist: Regex pattern for status codes to include (only these will be processed)
        status_code_blacklist: Regex pattern for status codes to exclude (all except these will be processed)
        
    Returns:
        str: Path to the packaged pipeline
    """
    print(f"Packaging model from {model_path}...")
    
    # Create the pipeline
    pipeline = LogClassificationPipeline(
        model_path, 
        threat_patterns_path,
        status_code_whitelist,
        status_code_blacklist
    )
    
    # Generate output path if not provided
    if not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"pipeline_model/pipeline_{timestamp}.pkl"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the pipeline
    print(f"Saving pipeline to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    print(f"Pipeline successfully packaged to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Package a trained model into a self-contained pipeline')
    parser.add_argument('--model', required=True, help='Path to the trained model directory')
    parser.add_argument('--threat-patterns', default='src/threat_int/malicious_req_patterns.txt', 
                        help='Path to the malicious request patterns file')
    parser.add_argument('--output', help='Path to save the packaged pipeline')
    parser.add_argument('--status-code-whitelist', help='Regex pattern for status codes to include (only these will be processed)')
    parser.add_argument('--status-code-blacklist', help='Regex pattern for status codes to exclude (all except these will be processed)')
    args = parser.parse_args()
    
    package_pipeline(
        args.model, 
        args.threat_patterns, 
        args.output,
        args.status_code_whitelist,
        args.status_code_blacklist
    )

if __name__ == "__main__":
    main()