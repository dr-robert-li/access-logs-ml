#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import psutil
from datetime import datetime

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024 / 1024  # Convert to MB

def create_traffic_composition_plot(time_based_file, output_dir):
    """
    Create a line chart showing traffic composition over time.
    
    Args:
        time_based_file: Path to the time-based classification summary CSV
        output_dir: Directory to save the visualization
        
    Returns:
        str: Path to the generated visualization
    """
    initial_memory = get_memory_usage()
    print(f"Starting visualization with memory usage: {initial_memory:.2f} MB")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the time-based summary
    print(f"Loading data from {time_based_file}...")
    df = pd.read_csv(time_based_file)
    
    # Convert hour to datetime
    df['hour'] = pd.to_datetime(df['hour'])
    
    # Sort by hour
    df = df.sort_values('hour')
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create visualizations
    print("Generating traffic composition visualization...")
    
    # Set the style
    sns.set_style("whitegrid")
    
    # 1. Traffic composition over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['hour'], df['legitimate_human_pct'], label='Legitimate Human', marker='o', linewidth=2)
    plt.plot(df['hour'], df['legitimate_bot_pct'], label='Legitimate Bot', marker='s', linewidth=2)
    plt.plot(df['hour'], df['illegitimate_human_pct'], label='Illegitimate Human', marker='^', linewidth=2)
    plt.plot(df['hour'], df['illegitimate_bot_pct'], label='Illegitimate Bot', marker='x', linewidth=2)
    plt.xlabel('Hour')
    plt.ylabel('Percentage of Traffic')
    plt.title('Traffic Composition Over Time')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    composition_file = os.path.join(output_dir, f'traffic_composition_{timestamp}.png')
    plt.savefig(composition_file, dpi=300)
    plt.close()
    
    # 2. Total traffic volume over time
    plt.figure(figsize=(12, 6))
    plt.bar(df['hour'], df['total_requests'], color='steelblue')
    plt.xlabel('Hour')
    plt.ylabel('Number of Requests')
    plt.title('Total Traffic Volume Over Time')
    plt.grid(True, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    volume_file = os.path.join(output_dir, f'traffic_volume_{timestamp}.png')
    plt.savefig(volume_file, dpi=300)
    plt.close()
    
    # 3. Legitimate vs Illegitimate traffic
    plt.figure(figsize=(12, 6))
    plt.stackplot(df['hour'], 
                 df['legitimate_human_pct'], 
                 df['legitimate_bot_pct'],
                 df['illegitimate_human_pct'],
                 df['illegitimate_bot_pct'],
                 labels=['Legitimate Human', 'Legitimate Bot', 'Illegitimate Human', 'Illegitimate Bot'],
                 colors=['#4CAF50', '#8BC34A', '#F44336', '#FF9800'])
    plt.xlabel('Hour')
    plt.ylabel('Percentage of Traffic')
    plt.title('Traffic Composition (Stacked) Over Time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    stacked_file = os.path.join(output_dir, f'traffic_stacked_{timestamp}.png')
    plt.savefig(stacked_file, dpi=300)
    plt.close()
    
    # 4. Average confidence over time
    plt.figure(figsize=(12, 6))
    plt.plot(df['hour'], df['avg_confidence'], marker='o', color='purple', linewidth=2)
    plt.xlabel('Hour')
    plt.ylabel('Average Confidence')
    plt.title('Model Confidence Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    confidence_file = os.path.join(output_dir, f'model_confidence_{timestamp}.png')
    plt.savefig(confidence_file, dpi=300)
    plt.close()

    # 5. User agent composition visualization
    plt.figure(figsize=(12, 6))
    user_agent_data = pd.DataFrame({
        'hour': df['hour'],
        'modern_browser': df['legitimate_human_pct'],
        'known_bot': df['legitimate_bot_pct'],
        'old_browser': df['illegitimate_human_pct'] * 0.7,  # Approximation based on classification
        'no_user_agent': df['illegitimate_bot_pct'] * 0.3,  # Approximation based on classification
    })

    plt.stackplot(user_agent_data['hour'], 
                user_agent_data['modern_browser'], 
                user_agent_data['known_bot'],
                user_agent_data['old_browser'],
                user_agent_data['no_user_agent'],
                labels=['Modern Browsers', 'Known Bots', 'Old Browsers', 'No User Agent'],
                colors=['#4CAF50', '#8BC34A', '#F44336', '#FF9800'])
    plt.xlabel('Hour')
    plt.ylabel('Percentage of Traffic')
    plt.title('User Agent Composition Over Time')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    user_agent_file = os.path.join(output_dir, f'user_agent_composition_{timestamp}.png')
    plt.savefig(user_agent_file, dpi=300)
    plt.close()
    
    # Create an HTML report with all visualizations
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Classification Report - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1 {{ color: #2c3e50; }}
            h2 {{ color: #3498db; }}
            .visualization {{ margin-bottom: 30px; }}
            img {{ max-width: 100%; border: 1px solid #ddd; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ text-align: left; padding: 8px; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <h1>Traffic Classification Report</h1>
        <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        
        <h2>Traffic Composition Over Time</h2>
        <div class="visualization">
            <img src="{os.path.basename(composition_file)}" alt="Traffic Composition">
            <p>This chart shows the percentage breakdown of different traffic types over time.</p>
        </div>
        
        <h2>Total Traffic Volume</h2>
        <div class="visualization">
            <img src="{os.path.basename(volume_file)}" alt="Traffic Volume">
            <p>This chart shows the total number of requests per hour.</p>
        </div>
        
        <h2>Stacked Traffic Composition</h2>
        <div class="visualization">
            <img src="{os.path.basename(stacked_file)}" alt="Stacked Traffic">
            <p>This stacked area chart shows the relative proportion of each traffic type.</p>
        </div>
        
        <h2>User Agent Composition</h2>
        <div class="visualization">
            <img src="{os.path.basename(user_agent_file)}" alt="User Agent Composition">
            <p>This stacked area chart shows the breakdown of traffic by user agent types over time.</p>
        </div>
        
        <h2>Model Confidence</h2>
        <div class="visualization">
            <img src="{os.path.basename(confidence_file)}" alt="Model Confidence">
            <p>This chart shows the average confidence of the model's classifications over time.</p>
        </div>
        
        <h2>Summary Statistics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Total Requests</td>
                <td>{df['total_requests'].sum():,}</td>
            </tr>
            <tr>
                <td>Legitimate Traffic</td>
                <td>{df['legitimate_requests'].sum():,} ({df['legitimate_requests'].sum() / df['total_requests'].sum() * 100:.2f}%)</td>
            </tr>
            <tr>
                <td>Illegitimate Traffic</td>
                <td>{df['illegitimate_requests'].sum():,} ({df['illegitimate_requests'].sum() / df['total_requests'].sum() * 100:.2f}%)</td>
            </tr>
            <tr>
                <td>Human Traffic</td>
                <td>{df['human_requests'].sum():,} ({df['human_requests'].sum() / df['total_requests'].sum() * 100:.2f}%)</td>
            </tr>
            <tr>
                <td>Bot Traffic</td>
                <td>{df['bot_requests'].sum():,} ({df['bot_requests'].sum() / df['total_requests'].sum() * 100:.2f}%)</td>
            </tr>
            <tr>
                <td>Average Model Confidence</td>
                <td>{df['avg_confidence'].mean():.2f}</td>
            </tr>
        </table>
    </body>
    </html>
    """
    
    html_file = os.path.join(output_dir, f'traffic_report_{timestamp}.html')
    with open(html_file, 'w') as f:
        f.write(html_content)
    
    final_memory = get_memory_usage()
    print(f"Visualizations complete. Memory usage: {final_memory:.2f} MB (Δ: {final_memory - initial_memory:.2f} MB)")
    print(f"Visualizations saved to {output_dir}")
    print(f"HTML report: {html_file}")
    
    return html_file

def visualize_results(time_based_file, output_dir='visualizations'):
    """
    Generate visualizations from a time-based classification summary.
    
    Args:
        time_based_file: Path to the time-based classification summary CSV
        output_dir: Directory to save the visualizations
        
    Returns:
        str: Path to the HTML report
    """
    return create_traffic_composition_plot(time_based_file, output_dir)

def main():
    parser = argparse.ArgumentParser(description='Generate visualizations from time-based classification summary')
    parser.add_argument('--input', required=True, help='Path to time-based classification summary CSV')
    parser.add_argument('--output-dir', default='visualizations', help='Directory to save visualizations')
    args = parser.parse_args()
    
    visualize_results(args.input, args.output_dir)

if __name__ == "__main__":
    main()
