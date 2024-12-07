from collections import Counter
from typing import List, Dict, Tuple
import re
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def get_ngrams(text: str, n: int) -> List[str]:
    """
    Extract n-grams from text.
    
    Args:
        text: Input text
        n: Length of n-gram
    Returns:
        List of n-grams
    """
    words = text.lower().split()
    ngrams = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    return ngrams

def analyze_ngrams(file_path: str, max_n: int = 3, min_count: int = 5) -> Dict[int, List[Tuple[str, int]]]:
    """
    Analyze n-gram frequencies from a file.
    
    Args:
        file_path: Path to input file
        max_n: Maximum n-gram length to analyze
        min_count: Minimum frequency to include in results
    Returns:
        Dictionary mapping n to list of (ngram, count) tuples
    """
    ngram_counters = {n: Counter() for n in range(1, max_n + 1)}
    total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total_lines, desc="Analyzing lines", unit="lines", ncols=100):
            line = line.strip()
            if not line:
                continue
                
            for n in range(1, max_n + 1):
                ngrams = get_ngrams(line, n)
                ngram_counters[n].update(ngrams)
    
    results = {}
    for n, counter in tqdm(ngram_counters.items(), desc="Processing results", 
                        unit="gram", ncols=100):
        frequent_ngrams = [(ngram, count) 
                          for ngram, count in tqdm(counter.items(), desc="Filtering n-grams", 
                                                unit="gram", leave=False, ncols=100) 
                          if count >= min_count]
        results[n] = sorted(frequent_ngrams, key=lambda x: (-x[1], x[0]))
    
    return results

def print_results(results: Dict[int, List[Tuple[str, int]]], top_k: int = 20):
    """Print n-gram analysis results."""
    for n, ngrams in results.items():
        print(f"\nTop {top_k} {n}-grams:")
        print("-" * 40)
        for ngram, count in ngrams[:top_k]:
            print(f"{count:5d}: {ngram}")

def plot_top_ngrams(results: Dict[int, List[Tuple[str, int]]], top_k: int = 10):
    """
    Create visualizations for n-gram frequencies.
    
    Args:
        results: Dictionary mapping n to list of (ngram, count) tuples
        top_k: Number of top results to show in plots
    """
    n_types = len(results)
    fig_height = 6 * n_types
    
    # Create figure with larger size for better readability
    plt.figure(figsize=(15, fig_height))
    
    # Set style parameters manually
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#dddddd'
    plt.rcParams['grid.linestyle'] = '--'
    
    # Generate colors using seaborn's color_palette
    colors = sns.color_palette("husl", n_types)
    
    for idx, (n, ngrams) in enumerate(results.items(), 1):
        # Get top k ngrams
        top_ngrams = ngrams[:top_k]
        
        # Create subplot
        plt.subplot(n_types, 1, idx)
        
        # Prepare data
        labels = [ngram for ngram, _ in top_ngrams]
        values = [count for _, count in top_ngrams]
        
        # Create horizontal bars
        bars = plt.barh(range(len(labels)), values, color=colors[idx-1], alpha=0.8)
        
        # Customize appearance
        plt.grid(True, axis='x', alpha=0.3)
        plt.yticks(range(len(labels)), labels)
        
        # Add value labels on the bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width * 1.02, bar.get_y() + bar.get_height()/2, 
                    f'{int(width):,}', 
                    ha='left', va='center', fontweight='bold')
        
        # Add titles and labels
        plt.title(f'Top {top_k} {n}-grams', pad=20, fontsize=12, fontweight='bold')
        plt.xlabel('Frequency')
    
    plt.tight_layout(pad=3.0)
    
    # Save the plot
    plt.savefig('ngram_frequencies.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved as 'ngram_frequencies.png'")
    plt.close()

def plot_frequency_distribution(results: Dict[int, List[Tuple[str, int]]]):
    """
    Create distribution plots for n-gram frequencies.
    
    Args:
        results: Dictionary mapping n to list of (ngram, count) tuples
    """
    plt.figure(figsize=(15, 6))
    
    # Set style parameters
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['grid.color'] = '#dddddd'
    plt.rcParams['grid.linestyle'] = '--'
    
    # Generate colors
    colors = sns.color_palette("husl", len(results))
    
    for (n, ngrams), color in zip(results.items(), colors):
        frequencies = [count for _, count in ngrams]
        
        # Plot frequency distribution
        sns.kdeplot(data=frequencies, label=f'{n}-grams', color=color)
    
    plt.grid(True, alpha=0.3)
    plt.title('N-gram Frequency Distributions', fontsize=12, fontweight='bold')
    plt.xlabel('Frequency')
    plt.ylabel('Density')
    plt.legend()
    
    # Use log scale for better visualization
    plt.xscale('log')
    
    # Save the plot
    plt.savefig('ngram_distributions.png', dpi=300, bbox_inches='tight')
    print("Distribution plot saved as 'ngram_distributions.png'")
    plt.close()

if __name__ == "__main__":
    input_file = "2-processed_sentences.txt"
    
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' not found!")
        exit(1)
    
    # Analysis parameters
    MAX_N = 10
    MIN_COUNT = 5
    TOP_K = 20
    
    print(f"Analyzing n-grams in {input_file}...")
    results = analyze_ngrams(input_file, max_n=MAX_N, min_count=MIN_COUNT)
    
    # Print text results
    print_results(results, top_k=TOP_K)
    
    # Create visualizations
    plot_top_ngrams(results, top_k=15)
    plot_frequency_distribution(results)

    print("\nAnalysis complete!")