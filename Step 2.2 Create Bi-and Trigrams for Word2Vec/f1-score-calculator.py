# Data structure to hold the results
results = [
    {
        'threshold': 50,
        'min_count': 2,
        'bigrams': {'recall': 62.80, 'precision': 0.56},
        'trigrams': {'recall': 56.86, 'precision': 0.06}
    },
    {
        'threshold': 50,
        'min_count': 3,
        'bigrams': {'recall': 59.18, 'precision': 0.89},
        'trigrams': {'recall': 49.02, 'precision': 0.08}
    },
    {
        'threshold': 200,
        'min_count': 2,
        'bigrams': {'recall': 60.63, 'precision': 0.95},
        'trigrams': {'recall': 43.14, 'precision': 0.08}
    },
    {
        'threshold': 50,
        'min_count': 4,
        'bigrams': {'recall': 53.86, 'precision': 1.16},
        'trigrams': {'recall': 45.10, 'precision': 0.11}
    },
    {
        'threshold': 200,
        'min_count': 3,
        'bigrams': {'recall': 57.00, 'precision': 1.53},
        'trigrams': {'recall': 37.25, 'precision': 0.12}
    },
    {
        'threshold': 400,
        'min_count': 2,
        'bigrams': {'recall': 58.94, 'precision': 1.20},
        'trigrams': {'recall': 33.33, 'precision': 0.09}
    },
    {
        'threshold': 200,
        'min_count': 4,
        'bigrams': {'recall': 52.17, 'precision': 2.00},
        'trigrams': {'recall': 31.37, 'precision': 0.14}
    },
    {
        'threshold': 400,
        'min_count': 3,
        'bigrams': {'recall': 55.07, 'precision': 1.92},
        'trigrams': {'recall': 27.45, 'precision': 0.12}
    },
    {
        'threshold': 400,
        'min_count': 4,
        'bigrams': {'recall': 50.72, 'precision': 2.51},
        'trigrams': {'recall': 23.53, 'precision': 0.14}
    }
]

def calculate_f1(precision, recall):
    """Calculate F1 score from precision and recall."""
    if precision == 0 or recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Calculate F1 scores and find best parameters
best_bigram_f1 = 0
best_trigram_f1 = 0
best_bigram_params = None
best_trigram_params = None

print("Detailed Results:")
print("-" * 100)
print(f"{'Threshold':^10} | {'Min Count':^10} | {'Type':^8} | {'Precision':^10} | {'Recall':^10} | {'F1 Score':^10}")
print("-" * 100)

for result in results:
    # Calculate F1 scores
    bigram_f1 = calculate_f1(result['bigrams']['precision'], result['bigrams']['recall'])
    trigram_f1 = calculate_f1(result['trigrams']['precision'], result['trigrams']['recall'])
    
    # Print detailed results
    print(f"{result['threshold']:^10} | {result['min_count']:^10} | {'Bigram':^8} | "
          f"{result['bigrams']['precision']:^10.2f} | {result['bigrams']['recall']:^10.2f} | {bigram_f1:^10.2f}")
    print(f"{result['threshold']:^10} | {result['min_count']:^10} | {'Trigram':^8} | "
          f"{result['trigrams']['precision']:^10.2f} | {result['trigrams']['recall']:^10.2f} | {trigram_f1:^10.2f}")
    
    # Update best scores
    if bigram_f1 > best_bigram_f1:
        best_bigram_f1 = bigram_f1
        best_bigram_params = (result['threshold'], result['min_count'])
    
    if trigram_f1 > best_trigram_f1:
        best_trigram_f1 = trigram_f1
        best_trigram_params = (result['threshold'], result['min_count'])

print("\nBest Parameters:")
print(f"Bigrams: threshold={best_bigram_params[0]}, min_count={best_bigram_params[1]}, F1={best_bigram_f1:.2f}")
print(f"Trigrams: threshold={best_trigram_params[0]}, min_count={best_trigram_params[1]}, F1={best_trigram_f1:.2f}")
