import logging
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import time
from tqdm import tqdm
import os
import multiprocessing as mp
from typing import List, Tuple, Dict, Union

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))

print(f'{mp.cpu_count()} CPU cores available')

# Set up logging to monitor the process
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO
)

def load_sentences(file_path):
    """Load and tokenize sentences from file."""
    print("Loading sentences...")
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading lines", unit="lines", ncols=100):
            if line.strip():  # Skip empty lines
                tokens = line.strip().split()
                if tokens:  # Only add if there are tokens
                    sentences.append(tokens)
    print(f"Loaded {len(sentences)} sentences")
    return sentences

def create_ngrams(sentences, save=False):
    """
    Create bigrams and trigrams from sentences with detailed logging.
    """
    print("\n\n\nCreating bigrams...\n")
    # Build bigram model
    bigram = Phrases(
        sentences,
        min_count=4,    # Minimum occurrences of bigram
        threshold=400,   # Higher means fewer phrases
        delimiter='_'
    )
    # Create faster model
    bigram_model = Phraser(bigram)
    print(f"Number of bigrams found: {len(bigram_model.phrasegrams)}")

    if save:
        # Save bigrams with more detailed information
        with open('bigrams.txt', 'w', encoding='utf-8') as f:
            f.write("Format: (word1_word2): score\n\n")
            for phrase, score in sorted(bigram_model.phrasegrams.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{phrase}: {score}\n")
        print("Bigrams saved to bigrams.txt")
    if not save:
        print("ngrams not saved. Set 'save=True' to save them.")
    
    print("\n\n\nCreating trigrams...\n")
    trigram = Phrases(
        bigram_model[sentences],
        min_count=3,      # Can be lower for trigrams as they're naturally rarer
        threshold=200,     # Can be lower for trigrams
        delimiter='_'
    )
    trigram_model = Phraser(trigram)
    print(f"Number of trigrams found: {len(trigram_model.phrasegrams)}")

    if save:
        # Save trigrams with more detailed information
        with open('trigrams.txt', 'w', encoding='utf-8') as f:
            f.write("Format: (word1_word2_word3): score\n\n")
            for phrase, score in sorted(trigram_model.phrasegrams.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{phrase}: {score}\n")
        print("Trigrams saved to trigrams.txt")
    if not save:
        print("ngrams not saved. Set 'save=True' to save them.")

    # Transform sentences and add debugging information
    print("\nTransforming sentences with ngrams...")
    sentences_with_ngrams = []
    sample_size = min(5, len(sentences))  # Show first 5 sentences as samples
    
    for i, sent in enumerate(sentences):
        transformed = trigram_model[bigram_model[sent]]
        sentences_with_ngrams.append(transformed)
        
        # Print some sample transformations
        if i < sample_size:
            print(f"\nOriginal sentence {i+1}: {' '.join(sent)}")
            print(f"Transformed sentence {i+1}: {' '.join(transformed)}")
    
    # Save a sample of transformations for verification
    with open('sample_transformations.txt', 'w', encoding='utf-8') as f:
        f.write("Sample of original vs transformed sentences:\n\n")
        for i in range(min(20, len(sentences))):  # Save first 20 examples
            f.write(f"Original: {' '.join(sentences[i])}\n")
            f.write(f"Transformed: {' '.join(sentences_with_ngrams[i])}\n")
            f.write("-" * 50 + "\n")
    
    print(f"\nProcessed {len(sentences)} sentences")
    print("Sample transformations saved to sample_transformations.txt")
    
    return sentences_with_ngrams

def test_thresholds(sentences):
    """Test different threshold combinations for bigram and trigram creation."""
    # Load artist names and create reference sets
    with open('../Supporting - List of Rappers/Spotify PLaylist scrape/all_artists.txt', 'r', encoding='utf-8') as f:
        artists = [line.strip() for line in f if line.strip()]
    
    # Create sets of multi-word artists
    bigram_artists = {artist.lower() for artist in artists if len(artist.split()) == 2}
    trigram_artists = {artist.lower() for artist in artists if len(artist.split()) == 3}
    
    print(f"Found {len(bigram_artists)} bigram artists and {len(trigram_artists)} trigram artists")
    
    # Test parameters
    threshold_values = [50, 200, 400]
    min_count_values = [2, 3, 4]
    
    best_bigram_score = 0
    best_bigram_params = None
    best_trigram_score = 0
    best_trigram_params = None
    
    results = []
    
    # Test different combinations
    for threshold in tqdm(threshold_values, desc="Testing thresholds", ncols=100):
        for min_count in tqdm(min_count_values, desc="Testing min counts", ncols=100, leave=False):
            print(f"\nTesting threshold={threshold}, min_count={min_count}")
            
            # Create bigrams
            bigram = Phrases(
                sentences,
                min_count=min_count,
                threshold=threshold,
                delimiter='_'
            )
            bigram_model = Phraser(bigram)
            
            # Create trigrams
            trigram = Phrases(
                bigram_model[sentences],
                min_count=min_count,
                threshold=threshold,
                delimiter='_'
            )
            trigram_model = Phraser(trigram)
            
            # Get all generated n-grams
            all_bigrams = set(bigram_model.phrasegrams.keys())
            all_trigrams = set(trigram_model.phrasegrams.keys())
            
            # Count found artists
            found_bigrams = set()
            found_trigrams = set()
            
            # Check sentences for found artists
            for sent in sentences:
                # Transform sentence
                bigram_sent = ' '.join(bigram_model[sent])
                trigram_sent = ' '.join(trigram_model[bigram_model[sent]])
                
                # Check for bigram artists
                for artist in bigram_artists:
                    if artist.replace(' ', '_') in bigram_sent.lower():
                        found_bigrams.add(artist)
                
                # Check for trigram artists
                for artist in trigram_artists:
                    if artist.replace(' ', '_') in trigram_sent.lower():
                        found_trigrams.add(artist)
            
            # Calculate scores
            bigram_score = len(found_bigrams) / len(bigram_artists) if bigram_artists else 0
            trigram_score = len(found_trigrams) / len(trigram_artists) if trigram_artists else 0
            
            # Calculate precision (what percentage of found n-grams are artist names)
            bigram_precision = len(found_bigrams) / len(all_bigrams) if all_bigrams else 0
            trigram_precision = len(found_trigrams) / len(all_trigrams) if all_trigrams else 0
            
            results.append({
                'threshold': threshold,
                'min_count': min_count,
                'bigram_score': bigram_score,
                'trigram_score': trigram_score,
                'found_bigrams': len(found_bigrams),
                'total_bigrams': len(bigram_artists),
                'found_trigrams': len(found_trigrams),
                'total_trigrams': len(trigram_artists),
                'all_bigrams': len(all_bigrams),
                'all_trigrams': len(all_trigrams),
                'bigram_precision': bigram_precision,
                'trigram_precision': trigram_precision
            })
            
            # Update best scores
            if bigram_score > best_bigram_score:
                best_bigram_score = bigram_score
                best_bigram_params = (threshold, min_count)
            
            if trigram_score > best_trigram_score:
                best_trigram_score = trigram_score
                best_trigram_params = (threshold, min_count)
    
    # Print results
    print("\nResults Summary:")
    print("-" * 100)
    for result in sorted(results, key=lambda x: (x['bigram_score'] + x['trigram_score'])/2, reverse=True):
        print(f"Threshold: {result['threshold']}, Min Count: {result['min_count']}")
        print(f"Bigrams:")
        print(f"  - Recall: {result['bigram_score']:.2%} ({result['found_bigrams']}/{result['total_bigrams']} artists found)")
        print(f"  - Total n-grams generated: {result['all_bigrams']}")
        print(f"  - Precision: {result['bigram_precision']:.2%} (artist names / total bigrams)")
        print(f"Trigrams:")
        print(f"  - Recall: {result['trigram_score']:.2%} ({result['found_trigrams']}/{result['total_trigrams']} artists found)")
        print(f"  - Total n-grams generated: {result['all_trigrams']}")
        print(f"  - Precision: {result['trigram_precision']:.2%} (artist names / total trigrams)")
        print("-" * 100)
    
    print(f"\nBest Bigram Parameters: threshold={best_bigram_params[0]}, min_count={best_bigram_params[1]}")
    print(f"Best Bigram Score: {best_bigram_score:.2%}")
    print(f"\nBest Trigram Parameters: threshold={best_trigram_params[0]}, min_count={best_trigram_params[1]}")
    print(f"Best Trigram Score: {best_trigram_score:.2%}")
    
    return best_bigram_params, best_trigram_params

def test(sentences):
    start_time = time.time()
    
    print("Starting threshold testing...")
    best_bigram_params, best_trigram_params = test_thresholds(sentences)
    
    # Print total time taken
    elapsed_time = time.time() - start_time
    print(f"\nTotal time taken: {elapsed_time/60:.2f} minutes")
    
    return best_bigram_params, best_trigram_params


def load_sentences_with_timestamps(file_path):
    """Load and tokenize sentences while preserving timestamps."""
    print("Loading sentences with timestamps...")
    sentences_with_timestamps = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading lines", unit="lines", ncols=100):
            if line.strip():
                sentence, timestamp = line.strip().split('\t')  # Using tab as separator for input
                tokens = sentence.strip().split()
                if tokens:  # Only add if there are tokens
                    sentences_with_timestamps.append((tokens, int(timestamp)))
    print(f"Loaded {len(sentences_with_timestamps)} sentences")
    return sentences_with_timestamps

def create_ngrams_with_timestamps(sentences_with_timestamps, save=False):
    """
    Create bigrams and trigrams from sentences while preserving timestamps.
    Saves sentences and timestamp mapping separately.
    """
    # Extract just the sentences for training the phrase models
    sentences = [sent for sent, _ in sentences_with_timestamps]
    
    print("\nCreating bigrams...")
    # Build bigram model
    bigram = Phrases(
        sentences,
        min_count=4,    
        threshold=400,   
        delimiter='_'
    )
    bigram_model = Phraser(bigram)
    print(f"Number of bigrams found: {len(bigram_model.phrasegrams)}")

    if save:
        with open('bigrams.txt', 'w', encoding='utf-8') as f:
            f.write("Format: (word1_word2): score\n\n")
            for phrase, score in sorted(bigram_model.phrasegrams.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{phrase}: {score}\n")
    
    print("\nCreating trigrams...")
    trigram = Phrases(
        bigram_model[sentences],
        min_count=3,      
        threshold=200,     
        delimiter='_'
    )
    trigram_model = Phraser(trigram)
    print(f"Number of trigrams found: {len(trigram_model.phrasegrams)}")

    if save:
        with open('trigrams.txt', 'w', encoding='utf-8') as f:
            f.write("Format: (word1_word2_word3): score\n\n")
            for phrase, score in sorted(trigram_model.phrasegrams.items(), key=lambda x: x[1], reverse=True):
                f.write(f"{phrase}: {score}\n")

    # Transform sentences while preserving timestamps
    print("\nTransforming sentences with ngrams...")
    sentences_with_ngrams = []
    sample_size = min(5, len(sentences_with_timestamps))
    
    for i, (sent, timestamp) in enumerate(sentences_with_timestamps):
        transformed = trigram_model[bigram_model[sent]]
        sentences_with_ngrams.append((' '.join(transformed), timestamp))
        
        if i < sample_size:
            print(f"\nOriginal sentence {i+1}: {' '.join(sent)}")
            print(f"Transformed sentence {i+1}: {' '.join(transformed)}")
    
    # Save transformed sentences (without timestamps) - matches your current format
    with open('2_2-sentences_with_ngrams.txt', 'w', encoding='utf-8') as f:
        for sentence, _ in sentences_with_ngrams:
            f.write(f"{sentence}\n")
    
    # Save timestamp mapping separately
    with open('sentence_timestamps_mapping.txt', 'w', encoding='utf-8') as f:
        for sentence, timestamp in sentences_with_ngrams:
            f.write(f"{sentence}\t{timestamp}\n")
    
    print(f"\nProcessed {len(sentences_with_ngrams)} sentences")
    return sentences_with_ngrams

def main():
    start_time = time.time()
    
    # Load sentences with timestamps
    sentences_with_timestamps = load_sentences_with_timestamps('2_1-processed_sentences_with_time.txt')

    # Create n-grams while preserving timestamps
    sentences_with_ngrams = create_ngrams_with_timestamps(sentences_with_timestamps, save=True)

    elapsed_time = time.time() - start_time
    print(f"\nTotal time taken: {elapsed_time/60:.2f} minutes")

    return True

if __name__ == "__main__":
    main()





