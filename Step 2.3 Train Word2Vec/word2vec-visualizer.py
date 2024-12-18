import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from gensim.models import Word2Vec
import os
import re

os.environ['LOKY_MAX_CPU_COUNT'] = '16'

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def clean_text_for_matplotlib(text):
    """
    Clean text to avoid matplotlib interpretation of special characters
    """
    # Replace $ with \$ to escape it
    text = text.replace('$', '\$')
    # Add more replacements if needed
    return text

def visualize_word2vec(model, target_words, n_neighbors=10, figsize=(15, 15)):
    """
    Create visualization for word2vec model focusing on target words (rappers) and their neighbors.
    
    Parameters:
    - model: trained word2vec model
    - target_words: list of words to focus on (e.g., known rapper names)
    - n_neighbors: number of nearest neighbors to find for each target word
    - figsize: size of the output plot
    """
    # Get words and their vectors
    words = []
    vectors = []
    word_to_neighbors = defaultdict(list)
    
    # For each target word, get its nearest neighbors
    for word in target_words:
        if word in model.wv.key_to_index:
            # Get the word vector
            vectors.append(model.wv[word])
            words.append(word)
            
            # Find nearest neighbors
            similar_words = model.wv.most_similar(word, topn=n_neighbors)
            for similar_word, similarity in similar_words:
                vectors.append(model.wv[similar_word])
                words.append(similar_word)
                word_to_neighbors[word].append((similar_word, similarity))
    
    if not vectors:
        print("No valid words found in the model's vocabulary!")
        return
    
    # Convert vectors to numpy array
    vectors = np.array(vectors)
    
    # Calculate appropriate perplexity (should be smaller than n_samples - 1)
    n_samples = len(vectors)
    perplexity = min(30, max(5, n_samples // 5))  # Adaptive perplexity
    
    # Reduce dimensionality with t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    vectors_2d = tsne.fit_transform(vectors)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    # Plot all points
    n_target_words = sum(1 for word in target_words if word in model.wv.key_to_index)
    plt.scatter(vectors_2d[:n_target_words, 0], vectors_2d[:n_target_words, 1], 
                c='red', label='Target Words', s=100)
    plt.scatter(vectors_2d[n_target_words:, 0], vectors_2d[n_target_words:, 1], 
                c='blue', label='Related Words', alpha=0.5)
    
    # Add labels for all points, with cleaned text
    for i, word in enumerate(words):
        clean_word = clean_text_for_matplotlib(word)
        plt.annotate(clean_word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title("Word2Vec Visualization of Rappers and Related Terms")
    plt.legend()
    
    # Instead of tight_layout(), use more specific spacing
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()
    
    # Print associated terms for each target word
    print("\nMost similar terms for each target word:")
    for word in target_words:
        if word in word_to_neighbors:
            print(f"\n{word}:")
            for neighbor, similarity in word_to_neighbors[word]:
                print(f"  {neighbor}: {similarity:.3f}")

def create_similarity_heatmap(model, words, figsize=(12, 8)):
    """
    Create a heatmap showing similarities between words.
    
    Parameters:
    - model: trained word2vec model
    - words: list of words to compare
    - figsize: size of the output plot
    """
    # Filter out words not in vocabulary
    valid_words = [word for word in words if word in model.wv]
    if not valid_words:
        print("No valid words found in the model's vocabulary!")
        return
    
    # Calculate similarity matrix
    similarity_matrix = np.zeros((len(valid_words), len(valid_words)))
    for i, word1 in enumerate(valid_words):
        for j, word2 in enumerate(valid_words):
            similarity_matrix[i][j] = model.wv.similarity(word1, word2)
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, annot=True, cmap='YlOrRd', 
                xticklabels=[clean_text_for_matplotlib(w) for w in valid_words], 
                yticklabels=[clean_text_for_matplotlib(w) for w in valid_words])
    plt.title("Word Similarity Heatmap")
    
    # Instead of tight_layout(), use more specific spacing
    plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
    plt.show()


model = Word2Vec.load('word2vec_model.model')


print("Vocabulary size:", len(model.wv.key_to_index))
print("Sample words from vocabulary:", list(model.wv.key_to_index.keys())[:10])


with open('../Supporting - List of Rappers/Spotify PLaylist scrape/all_artists.txt', 'r', encoding='utf-8') as f:
    # Call lower() method on each line
    rappers = [line.strip().lower() for line in f if line.strip()]
    # replace spaces with underscores
    rappers = [re.sub(r'\s+', '_', rapper) for rapper in rappers]
    print(type(rappers))

# List of known rappers to start with
#rappers = ["kollegah", "bushido", "sido", "capital bra", 'azudemsk', 'og_keemo', ]  # Add your initial list

# Create the main visualization
visualize_word2vec(model, rappers, n_neighbors=8)

# Create a heatmap for the most frequently mentioned rappers
top_rappers = ["kollegah", "bushido", "sido", "capital bra"]  # Add your most mentioned rappers
create_similarity_heatmap(model, top_rappers)