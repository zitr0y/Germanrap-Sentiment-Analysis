import os
from openTSNE import TSNE
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from gensim.models import Word2Vec
import re
import time

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def clean_text_for_display(text):
    """
    Clean text to make it suitable for matplotlib display
    """
    # Replace dollar signs with 's' to avoid math mode triggers
    text = text.replace('$', 's')
    # Remove or replace emojis and other special characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text

def visualize_word2vec(model, words=None, n_components=2, perplexity=30):
    """
    Visualize word vectors using t-SNE
    """
    start_time = time.time()
    
    # Filter words to only include those in the model's vocabulary
    if words is not None:
        existing_words = [word for word in words if word in model.wv]
        if len(existing_words) < len(words):
            print(f"Warning: {len(words) - len(existing_words)} words were not found in the model's vocabulary")
            print("Missing words:", set(words) - set(existing_words))
        words = existing_words
    else:
        words = [word for word in model.wv.index_to_key]
    
    if not words:
        print("No valid words found in the model's vocabulary!")
        return
        
    word_vectors = np.array([model.wv[word] for word in words])
    
    # Clean words for display
    display_words = [clean_text_for_display(word) for word in words]
    
    # Perform t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=min(perplexity, len(words) - 1),  # Ensure perplexity is valid
        random_state=42,
        n_jobs=16
    )
    embedding = tsne.fit(word_vectors)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    if n_components == 2:
        # 2D plot
        sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], alpha=0.6)
        
        # Add word labels
        for i, word in tqdm(enumerate(display_words)):
            if word.strip():  # Only add label if there's text after cleaning
                plt.annotate(word, (embedding[i, 0], embedding[i, 1]), alpha=0.7)
    
    elif n_components == 3:
        # 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], alpha=0.6)
        
        for i, word in tqdm(enumerate(display_words)):
            if word.strip():  # Only add label if there's text after cleaning
                ax.text(embedding[i, 0], embedding[i, 1], embedding[i, 2], word, alpha=0.7)
    
    print(f"Time taken: {time.time() - start_time:.2f} seconds with {n_components} components and "
          f"perplexity {perplexity} and {len(words)} words and {len(word_vectors[0])} features and "
          f"{len(word_vectors)} vectors and {len(embedding)} embeddings and {tsne.n_jobs} jobs")

    plt.title(f'Word2Vec Embeddings Visualization ({len(words)} rappers)')
    plt.tight_layout()
    plt.show()

# Load rapper names
with open('../Supporting - List of Rappers/Spotify PLaylist scrape/all_artists.txt', 'r', encoding='utf-8') as f:
    # Clean and process rapper names
    rapper_names = [re.sub(r'\s+', '_', line.strip().lower()) for line in f if line.strip()]
    print(f"Loaded {len(rapper_names)} rapper names")

# Load model and visualize
model = Word2Vec.load('word2vec_model.model')

# 2D visualization
visualize_word2vec(model, words=rapper_names)

# 3D visualization
visualize_word2vec(model, words=rapper_names, n_components=3)