import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import pandas as pd
from gensim.models import Word2Vec
import re
import numpy as np
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def create_interactive_viz(model, words=None):
    """
    Create an interactive visualization of word vectors using Plotly
    """
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

    # Perform t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embedding = tsne.fit_transform(word_vectors)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'word': words,
        'x': embedding[:, 0],
        'y': embedding[:, 1]
    })
    
    # Create interactive plot
    fig = px.scatter(
        df, x='x', y='y', text='word',
        title='Interactive Word2Vec Embeddings',
        template='plotly_white'
    )
    
    # Update traces for better visualization
    fig.update_traces(
        textposition='top center',
        marker=dict(size=10),
        hovertemplate='<b>%{text}</b><extra></extra>'
    )
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        width=1000,
        height=800
    )
    
    return fig

# Load rapper names
with open('../Supporting - List of Rappers/Spotify Playlist scrape/all_artists.txt', 'r', encoding='utf-8') as f:
    # Clean and process rapper names
    rapper_names = [re.sub(r'\s+', '_', line.strip().lower()) for line in f if line.strip()]
    print(f"Loaded {len(rapper_names)} rapper names")

# Load model and visualize
model = Word2Vec.load('word2vec_model.model')

fig = create_interactive_viz(model, rapper_names)
# Show in browser
fig.show()
# Save as HTML
fig.write_html("rapper_embeddings.html")

# Example usage:
"""
# Create visualization
fig = create_interactive_viz(model, rapper_names)
# Show in browser
fig.show()
# Save as HTML
fig.write_html("rapper_embeddings.html")
"""