import plotly.express as px
import plotly.graph_objects as go
from sklearn.manifold import TSNE
import pandas as pd
from gensim.models import Word2Vec
import json
import re
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

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



class RapperAnalysisTools:
    def __init__(self, model_path: str, aliases_path: str = 'rapper_aliases.json'):
        """
        Initialize the analysis tools with Word2Vec model and aliases file
        
        Args:
            model_path: Path to the Word2Vec model
            aliases_path: Path to the JSON file storing rapper aliases
        """
        self.model = Word2Vec.load(model_path)
        self.aliases_path = aliases_path
        self.aliases = self.load_aliases()

    def preprocess_name(name: str) -> str:
        """
        Preprocess name to match the format in the model
        
        Args:
            name: Raw name string
            
        Returns:
            Preprocessed name string
        """
        # Convert to lowercase
        name = name.lower().strip()
        
        # Replace umlauts
        umlaut_mapping = {
            'ä': 'ae',
            'ö': 'oe',
            'ü': 'ue',
            'ß': 'ss'
        }
        for umlaut, replacement in umlaut_mapping.items():
            name = name.replace(umlaut, replacement)
        
        # Replace spaces with underscores and remove special characters
        name = re.sub(r'\s+', '_', name)
        
        return name

    
    def load_aliases(self) -> Dict[str, List[str]]:
        """Load aliases from JSON file, create if doesn't exist"""
        try:
            with open(self.aliases_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            with open(self.aliases_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=2)
            return {}

    def save_aliases(self):
        """Save aliases to JSON file"""
        with open(self.aliases_path, 'w', encoding='utf-8') as f:
            json.dump(self.aliases, f, indent=2, ensure_ascii=False)

    def add_rapper_aliases(self, main_name: str, aliases: List[str]):
        """
        Add or update rapper aliases
        """
        main_name = self.preprocess_name(main_name)
        aliases = [self.preprocess_name(alias) for alias in aliases]
        self.aliases[main_name] = aliases
        self.save_aliases()

    def find_similar_words(self, word: str, n: int = 10) -> List[Tuple[str, float]]:
        """Find similar words in the model's vocabulary"""
        processed_word = self.preprocess_name(word)
        if processed_word in self.model.wv:
            return self.model.wv.most_similar(processed_word, topn=n)
        return []

    def create_all_words_viz(self, min_freq: int = 50, max_words: Optional[int] = 1000):
        """Create visualization of all words meeting minimum frequency threshold"""
        words = [(word, self.model.wv.get_vecattr(word, "count"))
                for word in self.model.wv.index_to_key]
        
        words = sorted(words, key=lambda x: x[1], reverse=True)
        words = [w for w, f in words if f >= min_freq]
        if max_words:
            words = words[:max_words]
        
        fig = create_interactive_viz(self.model, words)
        return fig

    def batch_process_rappers(self, rapper_list_path: str, similarity_threshold: float = 0.5,
                            review_mode: str = 'interactive') -> Dict[str, List[Dict[str, float]]]:
        """
        Process a list of rappers and find potential aliases automatically
        """
        # Load and preprocess rapper names
        with open(rapper_list_path, 'r', encoding='utf-8') as f:
            rappers = [line.strip() for line in f if line.strip()]
        
        suggestions = {}
        not_found = []
        
        for rapper in rappers:
            original_name = rapper
            processed_name = self.preprocess_name(rapper)
            
            if processed_name not in self.model.wv:
                not_found.append(original_name)
                continue
                
            similar_words = self.find_similar_words(processed_name, n=20)
            potential_aliases = [
                {"word": word, "similarity": sim}
                for word, sim in similar_words
                if sim >= similarity_threshold
            ]
            
            if potential_aliases:
                suggestions[original_name] = potential_aliases
                
                if review_mode == 'interactive':
                    print(f"\nProcessing: {original_name}")
                    print(f"Processed name: {processed_name}")
                    print("Potential aliases found:")
                    for idx, alias in enumerate(potential_aliases, 1):
                        print(f"{idx}. {alias['word']} (similarity: {alias['similarity']:.3f})")
                    
                    print("\nEnter numbers of aliases to keep (comma-separated) or press Enter to skip:")
                    selection = input().strip()
                    
                    if selection:
                        try:
                            selected_indices = [int(i)-1 for i in selection.split(',')]
                            selected_aliases = [potential_aliases[i]['word'] for i in selected_indices]
                            self.add_rapper_aliases(original_name, selected_aliases)
                            print(f"Added aliases for {original_name}: {selected_aliases}")
                        except (ValueError, IndexError):
                            print("Invalid selection, skipping...")
                
                elif review_mode == 'auto':
                    aliases = [alias['word'] for alias in potential_aliases]
                    self.add_rapper_aliases(original_name, aliases)
        
        if not_found:
            print("\nRappers not found in model (showing original and processed names):")
            for rapper in not_found:
                print(f"- {rapper} -> {self.preprocess_name(rapper)}")
        
        return suggestions

    def export_analysis(self, output_path: str = 'rapper_analysis.json'):
        """
        Export complete analysis including aliases and similarity scores
        """
        analysis = {
            'aliases': self.aliases,
            'statistics': {
                'total_rappers': len(self.aliases),
                'total_aliases': sum(len(aliases) for aliases in self.aliases.values())
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)

# Example usage:
if __name__ == "__main__":
    # Initialize tools
    tools = RapperAnalysisTools('word2vec_model.model')
    
    """ # Process rappers from list
    suggestions = tools.batch_process_rappers(
        '../Supporting - List of Rappers/Spotify Playlist scrape/all_artists.txt',
        similarity_threshold=0.5,
        review_mode='interactive'  # Change to 'auto' for automatic processing
    ) """
    
    # Create visualization
    fig = tools.create_all_words_viz(min_freq=4, max_words=400000)
    fig.write_html("all_words_embeddings.html")
    
    # Export analysis
    tools.export_analysis()