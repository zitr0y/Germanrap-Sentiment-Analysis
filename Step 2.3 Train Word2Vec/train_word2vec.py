import logging
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
import time
from tqdm import tqdm
import os
import multiprocessing as mp

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def load_sentences(file_path):
    # Save sentences with bigrams to a file
    sentences = []
    with open('2_3-sentences_with_ngrams.txt', 'n', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading lines", unit="lines", ncols=100):
            sentences.append(line.strip())
    print("Sentences loaded!")

def train_word2vec(sentences):
    """Train Word2Vec model."""
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences,
        vector_size=150,
        window=8,
        min_count=10,
        sg=1,
        hs=1,
        negative=10,
        workers=mp.cpu_count()
    )
    print("Model training complete")
    return model



def test_model(model, test_words):
    """Test model with some example words."""
    print("\nTesting model with example words:")
    for word in test_words.lower():
        try:
            similar_words = model.wv.most_similar(word)
            print(f"\nMost similar to {word}:")
            for w, score in similar_words:
                print(f"{w}: {score:.4f}")
        except KeyError:
            print(f"'{word}' not in vocabulary")

def main():

    sentences_with_ngrams = load_sentences("lyrics.txt")

    # 3. Train model
    model = train_word2vec(sentences_with_ngrams)
    
    # 4. Basic model information
    print("\nModel Information:")
    print(f"Vocabulary size: {len(model.wv.key_to_index)}")
    
    # 5. Test the model
    test_words = ['kollegah', 'haftbefehl', 'rapper']  # Add relevant test words
    test_bigrams = ['sun_diego', 'kool_savas', 'mc_bomber','mc_fitti', 'farid_bang', 'juse_ju', 'yung_hurn' , 'og_keemo', 'og_pezo', 'mr._sample', 'mr._rap', 'funkvater_frank', 'private_paul', 'jan_delay']  # Add relevant bigrams
    test_model(model, test_words)
    
    # 6. Save the model
    model.save("word2vec_model.model")
    print("\nModel saved to word2vec_model.model")
    
    # Print total time taken
    elapsed_time = time.time() - start_time
    print(f"\nTotal time taken: {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    main()