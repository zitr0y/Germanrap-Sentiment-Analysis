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
    sentences = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading lines", unit="lines", ncols=100):
            # Split each line into words and append the resulting list
            sentences.append(line.strip().split())
    print("Sentences loaded!")
    return sentences

def train_word2vec(sentences):
    """Train Word2Vec model."""
    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences,
        vector_size=300,
        window=10,
        min_count=3,
        sg=1,
        hs=1,
        negative=15,
        epochs=20,
        alpha=0.025,
        min_alpha=0.0001,
        workers=mp.cpu_count()
    )
    print("Model training complete")
    return model



def test_model(model, test_words):
    """Test model with some example words."""
    print("\nTesting model with example words:")
    for word in test_words:
        try:
            word = word.lower()
            similar_words = model.wv.most_similar(word)
            print(f"\nMost similar to {word}:")
            for w, score in similar_words:
                print(f"{w}: {score:.4f}")
        except KeyError:
            print(f"'{word}' not in vocabulary")

def main():

    # 1. Start time
    start_time = time.time()

    # 2. Load sentences with bigrams
    sentences_with_ngrams = load_sentences("../Step 2.2 Create Bi-and Trigrams for Word2Vec/2_2-sentences_with_ngrams.txt")

    # 3. Train model
    model = train_word2vec(sentences_with_ngrams)
    
    # 4. Basic model information
    print("\nModel Information:")
    print(f"Vocabulary size: {len(model.wv.key_to_index)}")
    
    # 5. Test the model
    test_words = ['kollegah', 'haftbefehl', 'hiob', 'bushido', 'azudemsk', 'shneezin', 'prezident']  # Add relevant test words
    test_bigrams = ['sun_diego', 'kool_savas', 'mc_bomber','mc_fitti', 'farid_bang', 'juse_ju', 'yung_hurn' , 'og_keemo', 'og_pezo', 'mr._sample', 'mr._rap', 'funkvater_frank', 'private_paul', 'jan_delay', 'tj_beastboy', 'sugar_mmfk']  # Add relevant bigrams
    test_trigrams = ['eins_acht_sieben', 'audio88_und_yassin', 'Maedness_und_Doell', 'bass_sultan_hengst', 'k._i._z']
    test_model(model, test_words)
    test_model(model, test_bigrams)
    test_model(model, test_trigrams)
    
    # 6. Save the model
    model.save("word2vec_model.model")
    print("\nModel saved to word2vec_model.model")
    
    # Print total time taken
    elapsed_time = time.time() - start_time
    print(f"\nTotal time taken: {elapsed_time/60:.2f} minutes")

if __name__ == "__main__":
    main()