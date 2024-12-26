import os
# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))


def normalize_text(text):
    """Normalize text for comparison by removing underscores and extra spaces."""
    return ' '.join(text.replace('_', ' ').split())

def create_corrected_mapping(original_sentences_file, timestamp_mapping_file, output_file):
    # Load original sentences (these match the database)
    print("Loading original sentences...")
    original_sentences = []
    with open(original_sentences_file, 'r', encoding='utf-8') as f:
        original_sentences = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(original_sentences)} original sentences")

    # Load timestamp mapping
    print("Loading timestamp mapping...")
    timestamp_map = {}
    with open(timestamp_mapping_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                sentence, timestamp = line.strip().split('\t')
                normalized = normalize_text(sentence)
                timestamp_map[normalized] = timestamp
    print(f"Loaded {len(timestamp_map)} timestamp mappings")

    # Create new mapping using original sentences
    print("Creating corrected mapping...")
    matched = 0
    unmatched = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in original_sentences:
            normalized = normalize_text(sentence)
            if normalized in timestamp_map:
                f.write(f"{sentence}\t{timestamp_map[normalized]}\n")
                matched += 1
            else:
                unmatched += 1
                if unmatched <= 5:  # Show first 5 unmatched sentences
                    print(f"Could not find timestamp for: {sentence}")

    print(f"\nResults:")
    print(f"Matched sentences: {matched}")
    print(f"Unmatched sentences: {unmatched}")
    print(f"Success rate: {matched/(matched+unmatched)*100:.2f}%")
    print(f"Saved corrected mapping to {output_file}")

if __name__ == "__main__":
    create_corrected_mapping(
        "2_2-sentences_with_ngrams.txt",
        "sentence_timestamps_mapping.txt",
        "corrected_timestamps_mapping.txt"
    )
