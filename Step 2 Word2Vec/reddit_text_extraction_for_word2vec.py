import json
from typing import List, Dict
import re
import os

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def clean_text(text: str) -> str:
    """
    Clean Reddit-specific formatting while preserving meaningful content.
    Returns cleaned text with formatting removed but content preserved.
    """
    # Store original text for comparison
    original = text
    
    # 1. Handle quote blocks (preserve content without '>')
    text = re.sub(r'^\s*>\s*(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # 2. Handle spoiler tags
    text = re.sub(r'\&gt;\!(.*?)\!\&lt;', r'\1', text)
    
    # 3. Remove code blocks - they're rarely relevant for natural language
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    
    # 4. Remove markdown formatting while keeping content
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
    text = re.sub(r'~~(.*?)~~', r'\1', text)      # Strikethrough
    
    # 5. Handle lists - preserve content but add periods for sentence splitting
    text = re.sub(r'^\s*[\*\-]\s*(.+)$', r'\1.', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*(.+)$', r'\1.', text, flags=re.MULTILINE)
    
    # 6. Remove table formatting
    text = re.sub(r'\|.*\|', ' ', text)
    text = re.sub(r'[\-\|]+', ' ', text)
    
    # 7. Final cleanup
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    return text

def extract_text_for_word2vec(json_data: Dict) -> List[str]:
    """
    Extract and clean text from Reddit JSON data.
    Returns a list of sentences suitable for word2vec training.
    """
    sentences = []
    
    # Extract post title and body
    post_info = json_data["post_info"]
    if post_info["title"]:
        sentences.append(clean_text(post_info["title"]))
    if post_info["body"]:
        # Split body into sentences (roughly)
        body_sentences = re.split(r'[.!?]+', post_info["body"])
        sentences.extend([clean_text(s) for s in body_sentences if clean_text(s)])
    
    # Extract comments and their replies
    def process_comments(comments):
        for comment in comments:
            if comment["body"]:
                # Split comment into sentences
                comment_sentences = re.split(r'[.!?]+', comment["body"])
                sentences.extend([clean_text(s) for s in comment_sentences if clean_text(s)])
            
            # Process replies recursively
            if "replies" in comment and comment["replies"]:
                process_comments(comment["replies"])
    
    process_comments(json_data["comments"])
    
    # Remove empty sentences and very short ones (less than 3 words)
    sentences = [s for s in sentences if s and len(s.split()) >= 3]
    
    return sentences

def process_reddit_files(directory_path: str) -> List[str]:
    """
    Process multiple Reddit JSON files from a directory and return sentences for word2vec.
    """
    all_sentences = []
    processed_files = 0
    error_files = 0
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
    total_files = len(json_files)
    
    print(f"Found {total_files} JSON files to process")
    
    for filename in json_files:
        file_path = os.path.join(directory_path, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                sentences = extract_text_for_word2vec(json_data)
                all_sentences.extend(sentences)
                processed_files += 1
                
                # Print progress every 100 files
                if processed_files % 100 == 0:
                    print(f"Processed {processed_files}/{total_files} files...")
                    
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            error_files += 1
    
    print(f"\nProcessing complete:")
    print(f"Successfully processed: {processed_files} files")
    print(f"Errors encountered: {error_files} files")
    print(f"Total sentences extracted: {len(all_sentences)}")
    
    return all_sentences

if __name__ == "__main__":
    # Use raw string to handle backslashes in path
    directory_path = r"..\Step 1 Reddit Scraper\1-posts"
    
    # Process all files and get sentences
    sentences = process_reddit_files(directory_path)
    
    # Save sentences to a file
    output_file = "processed_sentences.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write(sentence + '\n')
    
    print(f"\nSaved processed sentences to {output_file}")