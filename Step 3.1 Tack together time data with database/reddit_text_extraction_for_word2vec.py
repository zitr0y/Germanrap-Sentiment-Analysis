import json
from typing import List, Dict
import re
import os
from tqdm import tqdm
import time
from typing import List, Tuple, Dict
from tqdm import tqdm

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def extract_text_with_timestamps(json_data: Dict) -> List[Tuple[str, int]]:
    """
    Extract text and timestamps from Reddit JSON data.
    Returns a list of tuples (sentence, timestamp).
    Raises ValueError if a required timestamp is missing.
    """
    sentences_with_timestamps = {}  # Use dict to ensure uniqueness
    
    # Extract post title and body with timestamp
    post_info = json_data["post_info"]
    post_timestamp = post_info.get("created_utc")
    if post_timestamp is None:
        raise ValueError(f"Missing timestamp for post {post_info.get('id')}")
    
    if post_info["title"]:
        cleaned_title = clean_text(post_info["title"])
        if cleaned_title:
            sentences_with_timestamps[cleaned_title] = post_timestamp
    
    if post_info["body"]:
        cleaned_body = clean_text(post_info["body"])
        if cleaned_body:
            body_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_body)
            for sentence in body_sentences:
                if len(sentence.split()) >= 3:
                    sentences_with_timestamps[sentence] = post_timestamp
    
    def process_comments(comments):
        for comment in comments:
            comment_timestamp = comment.get("created_utc")
            if comment_timestamp is None:
                raise ValueError(f"Missing timestamp for comment {comment.get('id')}")
                
            if comment["body"]:
                cleaned_comment = clean_text(comment["body"])
                if cleaned_comment:
                    comment_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_comment)
                    for sentence in comment_sentences:
                        if len(sentence.split()) >= 3 and sentence not in sentences_with_timestamps:
                            sentences_with_timestamps[sentence] = comment_timestamp
            
            if "replies" in comment and comment["replies"]:
                process_comments(comment["replies"])
    
    process_comments(json_data["comments"])
    
    return list(sentences_with_timestamps.items())

def save_sentences_with_timestamps(sentences_with_timestamps: List[Tuple[str, int]], output_file: str):
    """Save sentences and timestamps to a TSV file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, timestamp in sentences_with_timestamps:
            f.write(f"{sentence}\t{timestamp}\n")


def should_merge_lines(prev: str, curr: str) -> bool:
    """
    Determine if two lines should be merged based on content patterns.
    """
    if not prev or not curr:
        return False
        
    # Patterns indicating continuation
    continuation_patterns = [
        r'(?i)(?:feat|prod|ft\.?|remix|edit)\.?\s*$',  # Music-related endings
        r'\($',           # Open parenthesis at end
        r'–\s*$',        # Em dash at end
        r'-\s*$',        # Hyphen at end
        r'&\s*$',        # Ampersand at end
        r'\/\s*$',       # Forward slash at end
        r'^\s*[&\/]',    # Starts with ampersand or slash
        r'^\s*\)',       # Starts with closing parenthesis
    ]
    
    # Check if previous line ends with a continuation pattern
    if any(re.search(pattern, prev) for pattern in continuation_patterns):
        return True
        
    # Check for matching parentheses
    open_parens = prev.count('(') - prev.count(')')
    if open_parens > 0:
        return True
        
    return False

def clean_text(text: str) -> str:
    """
    Clean Reddit-specific formatting while preserving meaningful content.
    Returns cleaned text with formatting removed or None if the text should be filtered out.
    """
    # Early rejection for removed/deleted content
    removed_patterns = [
        r'^\s*\[ ?Removed by Reddit ?\]\s*$',
        r'^\s*\[deleted( by user)?\]\s*$',
        r'^\s*\[removed\]\s*$'
    ]
    if any(re.match(pattern, text, flags=re.IGNORECASE) for pattern in removed_patterns):
        return None

    # If text is too short or None, filter it out early
    if not text or len(text.strip()) < 10:
        return None
    
    text = text.lower()  # Add this line here
    text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue').replace('ß', 'ss')


    # First clean HTML entities and common Reddit formatting
    text = re.sub(r'&(?:[a-z\d]+|#\d+|#x[a-f\d]+);', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[\/?(?:s|spoiler)\]', '', text, flags=re.IGNORECASE)

    # Clean various URL patterns and links
    url_patterns = [
        r'https?://[^\s]+',
        r'(?:new|old|np|sh)\.reddit\.com/[^\s]+',
        r'reddit\.com/[^\s]+',
        r'you\.tube/[^\s]+',
        r'youtu\.be/[^\s]+',
        r'(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*',
        r'\[.*?\]\((?:https?://)?[^\s\)]+\)',
        r'(?:^|\s)u/[A-Za-z0-9_-]+',
        r'(?:^|\s)r/[A-Za-z0-9_-]+',
        r'/[rup]/[A-Za-z0-9_-]+',
        r'width=\d+&(?:format|height|auto|s)=[^\s]+',
        r'(?:youtube\.com|youtu\.be)/\S+'
    ]
    
    combined_pattern = '|'.join(f'({pattern})' for pattern in url_patterns)
    text = re.sub(combined_pattern, ' ', text)

    # Clean quote blocks (preserve content without '>')
    text = re.sub(r'^\s*>\s*(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # Remove code blocks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)

    # remove brackets
    text = re.sub(r'[()]', '', text)
    
    # Remove markdown formatting while keeping content
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'~~(.*?)~~', r'\1', text)
    
    # Handle lists without adding periods
    text = re.sub(r'^\s*[\*\-]\s*(.+)$', r'\1', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s*(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # Remove table formatting
    text = re.sub(r'\|.*\|', ' ', text)
    text = re.sub(r'[\-\|]+', ' ', text)

    # Split into lines first
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Merge lines that should be together
    merged_lines = []
    current_line = ''
    
    for line in lines:
        if not current_line:
            current_line = line
        elif should_merge_lines(current_line, line):
            current_line = f"{current_line} {line}"
        else:
            merged_lines.append(current_line)
            current_line = line
    
    if current_line:
        merged_lines.append(current_line)
    
    # Join lines and clean up spacing
    text = ' '.join(merged_lines)
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    # Filter out bot-like content
    if any(marker in text.lower() for marker in [
        'remindme!', 
        'click this link',
        'parent commenter can',
        'I am a bot',
        'action was performed',
        'Please if you have any questions or concerns',
        'aus dem folgenden grund entfernt:',
        'das schreibt der:die künstler:in zum eigenen werk',
        "[release friday]",
        '[diskussion] besprechung der releases vom',
        'welcher track lief bei euch übers wochenende im loop?',
        'gomunkul',
        '* * * *',
        'response gebaut: trau dich auch anderen usern ein feedback zu geben',
        '^^beep ^^boop',
        'der sonntagsthread',
        'this post was mass deleted and anonymized',
        'elo elo elo elo elo elo',
        'bitches bitches bitches bitches bitches bitches',
        'griff griff griff griff griff griff',
        '^^zum ^^letzten ^^thread ^^zu ^^dem ^^thema ^^geht ^^es ^^hier',
        'die beginner haben einmal die line wer hip hop macht aber nur hip hop hört betreibt inzest gedroppt'
        '!remindme',
        "I'm high on believing That you're in love with me",
        'Let me know the names of any other subreddits you want me to post in',
        'Meiste Upvote Antwort kommt in eine Playlist, sodass man am Ende das Alphabet vollständig hat',
        'a weekly recap with the top posts and their',
        '^^^^The ^^^^parent ^^^^commenter ^^^^can ^^^^reply ^^^^with',
        '(17. oktober 2021*) (10. oktober 2021*) (03. oktober 2021*) (02.',
        "i'm hooked on a feeling seems i got hooked on a feeling",
        '1 invite to get free stuff. :)',
        'der:die kuenstler:in zum eigenen werk',
        'oktober 2021* oktober 2021* oktober 2021*',
        'welches release hat euch ueberrarscht sowohl positiv als auch negativ?',
        'januar 2022* januar 2022*',
        'a message or send me a chat with the text:',
        'einen track vorschlagen. meiste upvote antwort kommt in eine playlist',
        'savevideobot',
        




    ]):
        return None
    
    # Define patterns for unwanted endings - order matters!
    end_patterns = [
    r'[.,!?:;&_/\\…]+$',    # Handle single or multiple punctuation marks at the end
    r'[\'"]+$',              # Quotes
    r'[)\]}]+$',             # Closing brackets/parens
    r'\s+$',                 # Trailing whitespace
    ]   

    cleaned_words = []
    for word in text.split():
        # Apply each pattern to clean word endings
        cleaned_word = word
        while True:
            old_word = cleaned_word
            for pattern in end_patterns:
                cleaned_word = re.sub(pattern, '', cleaned_word)
            # If no changes were made in this iteration, break
            if old_word == cleaned_word:
                break
        cleaned_words.append(cleaned_word)
    text = ' '.join(cleaned_words)

    # Count the ratio of alphanumeric characters
    alpha_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text) if text else 0
    if alpha_ratio < 0.5:
        return None

    return text if len(text.split()) >= 3 else None

def extract_text_for_word2vec(json_data: Dict) -> List[str]:
    """
    Extract and clean text from Reddit JSON data.
    Returns a list of sentences suitable for word2vec training.
    """
    sentences = []
    
    # Extract post title and body
    post_info = json_data["post_info"]
    if post_info["title"]:
        cleaned_title = clean_text(post_info["title"])
        if cleaned_title:
            sentences.append(cleaned_title)
    
    if post_info["body"]:
        cleaned_body = clean_text(post_info["body"])
        if cleaned_body:
            # Split on clear sentence boundaries only
            body_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_body)
            sentences.extend(s for s in body_sentences if len(s.split()) >= 3)
    
    def process_comments(comments):
        for comment in comments:
            if comment["body"]:
                cleaned_comment = clean_text(comment["body"])
                if cleaned_comment:
                    comment_sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', cleaned_comment)
                    sentences.extend(s for s in comment_sentences if len(s.split()) >= 3)
            
            if "replies" in comment and comment["replies"]:
                process_comments(comment["replies"])
    
    process_comments(json_data["comments"])
    
    return sentences

# In your main process:
def process_reddit_files(directory_path: str, output_file: str):
    all_sentences_with_timestamps = {}
    
    for filename in tqdm(os.listdir(directory_path)):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    sentences = extract_text_with_timestamps(json_data)
                    # Only keep first occurrence of each sentence
                    for sentence, timestamp in sentences:
                        if sentence not in all_sentences_with_timestamps:
                            all_sentences_with_timestamps[sentence] = timestamp
            except ValueError as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
            except Exception as e:
                print(f"Unexpected error processing {filename}: {str(e)}")
                continue

    # Convert dict to list of tuples and save
    sentences_list = list(all_sentences_with_timestamps.items())
    save_sentences_with_timestamps(sentences_list, output_file)
    print(f"Processed {len(sentences_list)} unique sentences")



if __name__ == "__main__":
    # Use raw string to handle backslashes in path
    directory_path = r"..\Step 1 Reddit Scraper\1-posts"
    
    # Process all files and get sentences
    sentences = process_reddit_files(directory_path, '2_1-processed_sentences_with_time.txt')