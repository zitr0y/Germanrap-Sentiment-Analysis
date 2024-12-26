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

# In your main process:
def process_reddit_files(directory_path: str, output_file: str):
    all_sentences_with_timestamps = {}
    
    for filename in os.listdir(directory_path):
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
