import os
import logging
import datetime
import json
import argparse
import sys
import requests
import time
from datetime import datetime
from dotenv import load_dotenv
from praw.models import MoreComments
import connect_to_reddit

os.chdir(os.path.dirname(os.path.realpath(__file__)))
    
# Define constants for file paths
OUTPUT_DIR = '1-posts'
LOG_FILE = 'scraper.log'
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command line arguments and merge with .env defaults"""
    parser = argparse.ArgumentParser(description="Pushshift Reddit Scraper")
    parser.add_argument("--access_token", type=str, required=True, help="Access token for Pushshift API")
    
    load_dotenv()
    args = parser.parse_args()
    
    config = {
        "subreddits": os.getenv('SUBREDDITS', 'germanrap').split(','),
        "limit": int(os.getenv('LIMIT', '1000000')),  # Default to 1M posts
        "since": os.getenv('SINCE', '2010-01-01'),    # Default to early Reddit history
        "until": os.getenv('UNTIL', datetime.now().strftime('%Y-%m-%d')),
        "access_token": args.access_token,
        "batch_size": 100  # Maximum allowed by Pushshift
    }
    
    logging.info(f"Configuration loaded (excluding access_token): {dict(config, access_token='[HIDDEN]')}")
    return config

def get_pushshift_data(subreddit, since, until, limit, access_token):
    """
    Fetches data from Pushshift API with optimized batch processing.
    """
    total_processed = 0
    batch_size = 100  # Pushshift maximum
    base_url = "https://api.pushshift.io"
    endpoint = "/reddit/search/submission"
    current_until = until

    session = requests.Session()
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    while total_processed < limit:
        params = {
            "subreddit": subreddit,
            "limit": batch_size,
            "since": since,
            "until": current_until,
            "sort": "created_utc",
            "order": "desc",
            "track_total_hits": "true"
        }
        
        try:
            response = session.get(f"{base_url}{endpoint}", headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            if isinstance(data, dict):
                data = data.get('data', [])
            
            if not data:
                logging.info("No more posts available")
                break

            logging.info(f"Received {len(data)} posts")
            yield data
            total_processed += len(data)
            logging.info(f"Total processed posts: {total_processed} of {limit}")

            # Update timestamp for next request
            if data:
                oldest_timestamp = min(post['created_utc'] for post in data)
                current_until = oldest_timestamp - 1
                logging.info(f"Updated 'until' timestamp to {current_until}")
            
            # If we got fewer posts than requested, we've reached the end
            if len(data) < batch_size:
                logging.info(f"Received data length {len(data)} is smaller than batch size {batch_size}, might have reached end of available posts")
                # break
            logging.info("Sleeping for 10 seconds to be nice to Pushshift API")
            time.sleep(10)  # Rate limiting

        except requests.RequestException as e:
            logging.error(f"Error fetching data from Pushshift: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                logging.error(f"Response content: {e.response.text}")
            logging.info("Rate Limit suspected! Retrying in 120 seconds")
            time.sleep(120)  # Longer wait on error
            continue

def process_comments(comment, parent_id=None):
    """Process comments recursively with optimized error handling"""
    if isinstance(comment, MoreComments):
        try:
            return [process_comments(c) for c in comment.comments()]
        except Exception as e:
            logging.error(f"Error expanding MoreComments: {str(e)}")
            return []

    try:
        comment_data = {
            "id": comment.id,
            "body": comment.body,
            "author": str(comment.author.name) if comment.author else '[deleted]',
            "created_utc": int(comment.created_utc),
            "score": comment.score,
            "parent_id": parent_id,
            "replies": []
        }

        if hasattr(comment, 'replies'):
            comment_data["replies"] = [
                process_comments(reply, comment.id) 
                for reply in comment.replies 
                if not isinstance(reply, MoreComments)
            ]

        return comment_data
    except Exception as e:
        logging.error(f"Error processing comment {getattr(comment, 'id', 'unknown')}: {str(e)}")
        return None

def save_post(post_data, output_file):
    """Save post data with error handling"""
    try:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(post_data, json_file, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        logging.error(f"Error saving post to {output_file}: {str(e)}")
        return False

def process_posts(posts, reddit):
    """Process posts with optimized error handling and progress tracking"""
    for post in posts:
        try:
            post_id = post.get("id")
            if not post_id:
                logging.error(f"Missing post ID in data: {post}")
                continue
            output_file = os.path.join(OUTPUT_DIR, f'{post_id}.json')
            
            if os.path.exists(output_file):
                logging.debug(f'Skipping existing post {post_id}')
                continue
            
            logging.debug(f'PRAW now processing post {post_id}')
            praw_post = reddit.submission(id=post_id)
            post_data = {
                "post_info": {
                    "id": praw_post.id,
                    "title": praw_post.title,
                    "body": praw_post.selftext,
                    "author": praw_post.author.name if praw_post.author else '[deleted]',
                    "created_utc": int(praw_post.created_utc),
                    "score": praw_post.score,
                    "upvote_ratio": praw_post.upvote_ratio,
                    "num_comments": praw_post.num_comments,
                    "permalink": praw_post.permalink,
                    "url": praw_post.url,
                    # Added additional fields
                    "is_self": praw_post.is_self,
                    "is_video": praw_post.is_video,
                    "over_18": praw_post.over_18,
                    "spoiler": praw_post.spoiler,
                    "stickied": praw_post.stickied,
                    # Optional fields that might be useful
                    "domain": praw_post.domain,
                    "archived": praw_post.archived,
                    "locked": praw_post.locked,
                    "removed": praw_post.removed if hasattr(praw_post, 'removed') else None,
                    "link_flair_text": praw_post.link_flair_text
                },
                "comments": []
            }
            logging.debug(f'Post data aquired')
            praw_post.comment_limit = 10000  # Increase from 2048
            praw_post.comments.replace_more(limit=None)
            logging.info(f"Processing {praw_post.num_comments} comments for post {post_id}")
            processed_comments = []
            
            for comment in praw_post.comments:
                processed_comment = process_comments(comment)
                if processed_comment:
                    processed_comments.append(processed_comment)
            
            post_data['comments'] = processed_comments
            
            if save_post(post_data, output_file):
                logging.info(f"Successfully saved post {post_id}")
            
        except Exception as e:
            logging.error(f"Error processing post {post.get('id', 'unknown')}: {str(e)}")
            continue

def main():
    try:
        config = parse_arguments()
        reddit = connect_to_reddit.connecttoreddit()
        
        for subreddit in config['subreddits']:
            logging.info(f"Starting archive of r/{subreddit}")
            
            until = int(datetime.strptime(config['until'], "%Y-%m-%d").timestamp())
            since = int(datetime.strptime(config['since'], "%Y-%m-%d").timestamp())
            
            for batch in get_pushshift_data(
                subreddit=subreddit,
                since=since,
                until=until,
                limit=config['limit'],
                access_token=config['access_token']
            ):
                process_posts(batch, reddit)
                
    except Exception as e:
        logging.error(f"Critical error in main function: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    logging.info('Starting Pushshift Reddit Archiver')
    main()
