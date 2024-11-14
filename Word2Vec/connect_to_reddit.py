import praw
import os
import logging
import sys
from dotenv import load_dotenv

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def connecttoreddit():
    
    try:
        load_dotenv()
        client_id = os.getenv('CLIENT_ID')
        client_secret = os.getenv('CLIENT_SECRET')
        refresh_token = os.getenv('REFRESH_TOKEN')
        user_agent = os.getenv('USER_AGENT')
        logging.info('Loaded .env file successfully.')
    except:
        logging.error('Error: Could not load .env file. Please make sure that the file exists and is in the same directory as the script.')
        sys.exit()



    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        refresh_token=refresh_token,
        user_agent=user_agent,
    )
    logging.info('Connected to Reddit!')
    return reddit