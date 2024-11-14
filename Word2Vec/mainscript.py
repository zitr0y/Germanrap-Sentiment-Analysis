import os
import subprocess
import logging
from dotenv import load_dotenv, set_key
from datetime import datetime
import sys
import jwt
import requests
import time

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# Setup logging with more detailed format
logging.basicConfig(
    filename='scraper.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def check_token_expiration(token):
    """Check if token needs renewal (returns True if token expires in less than 1 hour)"""
    try:
        # Decode token without verification (we just need to check expiration)
        decoded = jwt.decode(token, options={"verify_signature": False})
        expiration_time = decoded.get('expires')
        
        if not expiration_time:
            logging.error("Token does not contain expiration time")
            return True
            
        # Get current time
        current_time = int(time.time())
        
        # Return True if token expires in less than 1 hour
        return (expiration_time - current_time) < 3600
        
    except jwt.InvalidTokenError:
        logging.error("Invalid token format")
        return True

def refresh_token(current_token):
    """Refresh the Pushshift API token"""
    try:
        response = requests.post(
            'https://auth.pushshift.io/refresh',
            params={'access_token': current_token},  # Changed from json to params
            headers={'Content-Type': 'application/json'}
        )
        
        response.raise_for_status()
        new_token = response.json().get('access_token')
        
        if not new_token:
            raise ValueError("No access token in response")
            
        # Update token in .env file
        set_key('.env', 'PUSHSHIFT_ACCESS_TOKEN', new_token)
        logging.info("Successfully refreshed API token")
        
        return new_token
        
    except requests.RequestException as e:
        logging.error(f"Failed to refresh token: {str(e)}")
        if hasattr(e, 'response') and e.response is not None:
            logging.error(f"Response content: {e.response.text}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error while refreshing token: {str(e)}")
        raise

def validate_env_variables():
    """Validate required environment variables are present and properly formatted"""
    load_dotenv()
    
    required_vars = ['PUSHSHIFT_ACCESS_TOKEN', 'SUBREDDITS']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        return False
        
    # Validate date formats if provided
    for date_var in ['SINCE', 'UNTIL']:
        date_str = os.getenv(date_var)
        if date_str:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
            except ValueError:
                logging.error(f"Invalid date format for {date_var}: {date_str}. Expected format: YYYY-MM-DD")
                return False
    
    return True

def get_python_executable():
    """Get the correct Python executable path based on the environment"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(base_dir, "..", "..", ".venv", "Scripts", "python.exe"),  # Windows
        os.path.join(base_dir, "..", "..", ".venv", "bin", "python"),         # Unix
        os.path.join(base_dir, ".venv", "Scripts", "python.exe"),             # Windows (alternative)
        os.path.join(base_dir, ".venv", "bin", "python"),                     # Unix (alternative)
        sys.executable                                                         # Current Python
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            logging.info(f"Using Python executable: {path}")
            return path
            
    logging.warning("Using system Python as fallback")
    return "python"

def main():
    try:
        # Validate environment variables
        if not validate_env_variables():
            logging.error("Environment validation failed")
            sys.exit(1)
            
        # Get current access token
        access_token = os.getenv('PUSHSHIFT_ACCESS_TOKEN')
        
        # Check if token needs refresh
        if check_token_expiration(access_token):
            logging.info("Access token is expired or will expire soon. Attempting to refresh...")
            try:
                access_token = refresh_token(access_token)
            except Exception as e:
                logging.error("Failed to refresh token. Proceeding with current token.")
                # Continue with current token - it might still be valid
        
        # Get paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        python_exe = get_python_executable()
        scraper_path = os.path.join(base_dir, "1-pushshift-reddit-scraper.py")
        
        # Verify scraper script exists
        if not os.path.exists(scraper_path):
            logging.error(f"Scraper script not found at: {scraper_path}")
            sys.exit(1)
        
        # Prepare command
        command = [
            python_exe,
            scraper_path,
            "--access_token",
            access_token
        ]
        
        # Log command (without sensitive data)
        logging.info(f"Executing scraper script with Python: {python_exe}")
        logging.info(f"Scraper path: {scraper_path}")
        
        # Execute scraper
        try:
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=36000
            )
            
            # Log success
            logging.info("Scraper executed successfully")
            logging.debug(f"Scraper output: {result.stdout}")
            print("Scraper executed successfully. Check the log file for details.")
            
        except subprocess.CalledProcessError as e:
            logging.error(f"Scraper execution failed with return code {e.returncode}")
            logging.error(f"Error output: {e.stderr}")
            print(f"Error: Scraper execution failed. Check the log file for details.")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        print(f"An unexpected error occurred. Check the log file for details.")
        sys.exit(1)

if __name__ == "__main__":
    logging.info('Starting mainscript.py')
    main()