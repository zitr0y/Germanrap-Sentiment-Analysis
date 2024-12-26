import sqlite3
from typing import Tuple
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def clean_sentiment_database(db_path: str = 'rapper_sentiments.db') -> Tuple[int, int]:
    """
    Clean the sentiment database by:
    1. Deleting entries with 'ERROR' sentiment
    2. Converting 'NO_SENTIMENT' to '3' (neutral)
    
    Returns:
        Tuple of (number of deleted errors, number of converted no_sentiments)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # First count how many we'll modify
        cursor.execute("SELECT COUNT(*) FROM sentiment_analysis WHERE sentiment = 'ERROR'")
        error_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM sentiment_analysis WHERE sentiment = 'NO_SENTIMENT'")
        no_sentiment_count = cursor.fetchone()[0]
        
        # Delete ERROR entries
        cursor.execute("DELETE FROM sentiment_analysis WHERE sentiment = 'ERROR'")
        
        # Convert NO_SENTIMENT to 3
        cursor.execute("UPDATE sentiment_analysis SET sentiment = '3' WHERE sentiment = 'NO_SENTIMENT'")
        
        # Commit changes
        conn.commit()
        
        print(f"Successfully cleaned database:")
        print(f"- Deleted {error_count} entries with 'ERROR' sentiment")
        print(f"- Converted {no_sentiment_count} entries from 'NO_SENTIMENT' to '3'")
        
        return error_count, no_sentiment_count
        
    except Exception as e:
        print(f"An error occurred: {e}")
        conn.rollback()
        return 0, 0
        
    finally:
        conn.close()

if __name__ == "__main__":
    clean_sentiment_database()