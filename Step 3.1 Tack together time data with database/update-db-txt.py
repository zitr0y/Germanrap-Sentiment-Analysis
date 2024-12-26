import sqlite3
from datetime import datetime
import sys
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))


def load_sentence_timestamps(filename):
    """Load the sentence to timestamp mapping from mapping file."""
    timestamp_map = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                sentence, timestamp = line.strip().split('\t')
                timestamp_map[sentence] = int(timestamp)
            except ValueError as e:
                print(f"Error parsing line in timestamp file: {line.strip()}")
                continue
    return timestamp_map

def update_database_timestamps(db_path, mapping_file):
    """Update the sentiment database with original timestamps."""
    print("Loading timestamp mapping...")
    timestamp_map = load_sentence_timestamps(mapping_file)
    print(f"Loaded {len(timestamp_map)} sentence-timestamp pairs")
    
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Add a new column for original_timestamp if it doesn't exist
        cursor.execute("""
        SELECT name FROM pragma_table_info('sentiment_analysis') 
        WHERE name = 'original_timestamp';
        """)
        if not cursor.fetchone():
            cursor.execute("""
            ALTER TABLE sentiment_analysis 
            ADD COLUMN original_timestamp INTEGER;
            """)
            print("Added original_timestamp column")
        
        # Get all rows without timestamps
        cursor.execute("""
        SELECT id, text FROM sentiment_analysis
        WHERE original_timestamp IS NULL;
        """)
        rows = cursor.fetchall()
        print(f"Found {len(rows)} rows without timestamps")
        
        # Prepare updates
        updates = []
        missing_timestamps = []
        for row_id, text in rows:
            if text in timestamp_map:
                updates.append((timestamp_map[text], row_id))
            else:
                missing_timestamps.append(row_id)
        
        if missing_timestamps:
            print(f"Warning: Could not find timestamps for {len(missing_timestamps)} rows")
            print("First few missing texts:")
            cursor.execute("""
            SELECT id, text FROM sentiment_analysis
            WHERE id IN ({}) LIMIT 5;
            """.format(','.join(map(str, missing_timestamps[:5]))))
            for row in cursor.fetchall():
                print(f"ID {row[0]}: {row[1]}")
        
        # Batch update the database
        if updates:
            cursor.executemany("""
            UPDATE sentiment_analysis 
            SET original_timestamp = ?
            WHERE id = ?;
            """, updates)
            print(f"Updated {len(updates)} rows with timestamps")
        
        # Commit changes
        conn.commit()
        print("Changes committed to database")
        
    except Exception as e:
        conn.rollback()
        print(f"Error updating database: {str(e)}", file=sys.stderr)
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    db_path = "rapper_sentiments.db"
    mapping_file = "corrected_timestamps_mapping.txt"
    update_database_timestamps(db_path, mapping_file)
