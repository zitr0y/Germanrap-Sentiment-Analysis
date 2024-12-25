import os
import json
import ollama
import re
from typing import Dict, List, Set, Tuple, Optional
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import time
from collections import Counter
import sqlite3
from contextlib import contextmanager
import signal
import sys
import threading
import psutil  # Add this import

def kill_proc_tree(pid, including_parent=True):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    psutil.wait_procs(children, timeout=5)
    if including_parent:
        try:
            parent.kill()
        except psutil.NoSuchProcess:
            pass

os.chdir(os.path.dirname(os.path.realpath(__file__)))

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def _init_db(self):
        """Initialize the database schema."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    rapper_name TEXT NOT NULL,
                    found_alias TEXT NOT NULL,
                    sentiment VARCHAR(20),
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    processing_time FLOAT,
                    UNIQUE(text, rapper_name)
                )
            """)
            # Create indexes for faster querying
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rapper ON sentiment_analysis(rapper_name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment ON sentiment_analysis(sentiment)")
            conn.commit()

class RapperMentionAnalyzer:
    def __init__(self, db_path: str = 'rapper_sentiments.db'):
        self.db = DatabaseManager(db_path)
        
    def _get_existing_result(self, text: str, rapper_name: str) -> Optional[tuple]:
        """Get existing sentiment result from database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sentiment, processing_time FROM sentiment_analysis WHERE text = ? AND rapper_name = ?",
                (text, rapper_name)
            )
            result = cursor.fetchone()
            return result if result else None
    
    def _save_result(self, text: str, rapper_name: str, found_alias: str, sentiment: str, processing_time: float):
        """Save sentiment result to database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sentiment_analysis 
                (text, rapper_name, found_alias, sentiment, processing_time)
                VALUES (?, ?, ?, ?, ?)
            """, (text, rapper_name, found_alias, str(sentiment), processing_time))
            conn.commit()

import signal

class SentimentAnalyzer(RapperMentionAnalyzer):
    def __init__(self, rapper_aliases_path: str, db_path: str = 'rapper_sentiments.db'):
        super().__init__(db_path)
        self.name_mapping = self._load_rapper_aliases(rapper_aliases_path)
        self.reverse_mapping = self._create_reverse_mapping()
        self.model = "qwen2.5:3b"
        self.temperature = 0.2
        self.progress_file = 'sentiment_analysis_progress.json'
        self._shutdown = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\nReceived shutdown signal. Stopping all processing...")
        self._shutdown = True
        sys.exit(1)  # Force exit
        
    def _save_progress(self, processed_count: int, total_work_items: list):
        """Save progress to a JSON file."""
        remaining_items = total_work_items[processed_count:]
        progress = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'processed_count': processed_count,
            'remaining_items': remaining_items
        }
        
        # Save with timestamp in filename for backup
        if processed_count % 100000 == 0 or processed_count == len(total_work_items):
            backup_file = f'progress_backup_{time.strftime("%Y%m%d_%H%M%S")}_{processed_count}.json'
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        
        # Save current progress
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
            
    def _load_progress(self) -> Tuple[int, List]:
        """Load progress from JSON file if it exists."""
        try:
            with open(self.progress_file, 'r', encoding='utf-8') as f:
                progress = json.load(f)
                print(f"\nResuming from previous run:")
                print(f"Timestamp: {progress['timestamp']}")
                print(f"Processed: {progress['processed_count']} mentions")
                print(f"Remaining: {len(progress['remaining_items'])} mentions")
                return progress['processed_count'], progress['remaining_items']
        except FileNotFoundError:
            return 0, None
        
    def _load_rapper_aliases(self, json_path: str) -> Dict[str, str]:
        """Load rapper names and their aliases."""
        with open(json_path, 'r', encoding='utf-8') as f:
            rapper_data = json.load(f)
        
        name_mapping = {}
        for canonical_name, aliases in rapper_data.items():
            name_mapping[canonical_name.lower()] = canonical_name
            for alias in aliases:
                name_mapping[alias.lower()] = canonical_name
                
        return name_mapping
    
    def _create_reverse_mapping(self) -> Dict[str, List[str]]:
        """Create mapping from canonical names to all their aliases."""
        reverse_mapping = {}
        for variant, canonical in self.name_mapping.items():
            if canonical not in reverse_mapping:
                reverse_mapping[canonical] = []
            if variant.replace('_', ' ') != canonical.lower().replace('_', ' '):
                reverse_mapping[canonical].append(variant)
        return reverse_mapping

    def find_rappers_in_text(self, text: str) -> List[Tuple[str, str]]:
        """Find all rapper mentions in text and return tuple of (found_alias, canonical_name)."""
        text = text.lower().replace('_', ' ')
        found_rappers = set()
        
        # Pre-compile all patterns at initialization
        if not hasattr(self, '_rapper_patterns'):
            self._rapper_patterns = {
                variant.replace('_', ' '): (
                    re.compile(r'\b' + re.escape(variant.replace('_', ' ')) + r'\b'),
                    self.name_mapping[variant]
                )
                for variant in self.name_mapping.keys()
            }
        
        for variant, (pattern, canonical) in self._rapper_patterns.items():
            if pattern.search(text):
                found_rappers.add((variant, canonical))
                
        print(f"Found {len(found_rappers)} rappers in text: {text[:100]}...")
        
        return list(found_rappers)

    def get_sentiment_prompt(self, text: str, found_alias: str, canonical_name: str) -> str:
        """Create sentiment analysis prompt using the best-performing context-focused format."""
        instructions = """
        Regeln:
        - Antworte NUR mit 1, 2, 3, 4 oder 5
        - Bewerte die Stärke der positiven/negativen Aussage:
          * 5 = sehr positiv/begeistert
          * 4 = positiv/gut
          * 3 = neutral/unklar
          * 2 = negativ/kritisch
          * 1 = sehr negativ/ablehnend
        - Beachte Rap-Kontext: "krass", "brutal" = meist positiv gemeint
        """

        examples = """
        Beispiele mit Kontext:

        Stark positive Bewertung (5):
        "<<X>> ist der beste Rapper" -> 5 (klares Lob)
        "<<X>> ist krass/brutal" -> 5 (Rap-Slang: positiv)
        "<<X>> absoluter ehrenmove" -> 5 (höchste Anerkennung)

        Positive Bewertung (4):
        "<<X>> feier ich" -> 4 (klare Zustimmung)
        "<<X>> macht gute Musik" -> 4 (positive Aussage)
        "<<X>> hat skills" -> 4 (Anerkennung)

        Neutral/Unklar (3):
        "<<X>> hat neues Album" -> 3 (reine Info)
        "Song mit <<X>>" -> 3 (nur Erwähnung)
        "<<X>> und Y" -> 3 (Aufzählung)

        Negative Bewertung (2):
        "<<X>> ist nicht so meins" -> 2 (milde Kritik)
        "früher war <<X>> besser" -> 2 (Qualitätsverlust)

        Stark negative Bewertung (1):
        "<<X>> ist müll" -> 1 (harte Ablehnung)
        "<<X>> sollte aufhören" -> 1 (starke Kritik)
        """

        marked_text = text.replace(found_alias.replace('_', ' '), f"<<{found_alias.replace('_', ' ')}>>" , 1)
            
        return f"""Bewerte das Sentiment zu {found_alias} im Text: {marked_text}

                {instructions}

                {examples}

                Antworte NUR mit einer Zahl von 1-5."""
    
    def _parse_response(self, response: str) -> str:
        """Parse and validate model response with improved handling."""
        response = response.upper().strip()
        
        # Handle N/A variations
        if any(na_variant in response for na_variant in ['N/A', 'NA', 'NONE', 'NULL', 'NOT APPLICABLE']):
            return "NO_SENTIMENT"
            
        # Try to extract just the number if there's extra text
        number_match = re.search(r'[1-5]', response)
        if number_match:
            return number_match.group()
            
        # Try direct conversion
        try:
            sentiment = int(float(response))
            if 1 <= sentiment <= 5:
                return str(sentiment)
        except (ValueError, TypeError):
            pass
            
        return None

    def get_sentiment(self, text: str, found_alias: str, canonical_name: str) -> Tuple[Optional[str], float]:
        """Get sentiment rating from LLM with improved error handling and retry logic."""
        # Check if we already have this result
        existing_result = self._get_existing_result(text, canonical_name)
        if existing_result is not None:
            return existing_result[0], existing_result[1]
        
        prompt = self.get_sentiment_prompt(text, found_alias, canonical_name)
        start_time = time.time()
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    'temperature': self.temperature,
                }
            )
            raw_result = response['response'].strip()
            result = self._parse_response(raw_result)
            
            if not result:
                result = "ERROR"
                print(f"Invalid response format: {raw_result}")
                
        except Exception as e:
            print(f"Error getting sentiment: {e}")
            result = "ERROR"
        
        processing_time = time.time() - start_time
        
        # Save result immediately
        self._save_result(text, canonical_name, found_alias, result, processing_time)
        
        return result, processing_time

    def notify_crash(self, error_msg: str):
        """Send crash notification via Discord webhook."""
        if hasattr(self, 'discord_webhook_url') and self.discord_webhook_url:
            try:
                import requests
                payload = {
                    "content": f"❌ Sentiment Analysis Crash:\n```{error_msg}```"
                }
                requests.post(self.discord_webhook_url, json=payload)
            except Exception as e:
                print(f"Failed to send notification: {e}")

    def analyze_file(self, text_path: str, discord_webhook_url: str = None, n_workers: int = 2) -> None:
        """Analyze file with controlled parallelism and progress tracking."""
        self.discord_webhook_url = discord_webhook_url
        self._shutdown = False  # Flag to control shutdown
        
        print("Starting analysis...")
        
        # Try to load previous progress
        processed_count, remaining_items = self._load_progress()
        
        if remaining_items is None:
            # First run - collect all work items
            work_items = []
            total_lines = sum(1 for _ in open(text_path, 'r', encoding='utf-8'))
            
            print(f"Reading {total_lines} lines from file...")
            
            with open(text_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(tqdm(f, total=total_lines, desc="Finding rapper mentions")):
                    if not line.strip():
                        continue
                        
                    rapper_mentions = self.find_rappers_in_text(line.strip())
                    if rapper_mentions:
                        for found_alias, canonical_name in rapper_mentions:
                            # Check if already processed
                            if not self._get_existing_result(line.strip(), canonical_name):
                                work_items.append((line.strip(), found_alias, canonical_name))
                    
                    if line_num % 40000 == 0:
                        print(f"Found {len(work_items)} unprocessed mentions so far...")
        else:
            # Resume from previous progress
            work_items = remaining_items
        
        if not work_items:
            print("No new mentions to process!")
            return
            
        print(f"\nProcessing {len(work_items)} mentions starting from position {processed_count}")
        
        def process_mention(args):
            if self._shutdown:  # Check shutdown flag
                return None
            line, found_alias, canonical_name = args
            try:
                return self.get_sentiment(line, found_alias, canonical_name)
            except Exception as e:
                error_msg = f"Error processing line: {line}\nRapper: {canonical_name}\nError: {str(e)}"
                print(f"\n{error_msg}")
                self.notify_crash(error_msg)
                return None
        
        # Process mentions with threading
        total_time = 0
        current_processed = 0
        executor = None
        
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            print(f"\nProcessing mentions with {n_workers} workers...")
            
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                # Submit initial batch of tasks
                futures = []
                for item in work_items:
                    if self._shutdown:
                        break
                    futures.append(executor.submit(process_mention, item))
                
                # Process results as they complete
                for future in tqdm(as_completed(futures), total=len(futures), desc="Processing mentions"):
                    if self._shutdown:
                        break
                    
                    result = future.result()
                    if result:
                        _, proc_time = result
                        total_time += proc_time
                        current_processed += 1
                        
                        # Save progress periodically
                        if current_processed % 1000 == 0 or \
                           (processed_count + current_processed) % 100000 == 0 or \
                           current_processed == len(work_items):
                            self._save_progress(
                                processed_count + current_processed,
                                work_items[current_processed:]
                            )
                        
                        if current_processed % 2200 == 0:
                            avg_time = total_time / current_processed
                            print(f"\nProcessed {processed_count + current_processed}/{len(work_items)} mentions. Average time: {avg_time:.2f}s")
                            
        except KeyboardInterrupt:
            print("\nGracefully shutting down...")
            self._shutdown = True  # Set shutdown flag
            
            if executor:
                print("Canceling pending tasks...")
                executor.shutdown(wait=False)  # Don't wait for pending tasks
                
            print("Saving progress...")
            self._save_progress(processed_count + current_processed, work_items[current_processed:])
            print("Shutdown complete.")
            sys.exit(0)
            
        except Exception as e:
            print(f"\nError during processing: {e}")
            if executor:
                executor.shutdown(wait=False)
            self._save_progress(processed_count + current_processed, work_items[current_processed:])
            raise
        
        try:
            # Final progress save
            self._save_progress(processed_count + current_processed, [])
                            
        except Exception as e:
            error_msg = f"Critical error in analyze_file:\n{str(e)}"
            print(f"\n{error_msg}")
            self.notify_crash(error_msg)
            raise  # Re-raise the exception after notification

    def analyze_results(self) -> pd.DataFrame:
        """Analyze and print statistics from results with improved metrics."""
        with self.db.get_connection() as conn:
            # Load results into DataFrame
            results_df = pd.read_sql_query("""
                SELECT * FROM sentiment_analysis 
                WHERE sentiment != 'NO_SENTIMENT' 
                AND sentiment != 'ERROR'
            """, conn)
            
            # Convert sentiment to numeric
            results_df['sentiment'] = pd.to_numeric(results_df['sentiment'], errors='coerce')
            
            # Calculate statistics
            rapper_stats = results_df.groupby('rapper_name').agg({
                'sentiment': ['count', 'mean', 'std'],
                'text': 'count',
                'processing_time': 'mean'
            }).round(2)
            
            rapper_stats.columns = ['sentiment_count', 'avg_sentiment', 'sentiment_std', 
                                  'total_mentions', 'avg_processing_time']
            rapper_stats = rapper_stats.sort_values('total_mentions', ascending=False)
            
            print("\nTop 10 most mentioned rappers:")
            print(rapper_stats.head(10))
            
            print("\nTop 10 highest average sentiment (min 5 mentions):")
            min_mentions = rapper_stats[rapper_stats['sentiment_count'] >= 5]
            print(min_mentions.sort_values('avg_sentiment', ascending=False).head(10))
            
            # Additional performance metrics
            total_samples = len(results_df)
            errors_count = pd.read_sql_query(
                "SELECT COUNT(*) FROM sentiment_analysis WHERE sentiment = 'ERROR'", 
                conn
            ).iloc[0,0]
            no_sentiment_count = pd.read_sql_query(
                "SELECT COUNT(*) FROM sentiment_analysis WHERE sentiment = 'NO_SENTIMENT'", 
                conn
            ).iloc[0,0]
            
            print("\nPerformance Metrics:")
            print(f"Total processed samples: {total_samples}")
            print(f"Error rate: {errors_count/total_samples*100:.2f}%")
            print(f"No sentiment rate: {no_sentiment_count/total_samples*100:.2f}%")
            print(f"Average processing time: {results_df['processing_time'].mean():.2f}s")
            
            return results_df

def main():
    # Get Discord webhook URL from environment variable or config file
    discord_webhook_url = os.getenv('DISCORD_WEBHOOK_URL')
    
    analyzer = SentimentAnalyzer(
        rapper_aliases_path='../Step 2.3 Train Word2Vec/rapper_aliases.json',
        db_path='rapper_sentiments.db'
    )
    
    analyzer.analyze_file(
        '../Step 2.2 Create Bi-and Trigrams for Word2Vec/2_2-sentences_with_ngrams.txt',
        discord_webhook_url=discord_webhook_url,
        n_workers=4  # Start with 2 workers, adjust based on your GPU's behavior
    )
    results_df = analyzer.analyze_results()
    
    # Save aggregated results to CSV for further analysis
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'sentiment_analysis_results_{timestamp}.csv', index=False)
    
    # Save aggregated results to CSV for further analysis
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'sentiment_analysis_results_{timestamp}.csv', index=False)

if __name__ == "__main__":
    main()