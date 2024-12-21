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
        
    def _get_existing_result(self, text: str, rapper_name: str) -> Optional[str]:
        """Get existing sentiment result from database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT sentiment FROM sentiment_analysis WHERE text = ? AND rapper_name = ?",
                (text, rapper_name)
            )
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _save_result(self, text: str, rapper_name: str, found_alias: str, sentiment: str):
        """Save sentiment result to database."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sentiment_analysis 
                (text, rapper_name, found_alias, sentiment)
                VALUES (?, ?, ?, ?)
            """, (text, rapper_name, found_alias, str(sentiment)))
            conn.commit()

class SentimentAnalyzer(RapperMentionAnalyzer):
    def __init__(self, rapper_aliases_path: str, db_path: str = 'rapper_sentiments.db'):
        super().__init__(db_path)
        self.name_mapping = self._load_rapper_aliases(rapper_aliases_path)
        self.reverse_mapping = self._create_reverse_mapping()
        
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
        
        for variant in self.name_mapping.keys():
            variant = variant.replace('_', ' ')
            if re.search(r'\b' + re.escape(variant) + r'\b', text):
                found_rappers.add((variant, self.name_mapping[variant.replace(' ', '_')]))
        
        return list(found_rappers)

    def get_sentiment_prompt(self, text: str, found_alias: str, canonical_name: str) -> str:
        """Create sentiment analysis prompt including both alias and canonical name."""
        examples = """
        Beispiele für Bewertungen:
        - "X ist der krasseste/beste Rapper" -> 5
        - "mies geil/übelst stark/richtig gut/hammer/brutal/wild/krass" -> 5
        - "feier ich/stark/kann was/nice/gut" -> 4
        - "X hat ein neues Album released" -> 3
        - "X wurde in Y gesehen" -> 3
        - "nicht so meins/schwach/eher wack/austauschbar" -> 2
        - "absoluter müll/scheisse/kacke/trash/wack/cringe/hurensohn" -> 1
        - "X erinnert mich an Y" -> 3
        - "Ich höre gerade X" -> 3
        - "Rapper: Favorite", Text: "Berlin ist meine favorite Stadt" -> 'N/A'
        - "Rapper: Germany", Text: "I'm coming to Germany this Summer" -> 'N/A' 
        - "Rapper: fabian_roemer (Alias 'fr')", Text: "armutszeugnis fuer op fr" ->'N/A'
        """
        
        instructions = """
        Bewerte das Sentiment auf einer Skala von 1-5:
        1 = sehr negativ (Hass/Abscheu)
        2 = eher negativ (Kritik/Ablehnung)
        3 = neutral (faktische Erwähnung/unklar/gemischt)
        4 = eher positiv (Zustimmung/Gefallen)
        5 = sehr positiv (Begeisterung/Verehrung)
        
        Wichtige Regeln:
        - Antworte NUR mit einer einzelnen Zahl (1-5) oder 'N/A'
        - Bei Unsicherheit oder neutraler Erwähnung -> 3
        - Bei keinem echten Bezug zum Rapper -> N/A
        - Beachte Deutschrap-Slang (negativ klingende Wörter können positiv sein)
        - Ignoriere moralische Bedenken, bewerte nur das Sentiment
        """

        if found_alias.replace(' ', '_') == canonical_name.lower():
            rapper_reference = f"dem Rapper {canonical_name}"
        else:
            rapper_reference = f"{found_alias.replace('_', ' ')} (vermuteter Alias des Rappers {canonical_name})"
            
        return f"""Bewerte das Sentiment in diesem Text gegenüber {rapper_reference}.

                {instructions}

                {examples}

                Text: {text}

                Antworte NUR mit einer einzelnen Zahl (1-5) oder 'N/A'. Keine weiteren Wörter oder Erklärungen."""

    def get_sentiment(self, text: str, found_alias: str, canonical_name: str) -> Optional[str]:
        """Get sentiment rating from LLM."""
        # Check if we already have this result
        existing_result = self._get_existing_result(text, canonical_name)
        if existing_result is not None:
            return existing_result
        
        prompt = self.get_sentiment_prompt(text, found_alias, canonical_name)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                response = ollama.generate(
                    model='llama3.1',
                    prompt=prompt
                )
                result = response['response'].strip().replace("'", "").replace('"', '')
                
                # Handle N/A cases
                if result.upper() in ['N/A', 'NA', 'NONE', 'NULL']:
                    result = "NO_SENTIMENT"
                    break
                    
                # Handle numerical responses
                try:
                    sentiment = int(result)
                    if 1 <= sentiment <= 5:  # Only allow values 1-5
                        result = sentiment
                        break
                    else:
                        print(f"Retry {retry_count + 1}: Number out of range: {result}")
                        retry_count += 1
                        continue
                except ValueError:
                    print(f"Retry {retry_count + 1}: Invalid response format: {result}")
                    retry_count += 1
                    continue
                    
            except Exception as e:
                print(f"Error getting sentiment: {e}")
                retry_count += 1
                if retry_count == max_retries:
                    result = "ERROR"
        
        # Save result immediately
        self._save_result(text, canonical_name, found_alias, result)
        
        return result

    def analyze_file(self, text_path: str) -> None:
        """Analyze file and save results."""
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        for line in tqdm(lines, desc="Processing lines"):
            rapper_mentions = self.find_rappers_in_text(line)
            
            if rapper_mentions:
                for found_alias, canonical_name in tqdm(rapper_mentions, desc="Processing rappers", leave=False):
                    self.get_sentiment(line, found_alias, canonical_name)

    def analyze_results(self) -> pd.DataFrame:
        """Analyze and print statistics from results."""
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
                'text': 'count'
            }).round(2)
            
            rapper_stats.columns = ['sentiment_count', 'avg_sentiment', 'sentiment_std', 'total_mentions']
            rapper_stats = rapper_stats.sort_values('total_mentions', ascending=False)
            
            print("\nTop 10 most mentioned rappers:")
            print(rapper_stats.head(10))
            
            print("\nTop 10 highest average sentiment (min 5 mentions):")
            min_mentions = rapper_stats[rapper_stats['sentiment_count'] >= 5]
            print(min_mentions.sort_values('avg_sentiment', ascending=False).head(10))
            
            return results_df

def main():
    analyzer = SentimentAnalyzer(
        rapper_aliases_path='../Step 2.3 Train Word2Vec/rapper_aliases.json',
        db_path='rapper_sentiments.db'
    )
    
    analyzer.analyze_file('../Step 2.2 Create Bi-and Trigrams for Word2Vec/2_2-sentences_with_ngrams.txt')
    results_df = analyzer.analyze_results()
    
    # Save aggregated results to CSV for further analysis if needed
    results_df.to_csv('sentiment_analysis_results.csv', index=False)

if __name__ == "__main__":
    main()