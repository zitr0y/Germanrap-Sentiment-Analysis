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

os.chdir(os.path.dirname(os.path.realpath(__file__)))

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

class RapperMentionAnalyzer:
    def __init__(self, output_path: str = 'rapper_sentiments.json'):
        self.output_path = output_path
        self.existing_results = self._load_existing_results()
        
    def _load_existing_results(self) -> Dict:
        """Load existing results if any, otherwise return empty dict."""
        if Path(self.output_path).exists():
            with open(self.output_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_results(self, results: Dict):
        """Save results to file."""
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def _get_result_key(self, text: str, rapper_name: str) -> str:
        """Create unique key for each text-rapper combination."""
        return f"{text}|||{rapper_name}"

class SentimentAnalyzer(RapperMentionAnalyzer):
    def __init__(self, rapper_aliases_path: str, output_path: str = 'rapper_sentiments.json'):
        super().__init__(output_path)
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
        if found_alias.replace(' ', '_') == canonical_name.lower():
            # If the found name is the canonical name
            return f"""Bewerte das Sentiment in diesem Text gegenüber dem Rapper {canonical_name}. 
            Bewerte von 1 bis 5, von sehr schlecht bis sehr gut, antworte mit 'N/A', falls es nicht um den Rapper geht, und 3, wenn der rapper zwar erwähnt wird, aber keine Bewertung vorliegt.
            1 ist sehr schlecht, 2 ist eher schlecht, 3 ist neutral oder nur erwaehnt, 4 ist eher gut und 5 ist sehr gut. Vergesse nicht, dass die Sprache eventuell Slang benutzt, 'mies geil' ist zum Beispiel eine 5.
            Antworte NUR mit einer Zahl (1 bis 5) oder 'N/A'. Antworte NUR damit. Wenn du dir nicht sicher bist, wie die Bewertung lauten sollte, antworte einfach 3.

            Text: {text}"""
        else:
            # If the found name is an alias
            return f"""Bewerte das Sentiment in diesem Text gegenüber {found_alias.replace('_', ' ')} (dem Künstler {canonical_name}). 
            Bewerte von 1 bis 5, von sehr schlecht bis sehr gut, antworte mit 'N/A', falls es nicht um den Rapper geht, und 3, wenn der rapper zwar erwähnt wird, aber keine Bewertung vorliegt.
            1 ist sehr schlecht, 2 ist eher schlecht, 3 ist neutral oder nur erwaehnt, 4 ist eher gut und 5 ist sehr gut. Vergesse nicht, dass die Sprache eventuell Slang benutzt, 'mies geil' ist zum Beispiel eine 5.
            Antworte NUR mit einer Zahl (1 bis 5) oder 'N/A'. Antworte NUR damit. Wenn du dir nicht sicher bist, wie die Bewertung lauten sollte, antworte einfach 3.

            Text: {text}"""

    def get_sentiment(self, text: str, found_alias: str, canonical_name: str) -> Optional[int]:
        """Get sentiment rating from LLM."""
        key = self._get_result_key(text, canonical_name)
        
        # Check if we already have this result
        if key in self.existing_results:
            return self.existing_results[key]
        
        prompt = self.get_sentiment_prompt(text, found_alias, canonical_name)
        try:
            response = ollama.generate(
                model='llama3.1',
                prompt=prompt
            )
            result = response['response'].strip()
            
            if result.strip() == 'N/A' or result.strip() == "'N/A'":
                result = "NO_SENTIMENT"
            elif result.strip() == 'NEUTRAL' or result.strip() == "'NEUTRAL'":
                result = "NEUTRAL SENTIMENT"
            else:
                try:
                    sentiment = int(result)
                    if 0 <= sentiment <= 5:
                        result = sentiment
                    else:
                        print(f"Number out of range: {result}")
                except ValueError:
                    print(f"Invalid response: {result}")
            
        except Exception as e:
            print(f"Error getting sentiment: {e}")
            result = "ERROR"
        
        # Save result immediately
        self.existing_results[key] = result
        self._save_results(self.existing_results)
        
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

    def analyze_results(self) -> None:
        """Analyze and print statistics from results."""
        results_df = pd.DataFrame([
            {
                'text': key.split('|||')[0],
                'rapper': key.split('|||')[1],
                'sentiment': value
            }
            for key, value in self.existing_results.items()
        ])
        
        # Rest of the analysis code remains the same
        # [Previous analysis code...]

def main():
    analyzer = SentimentAnalyzer(
        rapper_aliases_path='../Step 2.3 Train Word2Vec/rapper_aliases.json',
        output_path='3-rapper_sentiments.json'
    )
    
    analyzer.analyze_file('../Step 2.2 Create Bi-and Trigrams for Word2Vec/2_2-sentences_with_ngrams.txt')
    analyzer.analyze_results()

if __name__ == "__main__":
    main()