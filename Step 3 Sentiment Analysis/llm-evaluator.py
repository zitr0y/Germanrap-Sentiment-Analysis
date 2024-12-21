import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict
import ollama
from tqdm import tqdm
import time
from datetime import datetime
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

class LLMEvaluator:
    def __init__(self, test_set_path: str, models: List[str]):
        self.test_set = self.load_test_set(test_set_path)
        self.models = models
        self.results = {}
        
    def load_test_set(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_sentiment_prompt(self, text: str, found_alias: str, canonical_name: str) -> str:
        """Same prompt generation as in main script."""
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
    
    def evaluate_model(self, model_name: str, max_samples: int = None):
        """Evaluate a single model on the test set."""
        results = []
        samples = self.test_set[:max_samples] if max_samples else self.test_set
        
        for sample in tqdm(samples, desc=f"Evaluating {model_name}"):
            prompt = self.get_sentiment_prompt(
                sample['text'],
                sample['found_alias'],
                sample['rapper_name']
            )
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = ollama.generate(
                        model=model_name,
                        prompt=prompt
                    )
                    result = response['response'].strip().replace("'", "").replace('"', '')
                    
                    # Handle N/A cases
                    if result.upper() in ['N/A', 'NA', 'NONE', 'NULL']:
                        result = "N/A"
                        break
                        
                    # Handle numerical responses
                    try:
                        sentiment = int(result)
                        if 1 <= sentiment <= 5:
                            result = str(sentiment)
                            break
                    except ValueError:
                        if attempt == max_retries - 1:
                            result = "ERROR"
                            
                    time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    if attempt == max_retries - 1:
                        result = "ERROR"
                    time.sleep(2)
            
            results.append({
                'text': sample['text'],
                'rapper_name': sample['rapper_name'],
                'found_alias': sample['found_alias'],
                'human_sentiment': sample['human_sentiment'],
                'model_sentiment': result
            })
        
        self.results[model_name] = results
    
    def evaluate_all_models(self, max_samples: int = None):
        """Evaluate all models on the test set."""
        for model in self.models:
            self.evaluate_model(model, max_samples)
            
    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate metrics for all models."""
        metrics = []
        
        for model_name, results in self.results.items():
            df = pd.DataFrame(results)
            
            # Filter out errors and convert to numeric
            valid_results = df[
                (df['model_sentiment'] != 'ERROR') & 
                (df['model_sentiment'] != 'N/A') & 
                (df['human_sentiment'] != 'N/A')
            ]
            
            if len(valid_results) == 0:
                continue
                
            # Calculate metrics
            report = classification_report(
                valid_results['human_sentiment'],
                valid_results['model_sentiment'],
                output_dict=True
            )
            
            metrics.append({
                'model': model_name,
                'accuracy': report['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'samples': len(valid_results),
                'errors': len(df[df['model_sentiment'] == 'ERROR']),
                'na_agreement': (
                    (df['model_sentiment'] == 'N/A') & 
                    (df['human_sentiment'] == 'N/A')
                ).sum() / (df['human_sentiment'] == 'N/A').sum()
            })
        
        return pd.DataFrame(metrics)
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        plt.figure(figsize=(15, 5 * len(self.models)))
        
        for i, (model_name, results) in enumerate(self.results.items(), 1):
            df = pd.DataFrame(results)
            
            valid_results = df[
                (df['model_sentiment'] != 'ERROR') & 
                (df['model_sentiment'] != 'N/A') & 
                (df['human_sentiment'] != 'N/A')
            ]
            
            if len(valid_results) == 0:
                continue
                
            plt.subplot(len(self.models), 1, i)
            cm = confusion_matrix(
                valid_results['human_sentiment'],
                valid_results['model_sentiment'],
                labels=['1', '2', '3', '4', '5']
            )
            
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d',
                xticklabels=['1', '2', '3', '4', '5'],
                yticklabels=['1', '2', '3', '4', '5']
            )
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('Human Label')
            plt.xlabel('Model Label')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'confusion_matrices_{timestamp}.png')
        
    def save_results(self):
        """Save detailed results and metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        with open(f'evaluation_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        
        # Save metrics
        metrics_df = self.calculate_metrics()
        metrics_df.to_csv(f'evaluation_metrics_{timestamp}.csv', index=False)
        
        return metrics_df

def main():
    # List of models to evaluate
    models = [
        'llama2',
        'llama2:13b',
        'llama2:70b',
        'mistral',
        'mixtral'
    ]
    
    evaluator = LLMEvaluator('test_set.json', models)
    
    # Evaluate first 20 samples as a test run
    evaluator.evaluate_all_models(max_samples=20)
    
    # Calculate and display metrics
    metrics_df = evaluator.save_results()
    print("\nModel Performance Metrics:")
    print(metrics_df)
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrices()

if __name__ == "__main__":
    main()