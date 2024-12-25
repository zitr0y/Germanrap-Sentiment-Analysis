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
from datetime import datetime, timedelta
import os
import psutil
import re

os.chdir(os.path.dirname(os.path.realpath(__file__)))

class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        return super().default(obj)

class LLMEvaluator:
    def __init__(self, test_set_path: str, models: List[str], temperature: float = 0.1):
        self.test_set = self.load_test_set(test_set_path)
        self.models = models
        self.temperature = temperature
        self.results = {}

    def load_test_set(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_sentiment_prompt(self, text: str, found_alias: str, canonical_name: str) -> str:
        examples = """
        Beispiele:
        Sehr positiv (5):
        - "<<X>> ist der beste Rapper"
        - "<<X>> ist krass/brutal/hammer"
        - "<<X>> absoluter ehrenmove/bestes album des jahres"

        Eher positiv (4):
        - "<<X>> feier ich/macht stabile Musik"
        - "<<X>> macht seit Jahren gute Sachen"
        - "bin kein fan aber <<X>> hat skills"

        Neutral/Unklar (3):
        - "<<X>> hat neues Album released"
        - "<<X>> wurde gesehen bei/in Y" 
        - "Ich höre <<X>>, <<Y>>, <<Z>>"
        - "<<X>> macht halt sein Ding"

        Eher negativ (2):
        - "<<X>> ist nicht so meins"
        - "früher war <<X>> besser"
        - "<<X>> wird überbewertet"

        Sehr negativ (1):
        - "<<X>> ist müll/hurensohn"
        - "<<X>> sollte aufhören"
        - "<<X>> komplett peinlich/whack"
        """
        
        instructions = """
        Regeln:
        - Wenn der Kontext unklar ist -> 3
        - Slang beachten (negativ kann positiv sein)
        - Nur den markierten Rapper bewerten
        - Auch indirekte Aussagen bewerten ("der Track ist whack" = negativ)
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
            return "N/A"
            
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
            
        return None  # Invalid response

    def evaluate_model(self, model_name: str, max_samples: int = None):
        """Evaluate a single model with improved error handling and rate limiting."""
        results = []
        samples = self.test_set[:max_samples] if max_samples else self.test_set
        total_time = 0
        processed_samples = 0
        num_samples = len(samples)
        
        # Add exponential backoff for retries
        backoff_times = [1, 2, 4]  # seconds
        
        for sample in tqdm(samples, desc=f"Evaluating {model_name}", leave=False):
            start_time = time.time()
            prompt = self.get_sentiment_prompt(
                sample['text'],
                sample['found_alias'],
                sample['rapper_name']
            )
            
            result = None
            for attempt, backoff in enumerate(backoff_times):
                try:
                    options = {
                        'temperature': self.temperature,
                        'timeout': 30
                    }
                    if processed_samples == num_samples - 1:
                        options['keep_alive'] = 0
                        
                    response = ollama.generate(
                        model=model_name,
                        prompt=prompt,
                        options=options
                    )
                    
                    raw_result = response['response'].strip().replace("'", "").replace('"', '')
                    result = self._parse_response(raw_result)
                    if result:  # If we got a valid result
                        break
                        
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for {model_name}: {str(e)}")
                    if attempt < len(backoff_times) - 1:
                        time.sleep(backoff)
                    else:
                        result = "ERROR"
            
            end_time = time.time()
            sample_time = end_time - start_time
            
            results.append({
                'text': sample['text'],
                'rapper_name': sample['rapper_name'],
                'found_alias': sample['found_alias'],
                'human_sentiment': sample['human_sentiment'],
                'model_sentiment': result or "ERROR",
                'processing_time': sample_time
            })
            
            total_time += sample_time
            processed_samples += 1
            
            # Add a small delay between requests to prevent rate limiting
            time.sleep(0.1)
        
        avg_time_per_sample = total_time / processed_samples if processed_samples > 0 else 0
        estimated_total_time = avg_time_per_sample * 400000
        
        self.results[model_name] = {
            'samples': results,
            'performance_metrics': {
                'avg_time_per_sample': avg_time_per_sample,
                'estimated_total_time': estimated_total_time,
                'total_errors': sum(1 for r in results if r['model_sentiment'] == "ERROR"),
                'total_na': sum(1 for r in results if r['model_sentiment'] == "N/A")
            }
        }

    def evaluate_all_models(self, max_samples: int = None):
        """Evaluate all models with enhanced timing statistics."""
        stats = {}
        
        for model in tqdm(self.models, desc="Evaluating all models", leave=True):
            stats[model] = {
                'start_time': time.time(),
                'samples_processed': 0,
                'errors': [],
                'memory_before': None,
                'memory_after': None,
                'success': False
            }
            
            try:
                try:
                    ollama.generate(model=model, prompt="test", options={'temperature': self.temperature})
                    stats[model]['model_exists'] = True
                except Exception as e:
                    print(f"Error: Model {model} not accessible: {e}")
                    stats[model]['model_exists'] = False
                    stats[model]['errors'].append(f"Model not accessible: {str(e)}")
                    continue
                
                stats[model]['memory_before'] = psutil.Process().memory_info().rss / 1024 / 1024
                
                start_time = time.time()
                self.evaluate_model(model, max_samples)
                eval_time = time.time() - start_time
                
                stats[model]['memory_after'] = psutil.Process().memory_info().rss / 1024 / 1024
                stats[model]['evaluation_time'] = eval_time
                stats[model]['success'] = True
                
                # Calculate and format estimated time for 400k samples
                avg_time = self.results[model]['performance_metrics']['avg_time_per_sample']
                estimated_total_time = self.results[model]['performance_metrics']['estimated_total_time']
                estimated_duration = timedelta(seconds=estimated_total_time)
                
                print(f"\nModel {model} evaluation stats:")
                print(f"- Time: {eval_time:.2f} seconds")
                print(f"- Average time per sample: {avg_time:.2f} seconds")
                print(f"- Estimated time for 400k samples: {estimated_duration}")
                print(f"- Memory delta: {stats[model]['memory_after'] - stats[model]['memory_before']:.1f} MB")
                if stats[model]['errors']:
                    print(f"- Warnings/Errors: {len(stats[model]['errors'])}")
                
            except Exception as e:
                print(f"\nError during evaluation of {model}: {e}")
                stats[model]['errors'].append(f"Evaluation error: {str(e)}")
                stats[model]['success'] = False
        
        return stats

    def analyze_errors(self, df, max_examples=3):
        """Analyze worst disagreements between model and human labels."""
        # Convert sentiments to numeric for comparison
        def numeric_or_none(x):
            if x == 'N/A':
                return None
            try:
                return float(x)
            except:
                return None

        df = df.copy()
        df['human_numeric'] = df['human_sentiment'].apply(numeric_or_none)
        df['model_numeric'] = df['model_sentiment'].apply(numeric_or_none)
        
        # Find NA misclassifications
        na_mismatches = df[
            ((df['human_sentiment'] == 'N/A') & (df['model_sentiment'] != 'N/A')) |
            ((df['human_sentiment'] != 'N/A') & (df['model_sentiment'] == 'N/A'))
        ]
        
        # Find large sentiment misclassifications (difference ≥ 2)
        mask = df['human_numeric'].notna() & df['model_numeric'].notna()
        df.loc[mask, 'difference'] = abs(df.loc[mask, 'human_numeric'] - df.loc[mask, 'model_numeric'])
        large_mismatches = df[df['difference'] >= 2].sort_values('difference', ascending=False)
        
        #print("\nInteresting misclassifications:")
        
        #print("\nN/A Misclassifications:")
        #for _, row in na_mismatches.head(max_examples).iterrows():
        #    print(f"\nText: {row['text']}")
        #    print(f"Found alias: {row['found_alias']}")
        #    print(f"Human: {row['human_sentiment']} | Model: {row['model_sentiment']}")
            
        #print("\nLarge Sentiment Misclassifications (diff ≥ 2):")
        #for _, row in large_mismatches.head(max_examples).iterrows():
        #    print(f"\nText: {row['text']}")
        #    print(f"Found alias: {row['found_alias']}")
        #    print(f"Human: {row['human_sentiment']} | Model: {row['model_sentiment']}")
            
        return {
            'na_mismatches': len(na_mismatches),
            'large_mismatches': len(large_mismatches),
            'examples': {
                'na': na_mismatches.head(max_examples)[['text', 'found_alias', 'human_sentiment', 'model_sentiment']].to_dict('records'),
                'large': large_mismatches.head(max_examples)[['text', 'found_alias', 'human_sentiment', 'model_sentiment', 'difference']].to_dict('records')
            }
        }

    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate metrics for all models."""
        metrics = []
        
        for model_name, model_results in self.results.items():
            results = model_results['samples']
            df = pd.DataFrame(results)
            
            # Convert N/A to 3 in ground truth
            df['human_sentiment'] = df['human_sentiment'].replace('N/A', '3')
            
            print(f"\nDebug - Model {model_name} results:")
            print(f"Total samples: {len(df)}")
            print("Response distribution:")
            print(df['model_sentiment'].value_counts())
            print("\nTrue label distribution:")
            print(df['human_sentiment'].value_counts())
            
            def convert_to_numeric(x):
                try:
                    return pd.to_numeric(x)
                except:
                    return None
                    
            numeric_model = df['model_sentiment'].apply(convert_to_numeric)
            numeric_human = df['human_sentiment'].apply(convert_to_numeric)
            
            valid_results = df[
                (df['model_sentiment'] != 'ERROR') & 
                (df['model_sentiment'] != 'N/A') & 
                numeric_model.notna() &
                numeric_human.notna()
            ].copy()
            
            if len(valid_results) == 0:
                continue
                
            # Ensure numeric values for metrics calculation
            valid_results['model_sentiment'] = valid_results['model_sentiment'].astype(float).astype(str)
            valid_results['human_sentiment'] = valid_results['human_sentiment'].astype(float).astype(str)
            
            print(f"\nValid samples for {model_name}: {len(valid_results)}/{len(df)}")
            if len(valid_results) > 0:
                print("Valid response distribution:")
                print(valid_results['model_sentiment'].value_counts())
            
            report = classification_report(
                valid_results['human_sentiment'],
                valid_results['model_sentiment'],
                output_dict=True
            )
            
            disagreement_stats = self.calculate_weighted_disagreement(df)
            error_analysis = self.analyze_errors(df)
            
            metrics_dict = {
                'model': model_name,
                'accuracy': report['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'samples': len(valid_results),
                'errors': len(df[df['model_sentiment'] == 'ERROR']),
            }
            
            if disagreement_stats:
                metrics_dict.update({
                    'avg_weighted_disagreement': disagreement_stats['avg_weighted_disagreement'],
                    'exact_match_rate': disagreement_stats['exact_matches'] / disagreement_stats['total_valid_samples'],
                    'off_by_one_rate': disagreement_stats['off_by_one'] / disagreement_stats['total_valid_samples'],
                    'off_by_two_plus_rate': disagreement_stats['off_by_two_plus'] / disagreement_stats['total_valid_samples']
                })
            
            model_results['error_analysis'] = error_analysis
            model_results['disagreement_stats'] = disagreement_stats
            
            metrics.append(metrics_dict)
        
        return pd.DataFrame(metrics)

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        plt.figure(figsize=(20, 6 * len(self.models)))  # Increased width for N/A column
        
        for i, (model_name, model_results) in enumerate(self.results.items(), 1):
            df = pd.DataFrame(model_results['samples'])
            
            # Only filter out ERROR responses
            valid_results = df[df['model_sentiment'] != 'ERROR']
            
            if len(valid_results) == 0:
                continue
                
            plt.subplot(len(self.models), 1, i)
            
            # Include N/A in labels
            labels = ['1', '2', '3', '4', '5', 'N/A']
            cm = confusion_matrix(
                valid_results['human_sentiment'],
                valid_results['model_sentiment'],
                labels=labels
            )
            
            # Create a normalized confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0
            
            # Plot with both absolute numbers and percentages
            sns.heatmap(
                cm_normalized,
                annot=cm,  # Show absolute numbers
                fmt='d',
                xticklabels=labels,
                yticklabels=labels,
                cmap='YlOrRd'  # Changed colormap for better visibility
            )
            plt.title(f'Confusion Matrix - {model_name}\n(Values: Absolute, Colors: Row-normalized)')
            plt.ylabel('Human Label')
            plt.xlabel('Model Label')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'confusion_matrices_{timestamp}.png', dpi=300, bbox_inches='tight')

    def calculate_weighted_disagreement(self, df):
        """Calculate weighted disagreement score."""
        def numeric_or_none(x):
            if x == 'N/A':
                return x
            try:
                return float(x)
            except:
                return None
                
        human = df['human_sentiment'].apply(numeric_or_none)
        model = df['model_sentiment'].apply(numeric_or_none)
        
        # Handle numeric comparisons separately from N/A
        numeric_mask = (human != 'N/A') & (model != 'N/A') & human.notna() & model.notna()
        na_mask = (human == 'N/A') | (model == 'N/A')
        
        numeric_human = human[numeric_mask].astype(float)
        numeric_model = model[numeric_mask].astype(float)
        
        if len(numeric_human) == 0 and len(na_mask) == 0:
            return None
            
        # Calculate differences for numeric values
        differences = abs(numeric_human - numeric_model) if len(numeric_human) > 0 else pd.Series()
        weights = 2 ** (differences - 1) if len(differences) > 0 else pd.Series()
        
        # Calculate N/A agreement
        na_agreement = ((human == 'N/A') & (model == 'N/A')).sum()
        na_disagreement = ((human == 'N/A') ^ (model == 'N/A')).sum()
        
        return {
            'avg_weighted_disagreement': weights.mean() if len(weights) > 0 else None,
            'disagreement_distribution': differences.value_counts().sort_index().to_dict() if len(differences) > 0 else {},
            'total_valid_samples': len(numeric_human) + na_mask.sum(),
            'exact_matches': (differences == 0).sum() + na_agreement,
            'off_by_one': (differences == 1).sum() if len(differences) > 0 else 0,
            'off_by_two_plus': (differences >= 2).sum() if len(differences) > 0 else 0,
            'na_agreement': na_agreement,
            'na_disagreement': na_disagreement
        }

        
    def save_results(self):
        """Save detailed results and metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save raw results
        with open(f'evaluation_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(self.results, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)
        
        # Save metrics
        metrics_df = self.calculate_metrics()
        metrics_df.to_csv(f'evaluation_metrics_{timestamp}.csv', index=False)
        
        return metrics_df

def main():
    """Main function to run the evaluator."""
    models = [
        #'llama3.1',
        #'germanrapllm_Q8_v2',
        #'qwen2.5:3b',
        #'wizardlm2',
        #'mixtral',
        #'qwen2.5:7b',
        #'mistral',
        #'gemma:7b',
        #'aya-expanse:8b',
        'granite3.1-dense:2b',
        #'qwen2.5:1.5b',
        #'granite3.1-moe:1b',
        #'granite3.1-moe:3b',
    ]
    
    evaluator = LLMEvaluator('test_set.json', models, temperature=0.3)
    evaluator.evaluate_all_models(max_samples=200)
    metrics_df = evaluator.save_results()
    print("\nModel Performance Metrics:")
    print(metrics_df.to_string())
    evaluator.plot_confusion_matrices()

if __name__ == "__main__":
    main()