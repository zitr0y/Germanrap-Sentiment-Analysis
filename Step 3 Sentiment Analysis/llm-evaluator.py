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

    def serialize_numpy(self, obj):
        """Handle numpy types for JSON serialization"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj
    
    def save_results(self):
        """Save evaluation metrics as CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate and save metrics
        metrics_df = self.calculate_metrics()
        output_path = f'evaluation_metrics_{timestamp}.csv'
        metrics_df.to_csv(output_path, index=False)
        
        print(f"\nSaved metrics to: {output_path}")
        return metrics_df
    
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
                        'timeout': 30  # Add timeout
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
            
        return None  # Invalid response    def __init__(self, test_set_path: str, models: List[str], temperature: float = 0.1):
        self.test_set = self.load_test_set(test_set_path)
        self.models = models
        self.temperature = temperature
        self.results = {}

    def serialize_numpy(self, obj):
        """Handle numpy types for JSON serialization"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.ndarray,)):
                return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    def save_results(self):
            """Save detailed results and metrics with proper type handling."""
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save raw results using custom encoder
            with open(f'evaluation_results_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(self.results, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
            
            # Save metrics
            metrics_df = self.calculate_metrics()
            metrics_df.to_csv(f'evaluation_metrics_{timestamp}.csv', index=False)
            
            # Save summary metrics as JSON for easier parsing
            metrics_dict = metrics_df.to_dict(orient='records')
            with open(f'evaluation_metrics_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
            
            return metrics_df

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
                        'timeout': 30  # Add timeout
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
            
        return None  # Invalid response    def __init__(self, test_set_path: str, models: List[str], temperature: float = 0.1):
        self.test_set = self.load_test_set(test_set_path)
        self.models = models
        self.temperature = temperature
        self.results = {}

    def serialize_numpy(self, obj):
        """Handle numpy types for JSON serialization"""
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return obj

    def save_results(self):
        """Save detailed results and metrics with proper type handling."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types in results
        serializable_results = {}
        for model_name, model_data in self.results.items():
            serializable_results[model_name] = {
                'samples': [{
                    k: self.serialize_numpy(v) for k, v in sample.items()
                } for sample in model_data['samples']],
                'performance_metrics': {
                    k: self.serialize_numpy(v) 
                    for k, v in model_data['performance_metrics'].items()
                }
            }
            
            # Handle additional metrics if they exist
            for extra_key in ['error_analysis', 'disagreement_stats']:
                if extra_key in model_data:
                    serializable_results[model_name][extra_key] = {
                        k: self.serialize_numpy(v) if not isinstance(v, dict) else
                           {sk: self.serialize_numpy(sv) for sk, sv in v.items()}
                        for k, v in model_data[extra_key].items()
                    }

        # Save raw results
        with open(f'evaluation_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        # Save metrics
        metrics_df = self.calculate_metrics()
        metrics_df.to_csv(f'evaluation_metrics_{timestamp}.csv', index=False)
        
        # Save summary metrics as JSON for easier parsing
        metrics_dict = metrics_df.to_dict(orient='records')
        with open(f'evaluation_metrics_{timestamp}.json', 'w', encoding='utf-8') as f:
            json.dump(metrics_dict, f, ensure_ascii=False, indent=2)
        
        return metrics_df

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
                        'timeout': 30  # Add timeout
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
            
        return None  # Invalid response    def __init__(self, test_set_path: str, models: List[str], temperature: float = 0.1):
        self.test_set = self.load_test_set(test_set_path)
        self.models = models
        self.temperature = temperature
        self.results = {}
        
    def load_test_set(self, path: str) -> List[Dict]:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_sentiment_prompt(self, text: str, found_alias: str, canonical_name: str) -> str:
        """Same prompt generation as in main script."""
        examples = """
        Beispiele für Bewertungen:
        - "<<X>> ist der krasseste/beste Rapper" -> 5
        - "<<X>> einfach mies geil/übelst stark/richtig gut/hammer/brutal/wild/krass" -> 5
        - "<<X>> feier ich/stark/kann was/nice/gut" -> 4
        - "<<X>> hat ein neues Album released" -> 3
        - "<<X>> wurde in Y gesehen" -> 3
        - "<<X>> nicht so meins/schwach/eher wack/austauschbar" -> 2
        - "<<X>> absoluter müll/scheisse/kacke/trash/wack/cringe/hurensohn" -> 1
        - "<<X>> erinnert mich an Y" -> 3
        - "Ich höre gerade <<X>>" -> 3
        - "Rapper: <<Boss>>", Text: "Kollegah ist der <<Boss>>" -> 'N/A'
        - "Rapper: Germany", Text: "I'm coming to <<Germany>> this Summer" -> 'N/A' 
        - "Rapper: <<fabian_roemer>> (Alias 'fr')", Text: "armutszeugnis fuer op <<fr>>" ->'N/A'
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
        - Betrachte nur das Sentiment gegenüber dem mit <<...>> markierten Rapper
        """

        marked_text = text.replace(found_alias.replace('_', ' '), f"<<{found_alias.replace('_', ' ')}>>" , 1)

        if found_alias.replace(' ', '_') == canonical_name.lower():
            rapper_reference = f"dem Rapper {canonical_name}"
        else:
            rapper_reference = f"{found_alias.replace('_', ' ')} (vermuteter Alias des Rappers {canonical_name})"
            
        return f"""Bewerte das Sentiment in diesem Text gegenüber {rapper_reference}.

                {instructions}

                {examples}

                Text: {marked_text}

                Antworte NUR mit einer einzelnen Zahl (1-5) oder 'N/A'. Keine weiteren Wörter oder Erklärungen."""
    
    def evaluate_model(self, model_name: str, max_samples: int = None):
        """Evaluate a single model on the test set."""
        results = []
        samples = self.test_set[:max_samples] if max_samples else self.test_set
        total_time = 0
        processed_samples = 0
        num_samples = len(samples)
        
        for sample in tqdm(samples, desc=f"Evaluating {model_name}", leave=False):
            start_time = time.time()
            prompt = self.get_sentiment_prompt(
                sample['text'],
                sample['found_alias'],
                sample['rapper_name']
            )
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # For the last sample, add keep_alive: 0 to unload the model
                    options = {
                        'temperature': self.temperature
                    }
                    if processed_samples == num_samples - 1:  # If this is the last sample
                        options['keep_alive'] = 0
                        
                    response = ollama.generate(
                        model=model_name,
                        prompt=prompt,
                        options=options
                    )
                    result = response['response'].strip().replace("'", "").replace('"', '')
                    
                    if result.upper() in ['N/A', 'NA', 'NONE', 'NULL']:
                        result = "N/A"
                        break
                        
                    try:
                        # Add debug logging for response parsing
                        # print(f"\nDebug - Raw response for {model_name}: '{result}'")
                        
                        # Handle common variations of N/A responses
                        if any(na_variant in result.upper() for na_variant in ['N/A', 'NA', 'NONE', 'NULL', 'NOT APPLICABLE']):
                            result = "N/A"
                            break
                            
                        # Try to extract just the number if there's extra text
                        import re
                        number_match = re.search(r'[1-5]', result)
                        if number_match:
                            result = number_match.group()
                            sentiment = int(result)
                            if 1 <= sentiment <= 5:
                                break
                            
                        # If no number found, try direct conversion
                        sentiment = int(result)
                        if 1 <= sentiment <= 5:
                            result = str(sentiment)
                            break
                        else:
                            print(f"Warning - Value {sentiment} outside valid range 1-5")
                            
                    except ValueError as e:
                        print(f"Warning - Could not parse response '{result}': {str(e)}")
                        if attempt == max_retries - 1:
                            result = "ERROR"
                            
                    #time.sleep(1)  # Rate limiting
                    
                except Exception as e:
                    print(f"Error with {model_name}: {e}")
                    if attempt == max_retries - 1:
                        result = "ERROR"
                    #time.sleep(2)
            
            end_time = time.time()
            sample_time = end_time - start_time
            total_time += sample_time
            processed_samples += 1
            
            results.append({
                'text': sample['text'],
                'rapper_name': sample['rapper_name'],
                'found_alias': sample['found_alias'],
                'human_sentiment': sample['human_sentiment'],
                'model_sentiment': result,
                'processing_time': sample_time
            })
        
        # Calculate average time per sample and estimated total time for 400k samples
        avg_time_per_sample = total_time / processed_samples if processed_samples > 0 else 0
        estimated_total_time = avg_time_per_sample * 400000  # For 400k samples
        
        self.results[model_name] = {
            'samples': results,
            'performance_metrics': {
                'avg_time_per_sample': avg_time_per_sample,
                'estimated_total_time': estimated_total_time
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

    def calculate_weighted_disagreement(self, df):
        """
        Calculate weighted disagreement score where:
        - Off by 1 (e.g., 3 vs 4): weight = 1
        - Off by 2 (e.g., 3 vs 5): weight = 2
        - Off by 3: weight = 4
        - Off by 4: weight = 8
        Returns average weighted disagreement per prediction
        """
        def numeric_or_none(x):
            try:
                return float(x)
            except:
                return None
                
        # Convert to numeric, excluding N/A and ERROR
        human = df['human_sentiment'].apply(numeric_or_none)
        model = df['model_sentiment'].apply(numeric_or_none)
        
        # Only consider cases where both are numeric
        valid_mask = human.notna() & model.notna()
        human = human[valid_mask]
        model = model[valid_mask]
        
        if len(human) == 0:
            return None
            
        differences = abs(human - model)
        weights = 2 ** (differences - 1)  # 1->0.5, 2->1, 3->2, 4->4
        
        return {
            'avg_weighted_disagreement': weights.mean(),
            'disagreement_distribution': differences.value_counts().sort_index().to_dict(),
            'total_valid_samples': len(human),
            'exact_matches': (differences == 0).sum(),
            'off_by_one': (differences == 1).sum(),
            'off_by_two_plus': (differences >= 2).sum()
        }

    def analyze_errors(self, df, max_examples=5):
        """Analyze worst disagreements between model and human labels."""
        def numeric_or_none(x):
            try:
                return float(x)
            except:
                return None

        df = df.copy()
        df['human_numeric'] = df['human_sentiment'].apply(numeric_or_none)
        df['model_numeric'] = df['model_sentiment'].apply(numeric_or_none)
        
        # Calculate absolute difference for numeric cases
        mask = df['human_numeric'].notna() & df['model_numeric'].notna()
        df.loc[mask, 'difference'] = abs(df.loc[mask, 'human_numeric'] - df.loc[mask, 'model_numeric'])
        
        # Sort by difference and get worst cases
        worst_cases = df[mask].nlargest(max_examples, 'difference')
        
        return {
            'worst_cases': worst_cases[['text', 'human_sentiment', 'model_sentiment', 'difference']].to_dict('records'),
            'na_cases': df[df['model_sentiment'] == 'N/A'][['text', 'human_sentiment']].to_dict('records')[:max_examples]
        }

    def calculate_metrics(self) -> pd.DataFrame:
        """Calculate metrics for all models including timing information and disagreement stats."""
        metrics = []
        
        for model_name, model_results in self.results.items():
            results = model_results['samples']
            df = pd.DataFrame(results)
            
            # Debug information about results
            print(f"\nDebug - Model {model_name} results:")
            print(f"Total samples: {len(df)}")
            print("Response distribution:")
            print(df['model_sentiment'].value_counts())
            print("\nTrue label distribution:")
            print(df['human_sentiment'].value_counts())
            
            # Convert sentiment columns to numeric, handling errors
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
                (df['human_sentiment'] != 'N/A') &
                numeric_model.notna() &
                numeric_human.notna()
            ].copy()
            
            # Ensure we have valid numeric values for metrics calculation
            valid_results['model_sentiment'] = valid_results['model_sentiment'].astype(float).astype(str)
            valid_results['human_sentiment'] = valid_results['human_sentiment'].astype(float).astype(str)
            
            print(f"\nValid samples for {model_name}: {len(valid_results)}/{len(df)}")
            if len(valid_results) > 0:
                print("Valid response distribution:")
                print(valid_results['model_sentiment'].value_counts())
            
            if len(valid_results) == 0:
                continue
                
            report = classification_report(
                valid_results['human_sentiment'],
                valid_results['model_sentiment'],
                output_dict=True
            )
            
            # Calculate weighted disagreement
            disagreement_stats = self.calculate_weighted_disagreement(df)
            error_analysis = self.analyze_errors(df)
            
            metrics_dict = {
                'model': model_name,
                'accuracy': report['accuracy'],
                'macro_f1': report['macro avg']['f1-score'],
                'weighted_f1': report['weighted avg']['f1-score'],
                'samples': len(valid_results),
                'errors': len(df[df['model_sentiment'] == 'ERROR']),
                'na_agreement': (
                    (df['model_sentiment'] == 'N/A') & 
                    (df['human_sentiment'] == 'N/A')
                ).sum() / max((df['human_sentiment'] == 'N/A').sum(), 1),  # Avoid division by zero
                'avg_time_per_sample': model_results['performance_metrics']['avg_time_per_sample'],
                'estimated_hours_400k': model_results['performance_metrics']['estimated_total_time'] / 3600,
            }
            
            if disagreement_stats:
                metrics_dict.update({
                    'avg_weighted_disagreement': disagreement_stats['avg_weighted_disagreement'],
                    'exact_match_rate': disagreement_stats['exact_matches'] / disagreement_stats['total_valid_samples'],
                    'off_by_one_rate': disagreement_stats['off_by_one'] / disagreement_stats['total_valid_samples'],
                    'off_by_two_plus_rate': disagreement_stats['off_by_two_plus'] / disagreement_stats['total_valid_samples']
                })
            
            # Save detailed results including error analysis
            model_results['error_analysis'] = error_analysis
            model_results['disagreement_stats'] = disagreement_stats
            
            metrics.append(metrics_dict)
        
        return pd.DataFrame(metrics)

    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models."""
        plt.figure(figsize=(15, 5 * len(self.models)))
        
        for i, (model_name, model_results) in enumerate(self.results.items(), 1):
            df = pd.DataFrame(model_results['samples'])
            
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
               
        # Save metrics
        metrics_df = self.calculate_metrics()
        metrics_df.to_csv(f'evaluation_metrics_{timestamp}.csv', index=False)
        
        return metrics_df

def main2():
    # Test different temperatures for qwen
    temperatures = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    
    for temp in tqdm(temperatures):
        print(f"\nTesting qwen2.5:3b with temperature {temp}")
        
        # Initialize evaluator with just this temperature variant
        evaluator = LLMEvaluator('test_set.json', [f"qwen2.5:3b"], temp)
        
        # Evaluate all models (in this case just one)
        evaluator.evaluate_all_models(max_samples=200)
        
        # Calculate and display metrics
        metrics_df = evaluator.calculate_metrics()
        print("\nModel Performance Metrics:")
        print(metrics_df.to_string())
        
        # Save results for this temperature
        evaluator.save_results()



def main():
    # List of models to evaluate
    models = [
        'llama3.1',
        #'germanrapllm_Q8_v2',
        'qwen2.5:3b',
        #'wizardlm2',
        #'mixtral',
        #'qwen2.5:7b',
        'mistral',
        #'gemma:7b'
    ]
    
    # Initialize evaluator with low temperature for more consistent results
    evaluator = LLMEvaluator('test_set.json', models, temperature=0.3)
    
    # Evaluate first 5 samples as a test run
    evaluator.evaluate_all_models(max_samples=200)
    
    # Calculate and display metrics
    metrics_df = evaluator.save_results()
    print("\nModel Performance Metrics:")
    print(metrics_df.to_string())
    
    # Plot confusion matrices
    evaluator.plot_confusion_matrices()

if __name__ == "__main__":
    main()