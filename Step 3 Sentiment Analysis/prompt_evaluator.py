import json
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from copy import deepcopy
import ollama
from tqdm import tqdm
import time
from datetime import datetime
import os
from llm_evaluator import LLMEvaluator, NumpyJSONEncoder
import sys

os.chdir(os.path.dirname(os.path.realpath(__file__)))

@dataclass
class PromptConfig:
    name: str
    instructions: str
    examples: str
    temperature: float = 0.3
    
class PromptEvaluator(LLMEvaluator):
    def __init__(self, test_set_path: str, models: List[str], prompt_configs: List[PromptConfig]):
        super().__init__(test_set_path, models, temperature=0.1)
        self.prompt_configs = prompt_configs
        self.all_results = {}

    def get_sentiment_prompt(self, text: str, found_alias: str, canonical_name: str, config: PromptConfig) -> str:
        marked_text = text.replace(found_alias.replace('_', ' '), f"<<{found_alias.replace('_', ' ')}>>" , 1)
            
        return f"""Bewerte das Sentiment zu {found_alias} im Text: {marked_text}

                {config.instructions}

                {config.examples}

                Antworte NUR mit einer Zahl von 1-5."""
                
    def evaluate_model_with_config(self, model_name: str, config: PromptConfig, max_samples: int = None):
        """Evaluate a single model with a specific prompt configuration."""
        self.temperature = config.temperature
        results = []
        samples = self.test_set[:max_samples] if max_samples else self.test_set
        total_time = 0
        processed_samples = 0
        
        for sample in tqdm(samples, desc=f"Evaluating {model_name} with {config.name}", leave=False):
            start_time = time.time()
            prompt = self.get_sentiment_prompt(
                sample['text'],
                sample['found_alias'],
                sample['rapper_name'],
                config
            )
            
            result = None
            try:
                response = ollama.generate(
                    model=model_name,
                    prompt=prompt,
                    options={'temperature': config.temperature}
                )
                raw_result = response['response'].strip().replace("'", "").replace('"', '')
                result = self._parse_response(raw_result)
            except Exception as e:
                print(f"Error evaluating {model_name}: {str(e)}")
                result = "ERROR"
            
            end_time = time.time()
            sample_time = end_time - start_time
            
            results.append({
                'text': sample['text'],
                'rapper_name': sample['rapper_name'],
                'found_alias': sample['found_alias'],
                'human_sentiment': sample['human_sentiment'],
                'model_sentiment': result or "ERROR",
                'processing_time': sample_time,
                'prompt_config': config.name
            })
            
            total_time += sample_time
            processed_samples += 1
            time.sleep(0.1)
        
        return results, total_time / processed_samples

    def evaluate_all_configurations(self, max_samples: int = None):
        """Evaluate all models with all prompt configurations."""
        all_metrics = []
        
        for model in tqdm(self.models, desc="Evaluating models"):
            print(f"\nEvaluating model: {model}")
            
            for config in tqdm(self.prompt_configs, desc="Evaluating configurations", leave=False):
                print(f"\nTesting configuration: {config.name}")
                
                try:
                    results, avg_time = self.evaluate_model_with_config(model, config, max_samples)
                    
                    # Store results
                    config_key = f"{model}_{config.name}"
                    self.all_results[config_key] = {
                        'samples': results,
                        'config': config,
                        'performance_metrics': {
                            'avg_time_per_sample': avg_time,
                            'total_errors': sum(1 for r in results if r['model_sentiment'] == "ERROR"),
                            'total_na': sum(1 for r in results if r['model_sentiment'] == "N/A")
                        }
                    }
                    
                    # Calculate metrics
                    df = pd.DataFrame(results)
                    metrics = self.calculate_configuration_metrics(df, model, config)
                    all_metrics.append(metrics)
                    
                    print(f"Completed evaluation of {config.name} for {model}")
                    print(f"Accuracy: {metrics['accuracy']:.3f}")
                    print(f"Weighted F1: {metrics['weighted_f1']:.3f}")
                    print(f"Average processing time: {avg_time:.2f}s")
                    
                except Exception as e:
                    print(f"Error evaluating {model} with config {config.name}: {e}")
                    continue
        
        return pd.DataFrame(all_metrics)


    def calculate_configuration_metrics(self, df: pd.DataFrame, model: str, config: PromptConfig) -> Dict:
        """Calculate metrics for a specific configuration, including weighted disagreement."""
        # Convert N/A to 3 in ground truth
        df['human_sentiment'] = df['human_sentiment'].replace('N/A', '3')
        
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
            return {
                'model': model,
                'config_name': config.name,
                'accuracy': 0,
                'weighted_f1': 0,
                'samples': 0,
                'errors': len(df[df['model_sentiment'] == 'ERROR']),
                'avg_weighted_disagreement': None,
                'exact_match_rate': 0,
                'off_by_one_rate': 0,
                'off_by_two_plus_rate': 0
            }
        
        valid_results['model_sentiment'] = valid_results['model_sentiment'].astype(float).astype(str)
        valid_results['human_sentiment'] = valid_results['human_sentiment'].astype(float).astype(str)
        
        from sklearn.metrics import classification_report
        report = classification_report(
            valid_results['human_sentiment'],
            valid_results['model_sentiment'],
            output_dict=True
        )
        
        # Calculate weighted disagreement metrics
        disagreement_stats = self.calculate_weighted_disagreement(df)
        
        metrics = {
            'model': model,
            'config_name': config.name,
            'accuracy': report['accuracy'],
            'weighted_f1': report['weighted avg']['f1-score'],
            'samples': len(valid_results),
            'errors': len(df[df['model_sentiment'] == 'ERROR'])
        }
        
        # Add disagreement metrics if available
        if disagreement_stats:
            metrics.update({
                'avg_weighted_disagreement': disagreement_stats['avg_weighted_disagreement'],
                'exact_match_rate': disagreement_stats['exact_matches'] / disagreement_stats['total_valid_samples'],
                'off_by_one_rate': disagreement_stats['off_by_one'] / disagreement_stats['total_valid_samples'],
                'off_by_two_plus_rate': disagreement_stats['off_by_two_plus'] / disagreement_stats['total_valid_samples']
            })
            
        return metrics

    def save_results(self):
        """Save detailed results and metrics for all configurations."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate and save metrics for all configurations
        metrics_df = pd.DataFrame()
        for config_key, results in self.all_results.items():
            df = pd.DataFrame(results['samples'])
            model, config_name = config_key.split('_', 1)
            metrics = self.calculate_configuration_metrics(df, model, results['config'])
            metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        
        # Save detailed results
        with open(f'prompt_evaluation_results_{timestamp}.json', 'w', encoding='utf-8') as f:
            serializable_results = {}
            for key, value in self.all_results.items():
                serializable_results[key] = {
                    'samples': value['samples'],
                    'config': {
                        'name': value['config'].name,
                        'instructions': value['config'].instructions,
                        'examples': value['config'].examples,
                        'temperature': value['config'].temperature
                    },
                    'performance_metrics': value['performance_metrics']
                }
            json.dump(serializable_results, f, cls=NumpyJSONEncoder, ensure_ascii=False, indent=2)
        
        # Save metrics summary
        metrics_df.to_csv(f'prompt_evaluation_metrics_{timestamp}.csv', index=False)
        
        # Print summary of best configurations by different metrics
        print("\nBest configurations by accuracy:")
        print(metrics_df.sort_values('accuracy', ascending=False).head().to_string())
        
        print("\nBest configurations by weighted disagreement (lower is better):")
        disagreement_df = metrics_df[metrics_df['avg_weighted_disagreement'].notna()]
        if not disagreement_df.empty:
            print(disagreement_df.sort_values('avg_weighted_disagreement').head().to_string())
            
        print("\nMetrics summary by configuration:")
        summary_cols = ['config_name', 'accuracy', 'weighted_f1', 'avg_weighted_disagreement', 
                       'exact_match_rate', 'off_by_one_rate', 'off_by_two_plus_rate', 'samples']
        print(metrics_df[summary_cols].to_string())
        
        return metrics_df

# Example usage:
def main():
    # Define different prompt configurations to test
    prompt_configs = [
        PromptConfig(
            name="medium_format",
            instructions="""
            Regeln:
            - Bewerte nur den markierten Rapper
            - Bei unklarem Kontext -> 3
            - Berücksichtige Jugendsprache
            """,
            examples="""
            Beispiele für die Bewertung (1-5):

            5: Extreme Begeisterung
            "<<X>> ist der beste Rapper", "<<X>> ist krass/brutal"

            4: Positive Bewertung
            "<<X>> feier ich", "<<X>> macht seit Jahren gute Sachen"

            3: Neutral/Unklar
            "<<X>> hat neues Album", "<<X>> wurde gesehen"

            2: Leicht Negativ
            "<<X>> ist nicht so meins", "früher war <<X>> besser"

            1: Stark Negativ
            "<<X>> ist müll", "<<X>> komplett peinlich"
            """,
            temperature=0.3
        ),

        PromptConfig(
            name = "original_one",
            instructions = """
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bei Unsicherheit -> 3
            - Wenn nicht der Rapper gemeint ist -> 3
            - Slang beachten (negative wörter können positiv gemeint sein)
            - Nur den markierten Rapper <<X>> bewerten
            - Auch indirekte Aussagen bewerten
            """,
            examples = """
            Beispiele:
            Sehr positiv (5):
            - "<<X>> ist der beste Rapper" -> 5
            - "<<X>> ist krass/brutal/hammer/zu gut" -> 5

            Eher positiv (4):
            - "<<X>> feier ich/macht stabile Musik" -> 4
            - "<<X>> macht seit Jahren gute Sachen" -> 4

            Neutral (3):
            - "<<X>> hat neues Album released" -> 3
            - "<<X>> wurde gesehen bei/in Y" -> 3
            - "Ich höre <<X>>, <<Y>>, <<Z>>" -> 3
            - "<<X>> erinnert an Y" -> 3

            Eher negativ (2):
            - "<<X>> ist nicht so meins" -> 2
            - "früher war <<X>> besser" -> 2

            Sehr negativ (1):
            - "<<X>> ist müll/ ein hurensohn" -> 1
            - "<<X>> sollte aufhören" -> 1
            """,
            temperature = 0.3
        ),

        PromptConfig(
            name="context_focused",
            instructions="""
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bewerte die Stärke der positiven/negativen Aussage:
              * 5 = sehr positiv/begeistert
              * 4 = positiv/gut
              * 3 = neutral/unklar
              * 2 = negativ/kritisch
              * 1 = sehr negativ/ablehnend
            - Beachte Rap-Kontext: "krass", "brutal" = meist positiv gemeint
            """,
            examples="""
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
            """,
            temperature=0.3
        ),


        PromptConfig(
            name="ultimate",
            instructions="""
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bei Unsicherheit -> 3
            - Wenn nicht der Rapper gemeint ist -> 3
            - Slang beachten: "krass", "brutal", "heftig" = meist positiv
            - Nur den markierten Rapper <<X>> bewerten
            - Auch indirekte Aussagen zählen ("sein Track ist whack" = negativ)
            - Bei Aufzählungen oder neutralen Erwähnungen -> 3
            """,
            examples="""
            Sehr positiv (5):
            "<<X>> ist der beste Rapper" -> 5
            "<<X>> ist krass/brutal/heftig" -> 5
            "<<X>> macht nur Klassiker" -> 5
            "<<X>> absoluter ehrenmove" -> 5

            Eher positiv (4):
            "<<X>> feier ich" -> 4
            "<<X>> macht gute Musik" -> 4
            "<<X>> hat skills" -> 4
            "respektiere was <<X>> macht" -> 4

            Neutral (3):
            "<<X>> hat neues Album" -> 3
            "<<X>> war auch dabei" -> 3
            "Track von <<X>>, Y und Z" -> 3
            "erinnert an <<X>>" -> 3

            Eher negativ (2):
            "<<X>> ist nicht so meins" -> 2
            "früher war <<X>> besser" -> 2
            "<<X>> wird überbewertet" -> 2
            "verstehe <<X>> nicht mehr" -> 2

            Sehr negativ (1):
            "<<X>> ist müll" -> 1
            "<<X>> sollte aufhören" -> 1
            "<<X>> komplett whack" -> 1
            "von <<X>> kriegt man Ohrenkrebs" -> 1
            """,
            temperature=0.3
        ),
                PromptConfig(
            name="medium_format02",
            instructions="""
            Regeln:
            - Bewerte nur den markierten Rapper
            - Bei unklarem Kontext -> 3
            - Berücksichtige Jugendsprache
            """,
            examples="""
            Beispiele für die Bewertung (1-5):

            5: Extreme Begeisterung
            "<<X>> ist der beste Rapper", "<<X>> ist krass/brutal"

            4: Positive Bewertung
            "<<X>> feier ich", "<<X>> macht seit Jahren gute Sachen"

            3: Neutral/Unklar
            "<<X>> hat neues Album", "<<X>> wurde gesehen"

            2: Leicht Negativ
            "<<X>> ist nicht so meins", "früher war <<X>> besser"

            1: Stark Negativ
            "<<X>> ist müll", "<<X>> komplett peinlich"
            """,
            temperature=0.2
        ),

        PromptConfig(
            name = "original_one02",
            instructions = """
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bei Unsicherheit -> 3
            - Wenn nicht der Rapper gemeint ist -> 3
            - Slang beachten (negative wörter können positiv gemeint sein)
            - Nur den markierten Rapper <<X>> bewerten
            - Auch indirekte Aussagen bewerten
            """,
            examples = """
            Beispiele:
            Sehr positiv (5):
            - "<<X>> ist der beste Rapper" -> 5
            - "<<X>> ist krass/brutal/hammer/zu gut" -> 5

            Eher positiv (4):
            - "<<X>> feier ich/macht stabile Musik" -> 4
            - "<<X>> macht seit Jahren gute Sachen" -> 4

            Neutral (3):
            - "<<X>> hat neues Album released" -> 3
            - "<<X>> wurde gesehen bei/in Y" -> 3
            - "Ich höre <<X>>, <<Y>>, <<Z>>" -> 3
            - "<<X>> erinnert an Y" -> 3

            Eher negativ (2):
            - "<<X>> ist nicht so meins" -> 2
            - "früher war <<X>> besser" -> 2

            Sehr negativ (1):
            - "<<X>> ist müll/ ein hurensohn" -> 1
            - "<<X>> sollte aufhören" -> 1
            """,
            temperature = 0.2
        ),

        PromptConfig(
            name="context_focused02",
            instructions="""
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bewerte die Stärke der positiven/negativen Aussage:
              * 5 = sehr positiv/begeistert
              * 4 = positiv/gut
              * 3 = neutral/unklar
              * 2 = negativ/kritisch
              * 1 = sehr negativ/ablehnend
            - Beachte Rap-Kontext: "krass", "brutal" = meist positiv gemeint
            """,
            examples="""
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
            """,
            temperature=0.2
        ),


        PromptConfig(
            name="ultimate02",
            instructions="""
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bei Unsicherheit -> 3
            - Wenn nicht der Rapper gemeint ist -> 3
            - Slang beachten: "krass", "brutal", "heftig" = meist positiv
            - Nur den markierten Rapper <<X>> bewerten
            - Auch indirekte Aussagen zählen ("sein Track ist whack" = negativ)
            - Bei Aufzählungen oder neutralen Erwähnungen -> 3
            """,
            examples="""
            Sehr positiv (5):
            "<<X>> ist der beste Rapper" -> 5
            "<<X>> ist krass/brutal/heftig" -> 5
            "<<X>> macht nur Klassiker" -> 5
            "<<X>> absoluter ehrenmove" -> 5

            Eher positiv (4):
            "<<X>> feier ich" -> 4
            "<<X>> macht gute Musik" -> 4
            "<<X>> hat skills" -> 4
            "respektiere was <<X>> macht" -> 4

            Neutral (3):
            "<<X>> hat neues Album" -> 3
            "<<X>> war auch dabei" -> 3
            "Track von <<X>>, Y und Z" -> 3
            "erinnert an <<X>>" -> 3

            Eher negativ (2):
            "<<X>> ist nicht so meins" -> 2
            "früher war <<X>> besser" -> 2
            "<<X>> wird überbewertet" -> 2
            "verstehe <<X>> nicht mehr" -> 2

            Sehr negativ (1):
            "<<X>> ist müll" -> 1
            "<<X>> sollte aufhören" -> 1
            "<<X>> komplett whack" -> 1
            "von <<X>> kriegt man Ohrenkrebs" -> 1
            """,
            temperature=0.2
        ),
                PromptConfig(
            name="medium_format04",
            instructions="""
            Regeln:
            - Bewerte nur den markierten Rapper
            - Bei unklarem Kontext -> 3
            - Berücksichtige Jugendsprache
            """,
            examples="""
            Beispiele für die Bewertung (1-5):

            5: Extreme Begeisterung
            "<<X>> ist der beste Rapper", "<<X>> ist krass/brutal"

            4: Positive Bewertung
            "<<X>> feier ich", "<<X>> macht seit Jahren gute Sachen"

            3: Neutral/Unklar
            "<<X>> hat neues Album", "<<X>> wurde gesehen"

            2: Leicht Negativ
            "<<X>> ist nicht so meins", "früher war <<X>> besser"

            1: Stark Negativ
            "<<X>> ist müll", "<<X>> komplett peinlich"
            """,
            temperature=0.4
        ),

        PromptConfig(
            name = "original_one04",
            instructions = """
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bei Unsicherheit -> 3
            - Wenn nicht der Rapper gemeint ist -> 3
            - Slang beachten (negative wörter können positiv gemeint sein)
            - Nur den markierten Rapper <<X>> bewerten
            - Auch indirekte Aussagen bewerten
            """,
            examples = """
            Beispiele:
            Sehr positiv (5):
            - "<<X>> ist der beste Rapper" -> 5
            - "<<X>> ist krass/brutal/hammer/zu gut" -> 5

            Eher positiv (4):
            - "<<X>> feier ich/macht stabile Musik" -> 4
            - "<<X>> macht seit Jahren gute Sachen" -> 4

            Neutral (3):
            - "<<X>> hat neues Album released" -> 3
            - "<<X>> wurde gesehen bei/in Y" -> 3
            - "Ich höre <<X>>, <<Y>>, <<Z>>" -> 3
            - "<<X>> erinnert an Y" -> 3

            Eher negativ (2):
            - "<<X>> ist nicht so meins" -> 2
            - "früher war <<X>> besser" -> 2

            Sehr negativ (1):
            - "<<X>> ist müll/ ein hurensohn" -> 1
            - "<<X>> sollte aufhören" -> 1
            """,
            temperature = 0.4
        ),

        PromptConfig(
            name="context_focused04",
            instructions="""
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bewerte die Stärke der positiven/negativen Aussage:
              * 5 = sehr positiv/begeistert
              * 4 = positiv/gut
              * 3 = neutral/unklar
              * 2 = negativ/kritisch
              * 1 = sehr negativ/ablehnend
            - Beachte Rap-Kontext: "krass", "brutal" = meist positiv gemeint
            """,
            examples="""
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
            """,
            temperature=0.4
        ),


        PromptConfig(
            name="ultimate04",
            instructions="""
            Regeln:
            - Antworte NUR mit 1, 2, 3, 4 oder 5
            - Bei Unsicherheit -> 3
            - Wenn nicht der Rapper gemeint ist -> 3
            - Slang beachten: "krass", "brutal", "heftig" = meist positiv
            - Nur den markierten Rapper <<X>> bewerten
            - Auch indirekte Aussagen zählen ("sein Track ist whack" = negativ)
            - Bei Aufzählungen oder neutralen Erwähnungen -> 3
            """,
            examples="""
            Sehr positiv (5):
            "<<X>> ist der beste Rapper" -> 5
            "<<X>> ist krass/brutal/heftig" -> 5
            "<<X>> macht nur Klassiker" -> 5
            "<<X>> absoluter ehrenmove" -> 5

            Eher positiv (4):
            "<<X>> feier ich" -> 4
            "<<X>> macht gute Musik" -> 4
            "<<X>> hat skills" -> 4
            "respektiere was <<X>> macht" -> 4

            Neutral (3):
            "<<X>> hat neues Album" -> 3
            "<<X>> war auch dabei" -> 3
            "Track von <<X>>, Y und Z" -> 3
            "erinnert an <<X>>" -> 3

            Eher negativ (2):
            "<<X>> ist nicht so meins" -> 2
            "früher war <<X>> besser" -> 2
            "<<X>> wird überbewertet" -> 2
            "verstehe <<X>> nicht mehr" -> 2

            Sehr negativ (1):
            "<<X>> ist müll" -> 1
            "<<X>> sollte aufhören" -> 1
            "<<X>> komplett whack" -> 1
            "von <<X>> kriegt man Ohrenkrebs" -> 1
            """,
            temperature=0.4
        ),


    ]
    
    models = [
        #'granite3.1-dense:2b', 
        "qwen2.5:3b"
        ]  # Add your models here
    evaluator = PromptEvaluator('test_set.json', models, prompt_configs)
    metrics_df = evaluator.evaluate_all_configurations(max_samples=200)
    evaluator.save_results()

if __name__ == "__main__":
    main()