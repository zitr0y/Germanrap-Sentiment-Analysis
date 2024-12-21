import json
import random
from typing import List, Dict
import tkinter as tk
from tkinter import ttk, scrolledtext
from pathlib import Path
import re
import sys
import traceback
import os 
from tqdm import tqdm

os.chdir(os.path.dirname(os.path.realpath(__file__)))

class TestSetCreator:
    def __init__(
        self, 
        text_file_path: str, 
        rapper_aliases_path: str,
        output_path: str = 'test_set.json',
        samples_path: str = 'sampled_texts.json',  # New parameter for samples file
        seed: int = 42,
        target_samples: int = 200,
        batch_size: int = 500
    ):
        print("Initializing TestSetCreator...")
        
        # Store paths and configuration
        self.text_file_path = Path(text_file_path)
        self.rapper_aliases_path = Path(rapper_aliases_path)
        self.output_path = Path(output_path)
        self.samples_path = Path(samples_path)
        self.seed = seed
        self.target_samples = target_samples
        self.batch_size = batch_size
        
        # Check if files exist
        if not self.text_file_path.exists():
            raise FileNotFoundError(f"Text file not found: {self.text_file_path}")
        if not self.rapper_aliases_path.exists():
            raise FileNotFoundError(f"Rapper aliases file not found: {self.rapper_aliases_path}")
            
        print(f"Text file path: {self.text_file_path}")
        print(f"Rapper aliases path: {self.rapper_aliases_path}")
        
        # Set random seed for reproducibility
        random.seed(self.seed)
        
        try:
            print("Loading state...")
            self.load_state()
            
            print("Setting up GUI...")
            self.setup_gui()
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            traceback.print_exc()
            sys.exit(1)
    
    def load_rapper_aliases(self) -> Dict[str, str]:
        """Load rapper names and their aliases."""
        print("Loading rapper aliases...")
        try:
            with open(self.rapper_aliases_path, 'r', encoding='utf-8') as f:
                rapper_data = json.load(f)
            
            name_mapping = {}
            for canonical_name, aliases in rapper_data.items():
                name_mapping[canonical_name.lower()] = canonical_name
                for alias in aliases:
                    name_mapping[alias.lower()] = canonical_name
            
            print(f"Loaded {len(name_mapping)} rapper name variants")
            return name_mapping
            
        except Exception as e:
            print(f"Error loading rapper aliases: {str(e)}")
            traceback.print_exc()
            raise
    
    def prepare_rapper_patterns(self):
        """Create a compiled regex pattern for all rapper names."""
        variants = [(v.replace('_', ' '), v) for v in self.name_mapping.keys()]
        variants.sort(key=lambda x: len(x[0]), reverse=True)
        
        self.rapper_patterns = []
        for text_variant, orig_variant in variants:
            if text_variant:  # Skip empty strings
                pattern = re.compile(r'\b' + re.escape(text_variant) + r'\b')
                self.rapper_patterns.append((pattern, orig_variant))

    def find_rappers_in_text(self, text: str) -> List[tuple[str, str]]:
        """Find all rapper mentions in text and return tuple of (found_alias, canonical_name)."""
        text = text.lower()
        found_rappers = set()
        
        for pattern, orig_variant in self.rapper_patterns:
            if pattern.search(text):
                found_rappers.add((orig_variant, self.name_mapping[orig_variant]))
        
        return list(found_rappers)
    
    def get_samples(self) -> List[Dict]:
        """Get samples from the text file that contain rapper mentions."""
        print("Getting samples from text file...")
        try:
            self.name_mapping = self.load_rapper_aliases()
            self.prepare_rapper_patterns()
            samples = []
            processed_line_indices = set()
            
            # Read all lines from the file
            with open(self.text_file_path, 'r', encoding='utf-8') as f:
                all_lines = [line.strip() for line in f if line.strip()]
            
            print(f"Found {len(all_lines)} total lines in text file")
            
            # Keep processing random batches until we have enough samples
            while len(samples) < self.target_samples:
                # Get indices for unprocessed lines
                available_indices = list(set(range(len(all_lines))) - processed_line_indices)
                if not available_indices:
                    print("No more unprocessed lines available")
                    break
                
                # Sample a batch of indices
                batch_indices = random.sample(
                    available_indices,
                    min(self.batch_size, len(available_indices))
                )
                processed_line_indices.update(batch_indices)
                
                print(f"\nProcessing batch of {len(batch_indices)} lines...")
                print(f"Current samples: {len(samples)}, Target: {self.target_samples}")
                
                # Process the batch
                for idx in tqdm(batch_indices):
                    line = all_lines[idx]
                    rapper_mentions = self.find_rappers_in_text(line)
                    if rapper_mentions and len(samples) < self.target_samples:
                        for found_alias, canonical_name in rapper_mentions:
                            if len(samples) < self.target_samples:
                                samples.append({
                                    'text': line,
                                    'rapper_name': canonical_name,
                                    'found_alias': found_alias
                                })
                
                print(f"Found {len(samples)} samples so far")
                
                if len(processed_line_indices) == len(all_lines):
                    print("Processed all available lines")
                    break
            
            # Shuffle samples consistently
            random.shuffle(samples)
            
            print(f"\nFinished sampling:")
            print(f"- Processed {len(processed_line_indices)} lines")
            print(f"- Found {len(samples)} samples with rapper mentions")
            
            return samples
            
        except Exception as e:
            print(f"Error getting samples: {str(e)}")
            traceback.print_exc()
            raise
    
    def load_state(self):
        """Load existing annotations and state or initialize new ones."""
        print("Loading state...")
        try:
            # Try to load existing samples first
            if self.samples_path.exists():
                with open(self.samples_path, 'r', encoding='utf-8') as f:
                    saved_data = json.load(f)
                    if (saved_data.get('target_samples') == self.target_samples and 
                        saved_data.get('seed') == self.seed):
                        print("Loading existing samples...")
                        self.samples = saved_data['samples']
                    else:
                        print("Target samples or seed changed, resampling...")
                        self.samples = self.get_samples()
                        self.save_samples()
            else:
                print("No existing samples found, creating new ones...")
                self.samples = self.get_samples()
                self.save_samples()
            
            # Load annotations if they exist
            if Path(self.output_path).exists():
                with open(self.output_path, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
                print(f"Loaded {len(self.annotations)} existing annotations")
                
                # Find the next unannotated sample using text, rapper_name AND found_alias
                annotated_set = {
                    (a['text'], a['rapper_name'], a['found_alias']) 
                    for a in self.annotations
                }
                self.current_index = 0
                for i, sample in enumerate(self.samples):
                    if (sample['text'], sample['rapper_name'], sample['found_alias']) not in annotated_set:
                        self.current_index = i
                        break
            else:
                print("No existing annotations found, starting fresh")
                self.annotations = []
                self.current_index = 0
                
        except Exception as e:
            print(f"Error loading state: {str(e)}")
            traceback.print_exc()
            raise
            
    def save_samples(self):
        """Save the sampled texts to a separate file."""
        print(f"Saving {len(self.samples)} samples...")
        with open(self.samples_path, 'w', encoding='utf-8') as f:
            json.dump({
                'target_samples': self.target_samples,
                'seed': self.seed,
                'samples': self.samples
            }, f, ensure_ascii=False, indent=2)
    
    def setup_gui(self):
        """Setup the GUI interface."""
        print("Setting up GUI...")
        try:
            self.root = tk.Tk()
            self.root.title("Sentiment Annotation Tool")
            self.root.geometry("800x600")
            
            # Configure text highlighting
            text_font = ('TkDefaultFont', 10)
            self.root.option_add('*Text.Font', text_font)
            
            # Main frame with padding
            main_frame = ttk.Frame(self.root, padding="10")
            main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            main_frame.columnconfigure(0, weight=1)
            main_frame.rowconfigure(1, weight=1)
            
            # Progress info
            self.progress_var = tk.StringVar()
            ttk.Label(main_frame, textvariable=self.progress_var).grid(
                row=0, column=0, columnspan=7, pady=5, sticky='w'
            )
            
            # Scrollable text area
            text_frame = ttk.Frame(main_frame)
            text_frame.grid(row=1, column=0, columnspan=7, sticky=(tk.W, tk.E, tk.N, tk.S))
            text_frame.columnconfigure(0, weight=1)
            text_frame.rowconfigure(0, weight=1)
            
            self.text_widget = scrolledtext.ScrolledText(
                text_frame, 
                wrap=tk.WORD, 
                width=70, 
                height=20,
                font=('TkDefaultFont', 10)
            )
            self.text_widget.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
            self.text_widget.tag_configure('highlight', background='yellow', font=('TkDefaultFont', 10, 'bold'))
            self.text_widget.configure(state='disabled')
            
            # Rapper info
            self.rapper_var = tk.StringVar()
            ttk.Label(main_frame, textvariable=self.rapper_var, wraplength=750).grid(
                row=2, column=0, columnspan=7, pady=10, sticky='w'
            )
            
            # Button frame
            button_frame = ttk.Frame(main_frame)
            button_frame.grid(row=3, column=0, columnspan=7, pady=10)
            
            # Buttons
            for i, (label, description) in enumerate([
                ('1', 'sehr negativ'),
                ('2', 'eher negativ'),
                ('3', 'neutral'),
                ('4', 'eher positiv'),
                ('5', 'sehr positiv'),
                ('N/A', 'kein Bezug')
            ]):
                btn = ttk.Button(
                    button_frame, 
                    text=f'{label}\n{description}', 
                    command=lambda x=label: self.annotate(x),
                    width=15
                )
                btn.grid(row=0, column=i, padx=5)
            
            # Key bindings
            self.root.bind('1', lambda e: self.annotate('1'))
            self.root.bind('2', lambda e: self.annotate('2'))
            self.root.bind('3', lambda e: self.annotate('3'))
            self.root.bind('4', lambda e: self.annotate('4'))
            self.root.bind('5', lambda e: self.annotate('5'))
            self.root.bind('n', lambda e: self.annotate('N/A'))
            
            # Window close handler
            self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
            
            # Configure grid weights
            self.root.columnconfigure(0, weight=1)
            self.root.rowconfigure(0, weight=1)
            
            print("Loading first sample...")
            self.load_next_sample()
            
            print("Starting GUI mainloop...")
            self.root.mainloop()
            
        except Exception as e:
            print(f"Error setting up GUI: {str(e)}")
            traceback.print_exc()
            raise
    
    def load_next_sample(self):
        """Load the next unannotated sample."""
        if self.current_index >= len(self.samples):
            print("No more samples to annotate")
            self.save_annotations()
            self.root.quit()
            return
            
        sample = self.samples[self.current_index]
        
        # Update text widget
        self.text_widget.configure(state='normal')
        self.text_widget.delete(1.0, tk.END)
        
        # Insert text and highlight rapper mention
        text = sample['text']
        alias = sample['found_alias'].replace('_', ' ')
        
        # Find the position of the alias in the text (case insensitive)
        idx = text.lower().find(alias.lower())
        if idx != -1:
            # Insert text in parts and apply tag
            self.text_widget.insert(tk.END, text[:idx])
            self.text_widget.insert(tk.END, text[idx:idx+len(alias)], 'highlight')
            self.text_widget.insert(tk.END, text[idx+len(alias):])
        else:
            self.text_widget.insert(tk.END, text)
            
        self.text_widget.configure(state='disabled')
        
        # Update info
        total_annotated = len(self.annotations)
        self.rapper_var.set(f"Rapper: {sample['rapper_name']} (gefunden als: {sample['found_alias']})")
        self.progress_var.set(
            f"Fortschritt: {total_annotated}/{len(self.samples)} "
            f"({(total_annotated/len(self.samples)*100):.1f}%)"
        )
    
    def annotate(self, sentiment):
        """Save annotation and load next sample."""
        sample = self.samples[self.current_index]
        annotation = {
            'text': sample['text'],
            'rapper_name': sample['rapper_name'],
            'found_alias': sample['found_alias'],
            'human_sentiment': sentiment
        }
        self.annotations.append(annotation)
        self.save_annotations()
        self.current_index += 1
        self.load_next_sample()
    
    def save_annotations(self):
        """Save current annotations to file."""
        print(f"Saving {len(self.annotations)} annotations...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.annotations, f, ensure_ascii=False, indent=2)
    
    def on_closing(self):
        """Handle window closing."""
        print("Window closing, saving annotations...")
        self.save_annotations()
        self.root.destroy()

def main():
    try:
        print("Starting Test Set Creator...")
        creator = TestSetCreator(
            text_file_path='../Step 2.2 Create Bi-and Trigrams for Word2Vec/2_2-sentences_with_ngrams.txt',
            rapper_aliases_path='../Step 2.3 Train Word2Vec/rapper_aliases.json',
            output_path='test_set.json',
            seed=42,
            target_samples=200,
            batch_size=500
        )
    except Exception as e:
        print(f"Error in main: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()