import json
import re
from typing import Dict, List
import os

os.chdir(os.path.dirname(os.path.realpath(__file__)))

def preprocess_name(name: str) -> str:
    """
    Preprocess name to match the format in the model
    
    Args:
        name: Raw name string
        
    Returns:
        Preprocessed name string
    """
    # Convert to lowercase
    name = name.lower().strip()
    
    # Replace umlauts
    umlaut_mapping = {
        'ä': 'ae',
        'ö': 'oe',
        'ü': 'ue',
        'ß': 'ss'
    }
    for umlaut, replacement in umlaut_mapping.items():
        name = name.replace(umlaut, replacement)
    
    # Replace spaces with underscores and remove special characters
    name = re.sub(r'\s+', '_', name)
    
    return name

def load_aliases(aliases_path: str) -> Dict[str, List[str]]:
    """Load existing aliases, create file if doesn't exist"""
    try:
        with open(aliases_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        with open(aliases_path, 'w', encoding='utf-8') as f:
            json.dump({}, f, indent=2)
        return {}

def add_base_rappers(rapper_list_path: str, aliases_path: str) -> tuple:
    """
    Add all rappers from the list to the aliases JSON as base entries
    
    Args:
        rapper_list_path: Path to text file containing rapper names
        aliases_path: Path to JSON file storing aliases
    
    Returns:
        tuple: (number of new entries, number of existing entries)
    """
    # Load existing aliases
    aliases = load_aliases(aliases_path)
    
    # Keep track of statistics
    new_entries = 0
    existing_entries = 0
    
    # Load and process rapper names
    with open(rapper_list_path, 'r', encoding='utf-8') as f:
        rappers = [line.strip() for line in f if line.strip()]
    
    for rapper in rappers:
        original_name = rapper
        processed_name = preprocess_name(rapper)
        
        # Skip if rapper already exists
        if processed_name in aliases:
            print(f"Skipping existing entry: {original_name} -> {processed_name}")
            existing_entries += 1
            continue
        
        # Add new rapper with empty aliases list
        aliases[processed_name] = []
        new_entries += 1
        print(f"Added new entry: {original_name} -> {processed_name}")
    
    # Save updated aliases
    with open(aliases_path, 'w', encoding='utf-8') as f:
        json.dump(aliases, f, indent=2, ensure_ascii=False)
    
    return new_entries, existing_entries

if __name__ == "__main__":
    # Define paths
    rapper_list_path = '../Supporting - List of Rappers/Spotify Playlist scrape/all_artists.txt'
    aliases_path = 'rapper_aliases.json'
    
    # Add base rappers
    new_entries, existing_entries = add_base_rappers(rapper_list_path, aliases_path)
    
    # Print summary
    print("\nSummary:")
    print(f"New entries added: {new_entries}")
    print(f"Existing entries skipped: {existing_entries}")
    print(f"Total rappers processed: {new_entries + existing_entries}")
