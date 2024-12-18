import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def scrape_german_hiphop_wiki():
    """
    Scrapes German hip-hop artists from Wikipedia
    Returns two dataframes: one for solo artists and one for groups
    """
    url = "https://de.wikipedia.org/wiki/Liste_von_Hip-Hop-Musikern_Deutschlands"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Initialize lists to store data
    solo_artists = []
    groups = []
    
    # Find all tables in the page
    tables = soup.find_all('table', {'class': 'wikitable'})
    
    # First table is solo artists
    if tables:
        solo_table = tables[0]
        rows = solo_table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 5:  # Ensure row has enough columns
                artist_name = cols[1].get_text(strip=True)
                memberships = cols[2].get_text(strip=True)
                group_membership = cols[3].get_text(strip=True)
                birth_date = cols[4].get_text(strip=True)
                
                solo_artists.append({
                    'artist_name': artist_name,
                    'memberships': memberships,
                    'group_membership': group_membership,
                    'birth_date': birth_date
                })
    
    # Second table is groups
    if len(tables) > 1:
        group_table = tables[1]
        rows = group_table.find_all('tr')[1:]  # Skip header row
        
        for row in rows:
            cols = row.find_all(['td', 'th'])
            if len(cols) >= 4:  # Ensure row has enough columns
                group_name = cols[1].get_text(strip=True)
                members = cols[2].get_text(strip=True)
                former_members = cols[3].get_text(strip=True)
                
                groups.append({
                    'group_name': group_name,
                    'members': members,
                    'former_members': former_members
                })
    
    # Convert to DataFrames
    solo_df = pd.DataFrame(solo_artists)
    groups_df = pd.DataFrame(groups)
    
    # Save to CSV files
    solo_df.to_csv('rappers.csv', index=False, encoding='utf-8')
    groups_df.to_csv('groups.csv', index=False, encoding='utf-8')
    
    return solo_df, groups_df

# Function to extract just the names
def extract_artist_names(solo_df, groups_df):
    """
    Extracts just the artist/group names and saves them to a text file
    """
    all_names = []
    
    # Add solo artists
    all_names.extend(solo_df['artist_name'].tolist())
    
    # Add groups
    all_names.extend(groups_df['group_name'].tolist())
    
    # Clean the names
    all_names = [name for name in all_names if name]  # Remove empty names
    all_names = sorted(set(all_names))  # Remove duplicates and sort
    
    # Save to file
    with open('german_hiphop_artists.txt', 'w', encoding='utf-8') as f:
        for name in all_names:
            f.write(f"{name}\n")
    
    return all_names

# Run the scraper
solo_df, groups_df = scrape_german_hiphop_wiki()

# Extract just the names
artist_names = extract_artist_names(solo_df, groups_df)

print(f"Found {len(solo_df)} solo artists and {len(groups_df)} groups")
print(f"Total unique names: {len(artist_names)}")