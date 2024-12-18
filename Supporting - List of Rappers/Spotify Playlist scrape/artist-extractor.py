import csv
import os
from collections import Counter
from pathlib import Path

# Set up directory to script location
os.chdir(os.path.dirname(os.path.realpath(__file__)))

def extract_data_from_file(csv_file):
    """
    Extract both artist names and full song entries from a single CSV file.
    """
    artists = []
    songs = []
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 3:  # Ensure row has enough columns
                    # Store complete song entry
                    songs.append(row)
                    
                    # Process artists
                    artist_field = row[2]
                    track_artists = artist_field.split(',')
                    
                    for artist in track_artists:
                        artist = artist.replace('feat.', '').strip()
                        if artist:
                            artists.append(artist)
    except Exception as e:
        print(f"Warning: Error processing {csv_file}: {e}")
    
    return artists, songs

def process_all_csvs():
    """
    Process all CSV files in the current directory and subdirectories.
    Return combined artist list and song statistics.
    """
    csv_files = list(Path('.').rglob('*.csv'))
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in the current directory or subdirectories")
    
    all_artists = []
    all_songs = []
    for csv_file in csv_files:
        print(f"Processing: {csv_file}")
        artists, songs = extract_data_from_file(csv_file)
        all_artists.extend(artists)
        all_songs.extend(songs)
    
    # Process artists
    artist_counts = Counter(all_artists)
    unique_artists = sorted(set(all_artists))
    top_artists = sorted(artist_counts.items(), key=lambda x: (-x[1], x[0]))
    
    # Process songs (count by spotify ID to handle duplicates)
    song_counts = Counter(song[0] for song in all_songs)  # Count by Spotify ID
    
    # Get top 100 songs with full entry data
    song_lookup = {song[0]: song for song in all_songs}  # ID -> full entry
    top_songs = []
    for song_id, count in sorted(song_counts.items(), key=lambda x: (-x[1], x[0]))[:100]:
        if song_id in song_lookup:
            top_songs.append(song_lookup[song_id])
    
    return unique_artists, top_artists, top_songs, len(csv_files)

def save_results(unique_artists, top_artists, top_songs, num_files):
    """
    Save results to files and print statistics.
    """
    # Save unique artists to txt file
    with open('all_artists.txt', 'w', encoding='utf-8') as f:
        for artist in unique_artists:
            f.write(f"{artist}\n")
    
    # Save statistics to a separate file
    with open('artist_statistics.txt', 'w', encoding='utf-8') as f:
        f.write("Artist Statistics\n")
        f.write("================\n\n")
        f.write(f"Total CSV files processed: {num_files}\n")
        f.write(f"Total unique artists found: {len(unique_artists)}\n\n")
        
        f.write("Top 50 Most Featured Artists:\n")
        f.write("----------------------------\n")
        for artist, count in top_artists[:50]:
            f.write(f"{artist}: {count} appearances\n")
    
    # Save top 100 songs to CSV
    with open('top_100_songs.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        for song in top_songs:
            writer.writerow(song)
    
    # Print summary to console
    print("\nProcessing complete!")
    print(f"Processed {num_files} CSV files")
    print(f"Found {len(unique_artists)} unique artists")
    print("\nTop 10 Most Featured Artists:")
    for artist, count in top_artists[:10]:
        print(f"{artist}: {count} appearances")
    print("\nResults have been saved to:")
    print("- all_artists.txt (complete artist list)")
    print("- artist_statistics.txt (detailed statistics)")
    print("- top_100_songs.csv (most common songs)")

def main():
    try:
        unique_artists, top_artists, top_songs, num_files = process_all_csvs()
        save_results(unique_artists, top_artists, top_songs, num_files)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()