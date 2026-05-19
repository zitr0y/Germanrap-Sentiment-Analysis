# German Rap Reddit Sentiment Analysis

Scrapes German rap subreddits, trains a Word2Vec model to find rapper aliases, then runs sentiment analysis on mentions of rappers using a local LLM via Ollama. Results land in SQLite with timestamps for temporal analysis.

This was a uni project, so expect some rough edges.

## Setup

Python 3.x, then:

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Create a `.env` file in `Step 1 Reddit Scraper/` with your Reddit and Pushshift credentials:

```dotenv
CLIENT_ID=your_reddit_client_id
CLIENT_SECRET=your_reddit_client_secret
REFRESH_TOKEN=your_reddit_refresh_token
USER_AGENT='YourAppDescription by /u/YourUsername'
PUSHSHIFT_ACCESS_TOKEN=your_pushshift_access_token

SUBREDDITS=germanrap
LIMIT=1000000
SINCE=2010-01-01
UNTIL=YYYY-MM-DD
```

A `DISCORD_WEBHOOK_URL` can be added if you want crash notifications.

Install [Ollama](https://ollama.com/) and pull whatever model you want to use for sentiment scoring. The default is `qwen2.5:3b`, set in `Step 3 Sentiment Analysis/sentiment-analysis.py`.

## Workflow

The directories are numbered, run them in order.

**Supporting - List of Rappers/** scripts that pull artist names from Spotify and Wikipedia into `all_artists.txt`.

**Step 1 Reddit Scraper/**`python mainscript.py` dumps posts and comments as JSON into `1-posts/`.

**Step 2.1 Prepare Text for Word2Vec/** `reddit_text_extraction_for_word2vec.py` cleans the JSON and writes sentences to `2_1-processed_sentences.txt`.

**Step 2.2 Create Bi-and Trigrams for Word2Vec/** `creating_ngrams.py` joins common bigrams/trigrams with underscores (e.g. `kool_savas`) and writes `2_2-sentences_with_ngrams.txt`.

**Step 2.3 Train Word2Vec/** run `add_rappers_no_alias.py` to seed `rapper_aliases.json`, then `train_word2vec.py`. Use `find_rappers.py` to interactively confirm aliases based on word similarity. `create_interactive_view.py` and `word2vec-visualizer.py` are for exploring the embeddings.

**Step 3 Sentiment Analysis/**
- `test-set-creator.py` opens a GUI for hand-annotating a test set (saved as `test_set.json`).
- `sentiment-analysis.py` runs the LLM over the corpus and writes results to `rapper_sentiments.db`.
- `evaluate_BERT_baseline.py`, `llm_evaluator.py`, and `prompt_evaluator.py` compare models and prompts against the test set.
- `clean-sentiment-db.py` drops `ERROR` rows and rewrites `NO_SENTIMENT` to a neutral 3.
- `analyser.py` generates the final reports and plots in `sentiment_analysis_results/`.

**Step 3.1 Tack together time data with database/** this glues original post timestamps back onto the DB rows after the n-gram step. Run in order: `reddit_text_extraction_for_word2vec.py`, `ngrams-with-timestamps-txt.py`, `fix-mapping.py`, `update-db-txt.py`. The last one adds the `original_timestamp` column.

After that, re-run `analyser.py` to get the time-aware reports.

## Notes

Generated files (JSON dumps, `.txt`, `.csv`, `.model`, `.db`, plots, logs, `.env`) are gitignored.

The rapper alias mapping lives in `Step 2.3 Train Word2Vec/rapper_aliases.json` if you want to edit it by hand.
