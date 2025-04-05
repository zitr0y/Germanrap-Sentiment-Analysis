# German Rap Reddit Analysis Project

This project scrapes data from German Rap subreddits, processes the text, trains a Word2Vec model, performs sentiment analysis on mentions of rappers using an LLM, and analyzes the results, including temporal trends.

## ✨ Features

* **Reddit Data Scraping:** Fetches posts and comments from specified subreddits using Pushshift and PRAW APIs.
* **Text Processing:** Cleans raw Reddit text, removing noise like URLs, markdown, and bot messages.
* **N-gram Generation:** Identifies and creates meaningful bigrams and trigrams (e.g., "kool\_savas", "k\_i\_z").
* **Word Embedding:** Trains a Word2Vec model on the processed Reddit corpus.
* **Rapper Alias Discovery:** Uses the Word2Vec model to find potential aliases or related terms for known rappers.
* **Sentiment Analysis:** Utilizes an LLM (via Ollama) to assign sentiment scores (1-5) to text snippets mentioning specific rappers.
* **Data Storage:** Stores sentiment results, including timestamps, in an SQLite database.
* **Evaluation Framework:** Includes tools for evaluating LLM performance against a manually annotated test set and comparing different models/prompts.
* **Temporal Analysis:** Links sentiment data with original timestamps for time-based analysis.
* **Visualization:** Generates interactive plots for word embeddings and sentiment analysis results.

## 🚀 Setup

1.  **Prerequisites:**
    * Python 3.x
    * Git

2.  **Clone the Repository:**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

3.  **Create a Virtual Environment:**
    ```bash
    python -m venv .venv
    # Activate the environment
    # Windows:
    .\.venv\Scripts\activate
    # macOS/Linux:
    source .venv/bin/activate
    ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Environment Variables:**
    * Create a `.env` file in the `Step 1 Reddit Scraper` directory.
    * Add the following required variables (obtain credentials from Reddit and Pushshift):
        ```dotenv
        # Reddit API Credentials (PRAW)
        CLIENT_ID=your_reddit_client_id
        CLIENT_SECRET=your_reddit_client_secret
        REFRESH_TOKEN=your_reddit_refresh_token
        USER_AGENT='YourAppDescription by /u/YourUsername'

        # Pushshift API Credentials
        PUSHSHIFT_ACCESS_TOKEN=your_pushshift_access_token

        # Scraper Configuration (Optional - Defaults shown)
        SUBREDDITS=germanrap # Comma-separated if multiple
        LIMIT=1000000
        SINCE=2010-01-01
        UNTIL=YYYY-MM-DD # Defaults to current date
        ```
    * You might need another `.env` file in the root or `Step 3 Sentiment Analysis` for the `DISCORD_WEBHOOK_URL` if you want crash notifications.

6.  **Ollama:**
    * Ensure you have Ollama installed and running.
    * Pull the required LLM model (e.g., `ollama pull qwen2.5:3b`). The model used for sentiment analysis is specified in `Step 3 Sentiment Analysis/sentiment-analysis.py`.

## ⚙️ Workflow / Usage

The project is structured in sequential steps. Run the main script within each step's directory in order.

1.  **Gather Rapper List (Supporting):**
    * Run scripts in `Supporting - List of Rappers/` (Spotify, Wikipedia) to generate an initial list of artists (e.g., `all_artists.txt`).

2.  **Step 1: Scrape Reddit Data:**
    * Navigate to `Step 1 Reddit Scraper/`.
    * Ensure `.env` is configured correctly.
    * Run: `python mainscript.py`
    * *Output:* JSON files containing post and comment data in `Step 1 Reddit Scraper/1-posts/`.

3.  **Step 2.1: Prepare Text:**
    * Navigate to `Step 2.1 Prepare Text for Word2Vec/`.
    * Run: `python reddit_text_extraction_for_word2vec.py`
    * *Output:* Cleaned sentences in `2_1-processed_sentences.txt`.

4.  **Step 2.2: Create N-grams:**
    * Navigate to `Step 2.2 Create Bi-and Trigrams for Word2Vec/`.
    * Run: `python creating_ngrams.py`
    * *Output:* Sentences with n-grams joined by underscores in `2_2-sentences_with_ngrams.txt`. Also creates `bigrams.txt` and `trigrams.txt`.

5.  **Step 2.3: Train Word2Vec & Analyze Aliases:**
    * Navigate to `Step 2.3 Train Word2Vec/`.
    * Run `python add_rappers_no_alias.py` to initialize `rapper_aliases.json` using the list from the Supporting step.
    * Run `python train_word2vec.py` to train the model.
    * *Output:* `word2vec_model.model`.
    * Run `python find_rappers.py` to interactively find and confirm potential aliases based on word similarity, updating `rapper_aliases.json`.
    * Use `create_interactive_view.py` or `word2vec-visualizer.py` to explore embeddings.

6.  **Step 3: Sentiment Analysis:**
    * Navigate to `Step 3 Sentiment Analysis/`.
    * **(Manual Step):** Run `python test-set-creator.py` to launch the GUI and manually annotate samples for evaluation. This creates `test_set.json`.
    * Run `python sentiment-analysis.py` to perform sentiment analysis using the configured LLM (requires Ollama running).
    * *Output:* Results stored in `rapper_sentiments.db`. Progress saved in `sentiment_analysis_progress.json`.
    * Use `evaluate_BERT_baseline.py` or `llm_evaluator.py` / `prompt_evaluator.py` to evaluate model performance against `test_set.json`.
    * Run `python clean-sentiment-db.py` to clean `ERROR` or `NO_SENTIMENT` entries if needed (converts `NO_SENTIMENT` to `3`).
    * Run `python analyser.py` to generate analysis reports and visualizations from the database results (saved to `sentiment_analysis_results/`).

7.  **Step 3.1: Add Timestamps to Database:**
    * This step refines the timestamp association *after* n-grams have been created.
    * Navigate to `Step 3.1 Tack together time data with database/`.
    * Run `python reddit_text_extraction_for_word2vec.py` (extracts sentences *with* timestamps).
        * *Output:* `2_1-processed_sentences_with_time.txt`.
    * Run `python ngrams-with-timestamps-txt.py` (applies n-grams, saves mapping).
        * *Outputs:* `2_2-sentences_with_ngrams.txt` (overwritten/same as Step 2.2 output), `sentence_timestamps_mapping.txt`.
    * Run `python fix-mapping.py` (corrects mapping using the final n-gram sentences).
        * *Output:* `corrected_timestamps_mapping.txt`.
    * Run `python update-db-txt.py` (updates the database using the corrected mapping).
        * *Output:* Adds/updates the `original_timestamp` column in `rapper_sentiments.db`.

8.  **Final Analysis:**
    * Navigate back to `Step 3 Sentiment Analysis/`.
    * Run `python analyser.py` again to generate reports incorporating the timestamp data.

## 🛠️ Key Scripts Overview

* **Step 1/mainscript.py:** Orchestrates Reddit scraping and token management.
* **Step 2.1/reddit\_text\_extraction\_for\_word2vec.py:** Cleans JSON data and extracts sentences.
* **Step 2.2/creating\_ngrams.py:** Identifies and applies bigrams/trigrams.
* **Step 2.3/train\_word2vec.py:** Trains the Word2Vec model.
* **Step 2.3/find\_rappers.py:** Analyzes rapper similarity and helps build the alias list.
* **Step 3/test-set-creator.py:** GUI tool for manual sentiment annotation.
* **Step 3/sentiment-analysis.py:** Performs LLM-based sentiment analysis and saves to DB.
* **Step 3/evaluate\_BERT\_baseline.py:** Evaluates model performance against the test set.
* **Step 3.1/update-db-txt.py:** Adds accurate timestamps to the sentiment database.
* **Step 3/analyser.py:** Generates final reports and visualizations from the sentiment database.

## 🔧 Configuration

* **API Keys & Scraper Settings:** Configure in `Step 1 Reddit Scraper/.env`.
* **LLM Model:** The model used for sentiment analysis is set within `Step 3 Sentiment Analysis/sentiment-analysis.py` (currently `qwen2.5:3b`). You may need to adjust this based on available models in Ollama.
* **Rapper List:** The initial list is generated by scripts in `Supporting - List of Rappers/`. The final alias mapping is managed in `Step 2.3 Train Word2Vec/rapper_aliases.json`.

## 📦 Dependencies

All required Python packages are listed in `requirements.txt`.

## 📄 .gitignore

Note that generated data files (JSONs in `1-posts/`, `.txt`, `.csv`, `.model`, `.db`, `.html`, `.png`), log files, and environment files (`.env`) are ignored by Git.

---

Happy Analyzing!
