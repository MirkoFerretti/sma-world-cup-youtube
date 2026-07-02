# Heroes, Villains, and Underdogs: Narrative Communities in World Cup YouTube Discussions

This repository contains the initial setup for a university Social Media Analytics project on how YouTube users build narratives around World Cup matches, teams, players, controversies, rivalries, national pride, and underdog stories.

## Research Questions

1. Do World Cup YouTube discussions form communities around teams, matches, or shared narratives?
2. Which narratives dominate each detected community, and how are they expressed through entities, keywords, sentiment, and emotions?
3. Which users, videos, teams, players, or controversy-related topics bridge different narrative communities?

## Planned Data Source

- YouTube video metadata and comments collected with the YouTube Data API
- A manually curated list of selected World Cup videos stored in `metadata/selected_videos_template.csv`

## Planned Methods

- YouTube data collection and metadata tracking
- Comment cleaning and preparation
- User-video bipartite network construction
- User-user co-commenting network construction
- Network A: users with at least 2 distinct videos, edges with weight >= 1
- Network B: same user pool, edges with weight >= 2, isolated nodes removed
- Louvain community detection
- Network summary statistics
- Centrality measures: degree, weighted degree, PageRank or eigenvector-style importance, and betweenness
- VADER sentiment analysis
- Emotion analysis with NRC EmoLex or another simple lexicon-based approach
- TF-IDF keyword extraction by community
- Entity extraction with spaCy and optional custom football or player dictionaries
- Entity co-occurrence network analysis
- Final narrative labeling based on combined network and content evidence

## Methodological Notes

- Co-commenting represents shared participation in the same video discussion, not direct interaction between users.
- Louvain communities are algorithmic clusters and should not be treated as objectively real social groups.
- Sentiment and emotion tools can be noisy, especially with sarcasm, slang, emojis, and multilingual comments.
- The project prioritizes interpretability, reproducibility, and clear justification over unnecessary complexity.

## Repository Structure

```text
sma-world-cup-youtube/
├── AGENTS.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
├── data/
│   ├── raw/
│   ├── processed/
│   └── final/
├── metadata/
├── notebooks/
├── src/
├── outputs/
│   ├── figures/
│   ├── tables/
│   └── networks/
└── report/
```

## Setup

1. Create a virtual environment:

   ```powershell
   python -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and add your YouTube API key.

## YouTube API Key

Create a `.env` file in the project root with:

```env
YOUTUBE_API_KEY=your_api_key_here
```

The project loads this value from `src/config.py`.

## Important Warning

- Do not commit `.env` or API keys.
- Do not commit raw data collected from the API.
- Keep intermediate and final outputs reproducible from scripts or notebooks where possible.
