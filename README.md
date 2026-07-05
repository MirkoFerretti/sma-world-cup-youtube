# Heroes, Villains, and Underdogs: Narrative Communities in World Cup YouTube Discussions

This repository contains a university Social Media Analytics project on how YouTube users build narrative communities around the FIFA World Cup 2026.

## Project Status

The current YouTube dataset is frozen for this project.

Do not rerun:

- `python src/collect_youtube.py discover`
- `python src/collect_youtube.py filter-eligible`
- `python src/collect_youtube.py collect`

unless you intentionally want to restart the data-collection process. YouTube search results, metadata, and comments can change over time, so rerunning collection could produce a different corpus and would require rerunning preprocessing, network construction, community detection, content analysis, and bridge analysis.

The normal workflow from this point starts from the existing collected files and the analysis scripts below.

## Research Questions

1. Do World Cup 2026 YouTube discussions form participation communities around specific teams, matches, or broader narratives?
2. Which narratives dominate detected communities, and how are they expressed through entities, keywords, sentiment, and emotions?
3. Which users, videos, matches, teams, players, or controversy-related entities bridge different narrative communities?

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the content-analysis step on Windows, prefer the project venv command:

```powershell
.\.venv\Scripts\python src/analyze_content.py
```

Optional spaCy model for hybrid entity extraction:

```powershell
python -m pip install -U spacy
python -m spacy download en_core_web_sm
```

`NRCLex` is required for emotion analysis. If it is missing or incompatible, reinstall it with `.\.venv\Scripts\python -m pip install --upgrade --force-reinstall NRCLex`.
Entity normalization is used for aliases such as `Ronaldo` or `CR7`, but full external knowledge-base NED/NEL is not used.

Create `.env` in the project root:

```env
YOUTUBE_API_KEY=your_api_key_here
```

Do not commit `.env` or API keys.

## Simplified Pipeline

1. Data collection, already completed and now frozen:

```powershell
python src/collect_youtube.py discover --config config/youtube_queries.json --output metadata/video_sampling_pilot.csv
python src/collect_youtube.py filter-eligible --candidates metadata/video_sampling_pilot.csv --output metadata/video_sampling_pilot.csv
python src/collect_youtube.py collect --selected metadata/video_sampling_pilot.csv --max-comments 200
```

2. Preprocessing:

```powershell
python src/clean_comments.py
```

This preprocessing step creates `data/processed/comments_processed.csv` and `outputs/tables/preprocessing_summary.csv`.

3. Network construction:

```powershell
python src/build_networks.py
```

This step creates the user-video edge table, the user-user co-commenting edge table, the user-node table, `outputs/tables/network_summary.csv`, and the diagnostic figures in `outputs/figures/`.

4. Community detection:

```powershell
python src/detect_communities.py
```

This step creates the main weight >= 2 community assignments, community summaries, community-video profiles, `outputs/tables/community_detection_summary.csv`, and the PNG report figures in `outputs/figures/`.

5. Content analysis:

```powershell
.\.venv\Scripts\python src/analyze_content.py
```

This step creates `data/processed/community_content_summary.csv`, `outputs/tables/content_analysis_summary.csv`, and the four content-analysis PNG figures in `outputs/figures/`.

The main pipeline uses `vaderSentiment` for sentiment, `NRCLex` for the eight-emotion NRC-style analysis, dictionary-based entity normalization, and optional spaCy NER. Full NED/NEL is not used.

6. Bridge analysis:

```powershell
.\.venv\Scripts\python src/analyze_bridges.py
```

This step creates `data/processed/bridge_users.csv`, `data/processed/bridge_videos.csv`, `data/processed/bridge_entities.csv`, `outputs/tables/bridge_analysis_summary.csv`, `outputs/figures/bridge_videos.png`, and `outputs/figures/bridge_entities.png`. Bridge entities are scored from comment-level normalized entity matches using participation coefficient across all detected communities plus mention support.

Optional detail tables:

```powershell
.\.venv\Scripts\python src/analyze_content.py --write-detail-tables
```

This writes detailed comment, term, entity, and representative-comment inspection CSVs, but they are not default outputs.

## Output Files

- `metadata/video_sampling_pilot.csv`: compact video sampling file with both eligible and excluded videos
- `data/raw/youtube_comments_pilot.csv`: raw top-level comments
- `outputs/tables/data_collection_summary_pilot.csv`: compact key-value summary
- `data/processed/comments_processed.csv`: lightly cleaned processed comments for later analysis
- `outputs/tables/preprocessing_summary.csv`: compact preprocessing and preflight summary
- `data/processed/user_video_edges.csv`: anonymized user-video participation edges
- `data/processed/user_network_edges.csv`: weighted user-user co-commenting edges
- `data/processed/user_network_nodes.csv`: user participation profiles
- `outputs/tables/network_summary.csv`: overall network and threshold diagnostics
- `outputs/figures/user_participation_distribution.png`: repeat-user participation figure
- `outputs/figures/network_threshold_diagnostics.png`: threshold comparison figure
- `data/processed/user_communities.csv`: Louvain assignments for users in the main graph
- `data/processed/community_summary.csv`: one-row summaries for detected communities
- `data/processed/community_video_profiles.csv`: community-video participation profiles
- `outputs/tables/community_detection_summary.csv`: compact community detection summary
- `data/processed/community_content_summary.csv`: one-row content summaries for detected communities
- `outputs/tables/content_analysis_summary.csv`: compact content-analysis summary
- `data/processed/bridge_users.csv`: bridge-user centrality summary on the main co-commenting graph
- `data/processed/bridge_videos.csv`: cross-community participation summary for videos
- `data/processed/bridge_entities.csv`: cross-community bridge-entity summary
- `outputs/tables/bridge_analysis_summary.csv`: compact bridge-analysis summary
- `outputs/figures/community_size_distribution.png`: community size figure
- `outputs/figures/community_size_long_tail.png`: full long-tail community size diagnostic
- `outputs/figures/community_context_profile.png`: community context profile figure
- `outputs/figures/community_category_profile.png`: community category profile figure
- `outputs/figures/community_network_overview.png`: main co-commenting network overview
- `outputs/figures/community_highlight_C1.png`: highlight view for the largest community
- `outputs/figures/community_highlight_C2.png`: highlight view for the second-largest community
- `outputs/figures/community_highlight_C3.png`: highlight view for the third-largest community
- `outputs/figures/community_highlight_C4.png`: highlight view for the fourth-largest community
- `outputs/figures/community_highlight_C5.png`: highlight view for the fifth-largest community
- `outputs/figures/community_sentiment_profile.png`: stacked sentiment profile for the main communities
- `outputs/figures/community_emotion_profile.png`: stacked emotion profile for the main communities
- `outputs/figures/community_top_terms.png`: top TF-IDF terms for the main communities
- `outputs/figures/community_entity_profile.png`: entity heatmap for the main communities
- `outputs/figures/bridge_videos.png`: top bridge videos across communities
- `outputs/figures/bridge_entities.png`: top bridge entities across communities

Generated outputs can be regenerated and are overwritten by default.

## Method Note

Later co-commenting networks represent shared participation in the same video discussion space, not direct interaction between users.
