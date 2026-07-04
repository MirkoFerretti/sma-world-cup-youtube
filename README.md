# Heroes, Villains, and Underdogs: Narrative Communities in World Cup YouTube Discussions

This repository contains a university Social Media Analytics project on how YouTube users build narrative communities around the FIFA World Cup 2026.

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

Create `.env` in the project root:

```env
YOUTUBE_API_KEY=your_api_key_here
```

Do not commit `.env` or API keys.

## Simplified Pipeline

1. Discover candidate videos:

```powershell
python src/collect_youtube.py discover --config config/youtube_queries.json --output metadata/video_sampling_pilot.csv
```

2. Apply automatic eligibility filtering:

```powershell
python src/collect_youtube.py filter-eligible --candidates metadata/video_sampling_pilot.csv --output metadata/video_sampling_pilot.csv
```

3. Collect top-level comments from eligible videos:

```powershell
python src/collect_youtube.py collect --selected metadata/video_sampling_pilot.csv --max-comments 200
```

4. Run preprocessing:

```powershell
python src/clean_comments.py
```

This preprocessing step creates `data/processed/comments_processed.csv` and `outputs/tables/preprocessing_summary.csv`.

5. Build participation networks and diagnostics:

```powershell
python src/build_networks.py
```

This step creates the user-video edge table, the user-user co-commenting edge table, the user-node table, `outputs/tables/network_summary.csv`, and the diagnostic figures in `outputs/figures/`.

6. Detect participation communities:

```powershell
python src/detect_communities.py
```

This step creates the main weight >= 2 community assignments, community summaries, community-video profiles, `outputs/tables/community_detection_summary.csv`, and the PNG report figures in `outputs/figures/`.

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

Generated outputs can be regenerated and are overwritten by default.

## Method Note

Later co-commenting networks represent shared participation in the same video discussion space, not direct interaction between users.
