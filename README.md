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

## Output Files

- `metadata/video_sampling_pilot.csv`: compact video sampling file with both eligible and excluded videos
- `data/raw/youtube_comments_pilot.csv`: raw top-level comments
- `outputs/tables/data_collection_summary_pilot.csv`: compact key-value summary

Generated outputs can be regenerated and are overwritten by default.

## Method Note

Later co-commenting networks represent shared participation in the same video discussion space, not direct interaction between users.