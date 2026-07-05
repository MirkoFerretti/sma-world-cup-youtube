"""Simplified YouTube pilot discovery, filtering, and collection CLI."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pandas as pd

from config import RAW_DATA_DIR, ROOT_DIR, YOUTUBE_API_KEY


DEFAULT_DISCOVERY_OUTPUT = ROOT_DIR / "metadata" / "video_sampling_pilot.csv"
DEFAULT_SAMPLING_OUTPUT = ROOT_DIR / "metadata" / "video_sampling_pilot.csv"
DEFAULT_COMMENTS_OUTPUT = RAW_DATA_DIR / "youtube_comments_pilot.csv"
DEFAULT_SUMMARY_OUTPUT = ROOT_DIR / "outputs" / "tables" / "data_collection_summary_pilot.csv"
DEFAULT_ELIGIBLE_SPLIT_OUTPUT = ROOT_DIR / "metadata" / "eligible_videos_pilot.csv"
DEFAULT_EXCLUDED_SPLIT_OUTPUT = ROOT_DIR / "metadata" / "excluded_videos_pilot.csv"

DEFAULT_VIDEO_PARTS = "snippet,statistics,contentDetails,status"
SEARCH_CATEGORY_SEPARATOR = " | "
SELECTED_TRUE_VALUES = {"1", "true", "yes", "y"}
ENGLISH_RELEVANT_TITLE_WORDS = {
    "analysis",
    "controversy",
    "cup",
    "draw",
    "fan",
    "fans",
    "football",
    "group",
    "highlight",
    "highlights",
    "legacy",
    "match",
    "player",
    "players",
    "preview",
    "reaction",
    "reactions",
    "referee",
    "roster",
    "squad",
    "star",
    "tactical",
    "team",
    "tournament",
    "var",
    "watchalong",
    "world",
}
GAME_OR_BETTING_TERMS = {
    "betting",
    "career mode",
    "ea fc",
    "efootball",
    "fantasy",
    "gaming",
    "odds",
    "pack opening",
    "simulation",
    "simulator",
}
VIDEO_CONTEXT_PATTERNS = {
    "var_referee_controversy": {"controversy", "referee", "robbed", "var"},
    "group_draw_roster": {"draw reaction", "group draw", "groups revealed", "roster", "squad"},
    "watchalong_reaction": {"live watchalong", "watch along", "watch party", "watchalong"},
    "star_player_legacy": {"goat", "legacy", "messi", "mbappe", "player to watch", "players to watch", "ronaldo", "star player"},
    "tactical_analysis": {"analysis", "breakdown", "formation", "lineup", "manager", "strategy", "tactical"},
    "upset_underdog": {"comeback", "shock", "shocking", "surprise", "underdog", "upset"},
    "post_match_reaction": {"fan reaction", "live reaction", "post match reaction", "reaction", "reactions"},
    "match_highlights": {"goal", "goals", "highlight", "highlights", "penalty shootout", "post game", "post-match"},
    "preview_background": {"everything to know", "guide", "match preview", "players to watch", "preview", "storylines", "tournament preview"},
    "rivalry_national_pride": {"derby", "national pride", "rivalry"},
}
NON_ENGLISH_PREFIXES = {
    "ar",
    "de",
    "es",
    "fa",
    "fr",
    "hi",
    "id",
    "it",
    "ja",
    "ko",
    "nl",
    "pl",
    "pt",
    "ru",
    "tr",
    "ur",
    "vi",
    "zh",
}

DISCOVERY_COLUMNS = [
    "video_id",
    "url",
    "title",
    "channel_title",
    "published_at",
    "duration_seconds",
    "view_count",
    "like_count",
    "comment_count",
    "default_language",
    "default_audio_language",
    "search_categories",
    "primary_category",
    "video_context_type",
    "eligibility_status",
    "exclusion_reason",
    "comments_collected",
    "collection_error",
    "collection_date",
]
SAMPLING_COLUMNS = DISCOVERY_COLUMNS
COMMENT_COLUMNS = [
    "comment_id",
    "video_id",
    "author_channel_id",
    "author_display_name",
    "comment_text",
    "published_at",
    "updated_at",
    "like_count",
    "reply_count",
    "collection_date",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simplified YouTube pilot collection CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="Discover candidate videos from predefined queries.")
    discover_parser.add_argument("--config", default="config/youtube_queries.json", help="Path to the query configuration JSON.")
    discover_parser.add_argument("--output", default=str(DEFAULT_DISCOVERY_OUTPUT.relative_to(ROOT_DIR)), help="Candidate CSV output path.")
    add_overwrite_arguments(discover_parser)

    filter_parser = subparsers.add_parser("filter-eligible", help="Filter discovered candidates into one compact sampling file.")
    filter_parser.add_argument("--config", default="config/youtube_queries.json", help="Path to the query configuration JSON.")
    filter_parser.add_argument("--candidates", default=str(DEFAULT_DISCOVERY_OUTPUT.relative_to(ROOT_DIR)), help="Candidate CSV path.")
    filter_parser.add_argument("--output", default=str(DEFAULT_SAMPLING_OUTPUT.relative_to(ROOT_DIR)), help="Compact sampling CSV output path.")
    filter_parser.add_argument("--write-audit-splits", action="store_true", help="Also write separate eligible and excluded CSV files.")
    add_overwrite_arguments(filter_parser)

    collect_parser = subparsers.add_parser("collect", help="Collect capped top-level comments for eligible videos.")
    collect_parser.add_argument("--config", default="config/youtube_queries.json", help="Path to the query configuration JSON.")
    collect_parser.add_argument("--selected", "--input", dest="selected", default=str(DEFAULT_SAMPLING_OUTPUT.relative_to(ROOT_DIR)), help="Sampling CSV path.")
    collect_parser.add_argument("--max-comments", type=int, default=None, help="Override the maximum number of comments per video.")
    collect_parser.add_argument("--comments-output", default=str(DEFAULT_COMMENTS_OUTPUT.relative_to(ROOT_DIR)), help="Raw comments CSV output path.")
    collect_parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_OUTPUT.relative_to(ROOT_DIR)), help="Collection summary CSV output path.")
    add_overwrite_arguments(collect_parser)

    return parser.parse_args()


def add_overwrite_arguments(parser: argparse.ArgumentParser) -> None:
    """Add overwrite flags with overwrite enabled by default."""
    parser.set_defaults(overwrite=True)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite generated outputs. This is the default behavior.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Stop instead of overwriting existing generated outputs.")


def resolve_path(path_text: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def display_path(path: Path) -> str:
    """Render a path relative to the repository root when possible."""
    if path.is_relative_to(ROOT_DIR):
        return str(path.relative_to(ROOT_DIR))
    return str(path)


def print_status(message: str) -> None:
    """Print a simple terminal status message."""
    print(f"[collect_youtube] {message}")


def utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_output_parent(path: Path) -> None:
    """Create the parent directory for an output path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def ensure_can_write(path: Path, overwrite: bool) -> None:
    """Check whether a file can be written."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {display_path(path)}")


def write_csv_atomic(dataframe: pd.DataFrame, output_path: Path, overwrite: bool, dedupe_column: str | None = None) -> None:
    """Write a CSV atomically and overwrite by default."""
    ensure_output_parent(output_path)
    ensure_can_write(output_path, overwrite)

    clean_df = remove_duplicate_suffix_columns(dataframe.copy())
    if dedupe_column and dedupe_column in clean_df.columns:
        clean_df = clean_df.drop_duplicates(subset=[dedupe_column], keep="first")

    temp_path = output_path.parent / f".{output_path.name}.tmp"
    if temp_path.exists():
        temp_path.unlink()

    if output_path.exists():
        print_status(f"Overwriting existing file: {display_path(output_path)}")
    else:
        print_status(f"Writing new file: {display_path(output_path)}")

    clean_df.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def remove_duplicate_suffix_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove accidental duplicate columns ending in .1, .2, and so on."""
    keep_indices: list[int] = []
    output_columns: list[str] = []

    for index, column in enumerate(dataframe.columns):
        match = re.fullmatch(r"(.+)\.(\d+)", str(column))
        normalized = match.group(1) if match else str(column)
        if normalized in output_columns:
            continue
        keep_indices.append(index)
        output_columns.append(normalized)

    cleaned = dataframe.iloc[:, keep_indices].copy()
    cleaned.columns = output_columns
    return cleaned


def load_clean_csv(path: Path) -> pd.DataFrame:
    """Load a CSV and remove duplicate suffix columns."""
    dataframe = pd.read_csv(path)
    cleaned = remove_duplicate_suffix_columns(dataframe)
    if len(cleaned.columns) != len(dataframe.columns):
        print_status(f"Removed accidental duplicate columns from {display_path(path)}.")
    return cleaned


def load_query_config(config_path: Path) -> dict[str, Any]:
    """Load and validate the query configuration JSON."""
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    categories = config.get("categories")
    if not isinstance(categories, dict) or not categories:
        raise ValueError("Config must contain a non-empty 'categories' object.")
    return config


def build_youtube_client(api_key: str):
    """Create an authenticated YouTube client."""
    if not api_key:
        raise ValueError("YOUTUBE_API_KEY is missing. Add it to the project .env file.")
    try:
        from googleapiclient.discovery import build
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("google-api-python-client is not installed. Install project requirements first.") from error
    return build("youtube", "v3", developerKey=api_key)


def search_videos_for_query(youtube, query: str, max_results: int, order: str) -> list[dict[str, Any]]:
    """Search YouTube for one query."""
    items: list[dict[str, Any]] = []
    next_page_token: str | None = None

    while len(items) < max_results:
        batch_size = min(50, max_results - len(items))
        request_kwargs: dict[str, Any] = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "order": order,
            "maxResults": batch_size,
        }
        if next_page_token:
            request_kwargs["pageToken"] = next_page_token
        response = youtube.search().list(**request_kwargs).execute()
        items.extend(response.get("items", []))
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break

    return items


def chunked(items: list[str], chunk_size: int) -> list[list[str]]:
    """Split a list into fixed-size batches."""
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def safe_int(value: Any) -> int | None:
    """Convert a value to int when possible."""
    if value is None or value == "" or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def safe_text(value: Any) -> str:
    """Convert missing values to an empty string."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def parse_youtube_duration(duration_text: str | None) -> int | None:
    """Parse a YouTube ISO 8601 duration string into seconds."""
    if not duration_text:
        return None

    match = re.fullmatch(r"PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?", str(duration_text))
    if not match:
        return None

    hours = int(match.group("hours") or 0)
    minutes = int(match.group("minutes") or 0)
    seconds = int(match.group("seconds") or 0)
    return hours * 3600 + minutes * 60 + seconds


def fetch_video_metadata(youtube, video_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Fetch detailed metadata for a list of video IDs."""
    metadata: dict[str, dict[str, Any]] = {}
    for batch in chunked(video_ids, 50):
        response = youtube.videos().list(part=DEFAULT_VIDEO_PARTS, id=",".join(batch)).execute()
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})
            content_details = item.get("contentDetails", {})
            status = item.get("status", {})
            duration_seconds = parse_youtube_duration(content_details.get("duration"))
            metadata[item["id"]] = {
                "video_id": item["id"],
                "url": f"https://www.youtube.com/watch?v={item['id']}",
                "title": snippet.get("title"),
                "channel_title": snippet.get("channelTitle"),
                "published_at": snippet.get("publishedAt"),
                "default_language": snippet.get("defaultLanguage"),
                "default_audio_language": snippet.get("defaultAudioLanguage"),
                "live_broadcast_content": snippet.get("liveBroadcastContent"),
                "privacy_status": status.get("privacyStatus"),
                "duration_seconds": duration_seconds,
                "view_count": safe_int(statistics.get("viewCount")),
                "like_count": safe_int(statistics.get("likeCount")),
                "comment_count": safe_int(statistics.get("commentCount")),
                "comment_count_visible": "commentCount" in statistics,
                "description": snippet.get("description"),
            }
    return metadata


def split_delimited_values(value: Any, separator: str = SEARCH_CATEGORY_SEPARATOR) -> list[str]:
    """Split a separator-delimited string into trimmed values."""
    if value is None or pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(separator) if part.strip()]


def normalize_text(value: Any) -> str:
    """Normalize text for simple heuristics."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def ascii_letter_ratio(text: str) -> float:
    """Estimate how much alphabetic text is ASCII."""
    letters = re.findall(r"[A-Za-zÀ-ÿ]", text)
    if not letters:
        return 0.0
    ascii_letters = re.findall(r"[A-Za-z]", text)
    return len(ascii_letters) / len(letters)


def normalize_language_code(value: Any) -> str:
    """Normalize a language code like en-US to en-us."""
    return normalize_text(value)


def is_explicit_english(language_code: str) -> bool:
    """Check if a normalized language code is explicitly English."""
    return language_code == "en" or language_code.startswith("en-")


def is_explicit_non_english(language_code: str) -> bool:
    """Check if a normalized language code is explicitly non-English."""
    if not language_code:
        return False
    prefix = language_code.split("-", 1)[0]
    return prefix in NON_ENGLISH_PREFIXES and not is_explicit_english(language_code)


def title_looks_english(title: str, channel_title: str) -> bool:
    """Use a simple fallback heuristic when YouTube language fields are missing."""
    text = f"{title} {channel_title}".strip()
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    if len(tokens) < 2:
        return False
    english_hits = len(set(tokens).intersection(ENGLISH_RELEVANT_TITLE_WORDS))
    return ascii_letter_ratio(text) >= 0.8 and english_hits >= 1


def detect_language_reason(row: pd.Series) -> tuple[bool, str]:
    """Apply the stricter English-focused language rule."""
    audio_language = normalize_language_code(row.get("default_audio_language"))
    default_language = normalize_language_code(row.get("default_language"))

    if audio_language and is_explicit_non_english(audio_language):
        return False, "clearly_non_english"

    if not audio_language and default_language and is_explicit_non_english(default_language):
        return False, "clearly_non_english"

    if audio_language and is_explicit_english(audio_language):
        return True, ""

    if default_language and is_explicit_english(default_language):
        return True, ""

    title = str(row.get("title") or "")
    channel_title = str(row.get("channel_title") or "")
    if title_looks_english(title, channel_title):
        return True, ""

    return False, "clearly_non_english"


def detect_irrelevant_content(title: Any, description: Any) -> bool:
    """Detect clear gaming or betting content."""
    text = normalize_text(f"{title or ''} {description or ''}")
    if "fifa world cup 2026" in text:
        protected_phrase_present = True
    else:
        protected_phrase_present = False

    if "fifa 26" in text and protected_phrase_present:
        text = text.replace("fifa world cup 2026", "")

    return any(term in text for term in GAME_OR_BETTING_TERMS)


def choose_primary_category(row: pd.Series) -> str:
    """Choose a simple primary category."""
    existing = normalize_text(row.get("primary_category"))
    if existing:
        return existing
    first_categories = split_delimited_values(row.get("search_categories"))
    return first_categories[0] if first_categories else ""


def classify_video_context(title: Any, description: Any, primary_category: Any = "", search_categories: Any = "") -> str:
    """Assign one compact video context type with simple rule-based logic.

    Rule order:
    1. controversy terms such as VAR, referee, penalty, controversy, robbed
    2. highlight terms such as highlights, goals, extended or full highlights
    3. reaction terms such as post-match reaction, fan reaction, instant reaction
    4. underdog terms such as upset, shock, surprise
    5. star-player terms such as legacy, GOAT, hat-trick, obvious player names
    6. tactical terms such as analysis, manager, formation, why a team lost
    7. rivalry or national-pride terms
    8. group draw, roster, squad, players-to-watch terms
    9. watchalong or live-reaction terms
    10. preview or background terms
    11. otherwise other_world_cup_discussion or uncertain

    This is video-level metadata used for later interpretation, not a label for
    every comment under the video.
    """
    text = normalize_text(f"{title or ''} {description or ''} {primary_category or ''} {search_categories or ''}")
    ordered_contexts = [
        "var_referee_controversy",
        "match_highlights",
        "post_match_reaction",
        "upset_underdog",
        "star_player_legacy",
        "tactical_analysis",
        "rivalry_national_pride",
        "group_draw_roster",
        "watchalong_reaction",
        "preview_background",
    ]
    for context_type in ordered_contexts:
        patterns = VIDEO_CONTEXT_PATTERNS[context_type]
        if any(pattern in text for pattern in patterns):
            return context_type
    if "world cup" in text:
        return "other_world_cup_discussion"
    return "uncertain"


def compact_video_row(row: dict[str, Any]) -> dict[str, Any]:
    """Project a row into the compact sampling schema."""
    return {
        "video_id": safe_text(row.get("video_id")),
        "url": safe_text(row.get("url")),
        "title": safe_text(row.get("title")),
        "channel_title": safe_text(row.get("channel_title")),
        "published_at": safe_text(row.get("published_at")),
        "duration_seconds": safe_int(row.get("duration_seconds")),
        "view_count": safe_int(row.get("view_count")),
        "like_count": safe_int(row.get("like_count")),
        "comment_count": safe_int(row.get("comment_count")),
        "default_language": safe_text(row.get("default_language")),
        "default_audio_language": safe_text(row.get("default_audio_language")),
        "search_categories": safe_text(row.get("search_categories")),
        "primary_category": safe_text(row.get("primary_category")),
        "video_context_type": safe_text(row.get("video_context_type")),
        "eligibility_status": safe_text(row.get("eligibility_status")),
        "exclusion_reason": safe_text(row.get("exclusion_reason")),
        "comments_collected": safe_int(row.get("comments_collected")) or 0,
        "collection_error": safe_text(row.get("collection_error")),
        "collection_date": safe_text(row.get("collection_date")),
    }


def discover_candidate_videos(config_path: Path, output_path: Path, overwrite: bool) -> None:
    """Discover candidate videos and save a detailed intermediate CSV."""
    config = load_query_config(config_path)
    youtube = build_youtube_client(YOUTUBE_API_KEY)
    max_results_per_query = int(config.get("max_search_results_per_query", 25))
    search_order = str(config.get("video_search_order", "relevance"))

    aggregated_hits: dict[str, dict[str, Any]] = {}
    print_status("Starting candidate discovery.")

    for category, queries in config["categories"].items():
        print_status(f"Searching category '{category}' with {len(queries)} query variants.")
        for query in queries:
            results = search_videos_for_query(youtube=youtube, query=query, max_results=max_results_per_query, order=search_order)
            for rank, item in enumerate(results, start=1):
                video_id = item.get("id", {}).get("videoId")
                if not video_id:
                    continue
                snippet = item.get("snippet", {})
                candidate = aggregated_hits.setdefault(
                    video_id,
                    {
                        "video_id": video_id,
                        "search_categories": set(),
                        "best_rank": rank,
                        "fallback_title": snippet.get("title"),
                        "fallback_channel_title": snippet.get("channelTitle"),
                        "fallback_published_at": snippet.get("publishedAt"),
                    },
                )
                candidate["search_categories"].add(category)
                candidate["best_rank"] = min(candidate["best_rank"], rank)

    metadata = fetch_video_metadata(youtube, list(aggregated_hits))
    rows: list[dict[str, Any]] = []
    for video_id, candidate in aggregated_hits.items():
        details = metadata.get(video_id, {})
        search_categories = SEARCH_CATEGORY_SEPARATOR.join(sorted(candidate["search_categories"]))
        row = {
            "video_id": video_id,
            "url": details.get("url") or f"https://www.youtube.com/watch?v={video_id}",
            "title": details.get("title") or candidate.get("fallback_title"),
            "channel_title": details.get("channel_title") or candidate.get("fallback_channel_title"),
            "published_at": details.get("published_at") or candidate.get("fallback_published_at"),
            "duration_seconds": details.get("duration_seconds"),
            "view_count": details.get("view_count"),
            "like_count": details.get("like_count"),
            "comment_count": details.get("comment_count"),
            "default_language": details.get("default_language"),
            "default_audio_language": details.get("default_audio_language"),
            "search_categories": search_categories,
            "primary_category": sorted(candidate["search_categories"])[0] if candidate["search_categories"] else "",
            "video_context_type": classify_video_context(
                details.get("title"),
                details.get("description"),
                sorted(candidate["search_categories"])[0] if candidate["search_categories"] else "",
                search_categories,
            ),
            "eligibility_status": "",
            "exclusion_reason": "",
            "comments_collected": 0,
            "collection_error": "",
            "collection_date": "",
            "privacy_status": details.get("privacy_status"),
            "live_broadcast_content": details.get("live_broadcast_content"),
            "comment_count_visible": details.get("comment_count_visible"),
            "description": details.get("description"),
        }
        rows.append(row)

    candidate_df = pd.DataFrame(rows).drop_duplicates(subset=["video_id"], keep="first")
    write_csv_atomic(candidate_df, output_path, overwrite=overwrite, dedupe_column="video_id")
    print_status(f"Saved {len(candidate_df)} candidate videos to {display_path(output_path)}.")


def main_exclusion_reason(row: pd.Series, min_comments: int, min_duration_seconds: int) -> str:
    """Return the first matching exclusion reason from the controlled vocabulary."""
    if pd.isna(row.get("privacy_status")) or pd.isna(row.get("live_broadcast_content")) or pd.isna(row.get("comment_count_visible")) or pd.isna(row.get("duration_seconds")):
        return "missing_required_metadata"
    if normalize_text(row.get("privacy_status")) != "public":
        return "non_public"
    if bool(row.get("comment_count_visible")) is not True:
        return "comments_unavailable"
    comment_count = safe_int(row.get("comment_count"))
    if comment_count is None:
        return "comments_unavailable"
    if comment_count < min_comments:
        return "below_min_comments"
    if normalize_text(row.get("live_broadcast_content")) != "none":
        return "live_or_scheduled"
    if safe_int(row.get("duration_seconds")) is None:
        return "missing_required_metadata"
    if safe_int(row.get("duration_seconds")) < min_duration_seconds:
        return "likely_short"
    language_ok, language_reason = detect_language_reason(row)
    if not language_ok:
        return language_reason
    if detect_irrelevant_content(row.get("title"), row.get("description")):
        return "irrelevant_gaming_or_betting"
    return ""


def filter_eligible_videos(
    config_path: Path,
    candidates_path: Path,
    output_path: Path,
    overwrite: bool,
    write_audit_splits: bool,
) -> None:
    """Filter candidate videos into one compact sampling file."""
    config = load_query_config(config_path)
    candidate_df = load_clean_csv(candidates_path).drop_duplicates(subset=["video_id"], keep="first").copy()
    min_comments = int(config.get("min_public_comments_pilot", 100))
    min_duration_seconds = int(config.get("min_duration_seconds", 90))

    sampling_rows: list[dict[str, Any]] = []
    for _, row in candidate_df.iterrows():
        row = row.copy()
        row["primary_category"] = choose_primary_category(row)
        row["video_context_type"] = classify_video_context(
            row.get("title"),
            row.get("description"),
            row.get("primary_category"),
            row.get("search_categories"),
        )
        exclusion_reason = main_exclusion_reason(row, min_comments=min_comments, min_duration_seconds=min_duration_seconds)
        row["eligibility_status"] = "excluded" if exclusion_reason else "eligible"
        row["exclusion_reason"] = exclusion_reason
        row["comments_collected"] = 0
        row["collection_error"] = ""
        row["collection_date"] = ""
        sampling_rows.append(compact_video_row(row.to_dict()))

    sampling_df = pd.DataFrame(sampling_rows, columns=SAMPLING_COLUMNS).drop_duplicates(subset=["video_id"], keep="first")
    write_csv_atomic(sampling_df, output_path, overwrite=overwrite, dedupe_column="video_id")
    print_status(f"Saved compact sampling file to {display_path(output_path)}.")

    if write_audit_splits:
        eligible_df = sampling_df[sampling_df["eligibility_status"] == "eligible"].copy()
        excluded_df = sampling_df[sampling_df["eligibility_status"] == "excluded"].copy()
        write_csv_atomic(eligible_df, DEFAULT_ELIGIBLE_SPLIT_OUTPUT, overwrite=overwrite, dedupe_column="video_id")
        write_csv_atomic(excluded_df, DEFAULT_EXCLUDED_SPLIT_OUTPUT, overwrite=overwrite, dedupe_column="video_id")
        print_status("Wrote optional eligible/excluded audit split files.")

    summary_df = build_summary_from_sampling_and_comments(sampling_df=sampling_df, comments_df=pd.DataFrame(columns=COMMENT_COLUMNS))
    write_csv_atomic(summary_df, DEFAULT_SUMMARY_OUTPUT, overwrite=overwrite)
    print_status(f"Saved summary file to {display_path(DEFAULT_SUMMARY_OUTPUT)}.")


def parse_selected_flag(value: Any) -> bool:
    """Interpret a CSV value as a boolean selection flag."""
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in SELECTED_TRUE_VALUES


def load_sampling_input(path: Path) -> pd.DataFrame:
    """Load the compact sampling file for collection."""
    sampling_df = load_clean_csv(path)
    if "eligibility_status" not in sampling_df.columns:
        raise ValueError("Sampling file must contain an eligibility_status column.")
    if "selected" in sampling_df.columns:
        selected_mask = sampling_df["selected"].apply(parse_selected_flag)
    else:
        selected_mask = pd.Series(False, index=sampling_df.index)

    eligible_mask = sampling_df["eligibility_status"].astype(str).str.lower().eq("eligible")
    chosen_df = sampling_df[eligible_mask | selected_mask].copy()
    chosen_df = chosen_df.drop_duplicates(subset=["video_id"], keep="first")
    return chosen_df


def collect_video_comments(
    youtube,
    video_id: str,
    max_comments: int,
    order: str,
    collection_date: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Collect top-level comments for one video."""
    rows: list[dict[str, Any]] = []
    seen_comment_ids: set[str] = set()
    next_page_token: str | None = None

    try:
        from googleapiclient.errors import HttpError
    except ModuleNotFoundError:
        HttpError = Exception  # pragma: no cover

    try:
        while len(rows) < max_comments:
            batch_size = min(100, max_comments - len(rows))
            request_kwargs: dict[str, Any] = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": batch_size,
                "order": order,
                "textFormat": "plainText",
            }
            if next_page_token:
                request_kwargs["pageToken"] = next_page_token
            response = youtube.commentThreads().list(**request_kwargs).execute()

            for item in response.get("items", []):
                comment_id = item.get("id")
                if not comment_id or comment_id in seen_comment_ids:
                    continue
                seen_comment_ids.add(comment_id)
                snippet = item.get("snippet", {})
                top_level = snippet.get("topLevelComment", {}).get("snippet", {})
                author_channel_id = top_level.get("authorChannelId") or {}
                rows.append(
                    {
                        "comment_id": comment_id,
                        "video_id": video_id,
                        "author_channel_id": author_channel_id.get("value"),
                        "author_display_name": top_level.get("authorDisplayName"),
                        "comment_text": top_level.get("textOriginal"),
                        "published_at": top_level.get("publishedAt"),
                        "updated_at": top_level.get("updatedAt"),
                        "like_count": safe_int(top_level.get("likeCount")),
                        "reply_count": safe_int(snippet.get("totalReplyCount")),
                        "collection_date": collection_date,
                    }
                )
                if len(rows) >= max_comments:
                    break

            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break
    except HttpError as error:
        return rows, {"collection_error": str(error), "comments_collected": len(rows)}

    return rows, {"collection_error": "", "comments_collected": len(rows)}


def build_summary_from_sampling_and_comments(sampling_df: pd.DataFrame, comments_df: pd.DataFrame) -> pd.DataFrame:
    """Build one compact key-value summary table."""
    eligible_df = sampling_df[sampling_df["eligibility_status"] == "eligible"].copy()
    excluded_df = sampling_df[sampling_df["eligibility_status"] == "excluded"].copy()

    comments_collected = len(comments_df)
    unique_commenters = int(comments_df["author_channel_id"].dropna().nunique()) if not comments_df.empty else 0
    videos_with_comments_collected = int(comments_df["video_id"].nunique()) if not comments_df.empty else 0

    if comments_df.empty:
        users_with_at_least_2_videos = 0
        users_with_at_least_3_videos = 0
        per_video_counts: list[int] = []
    else:
        user_video_counts = comments_df.dropna(subset=["author_channel_id"]).groupby("author_channel_id")["video_id"].nunique()
        users_with_at_least_2_videos = int((user_video_counts >= 2).sum())
        users_with_at_least_3_videos = int((user_video_counts >= 3).sum())
        per_video_counts = comments_df.groupby("video_id").size().tolist()

    eligible_by_primary = dict(sorted(eligible_df["primary_category"].fillna("").replace("", "uncertain").value_counts().to_dict().items()))
    eligible_by_context = dict(sorted(eligible_df["video_context_type"].fillna("uncertain").value_counts().to_dict().items()))
    excluded_by_reason = dict(sorted(excluded_df["exclusion_reason"].fillna("").replace("", "unknown").value_counts().to_dict().items()))

    rows = [
        {"metric": "total_candidate_videos", "value": len(sampling_df)},
        {"metric": "eligible_videos", "value": len(eligible_df)},
        {"metric": "excluded_videos", "value": len(excluded_df)},
        {"metric": "comments_collected", "value": comments_collected},
        {"metric": "unique_commenters", "value": unique_commenters},
        {"metric": "videos_with_comments_collected", "value": videos_with_comments_collected},
        {"metric": "users_with_at_least_2_videos", "value": users_with_at_least_2_videos},
        {"metric": "users_with_at_least_3_videos", "value": users_with_at_least_3_videos},
        {"metric": "min_comments_per_video", "value": min(per_video_counts) if per_video_counts else 0},
        {"metric": "mean_comments_per_video", "value": round(mean(per_video_counts), 2) if per_video_counts else 0},
        {"metric": "max_comments_per_video", "value": max(per_video_counts) if per_video_counts else 0},
        {"metric": "eligible_by_primary_category", "value": json.dumps(eligible_by_primary, ensure_ascii=False, sort_keys=True)},
        {"metric": "eligible_by_video_context_type", "value": json.dumps(eligible_by_context, ensure_ascii=False, sort_keys=True)},
        {"metric": "excluded_by_reason", "value": json.dumps(excluded_by_reason, ensure_ascii=False, sort_keys=True)},
    ]
    return pd.DataFrame(rows)


def collect_comments_for_videos(
    config_path: Path,
    selected_videos_path: Path,
    comments_output_path: Path,
    summary_output_path: Path,
    max_comments_override: int | None,
    overwrite: bool,
) -> None:
    """Collect capped top-level comments for eligible videos."""
    config = load_query_config(config_path)
    sampling_df = load_clean_csv(selected_videos_path)
    selected_df = load_sampling_input(selected_videos_path)
    youtube = build_youtube_client(YOUTUBE_API_KEY)

    max_comments = max_comments_override if max_comments_override is not None else int(config.get("max_comments_per_video_pilot", 200))
    comment_order = str(config.get("comments_order", "relevance"))
    collection_date = utc_timestamp()

    print_status(f"Collecting comments for {len(selected_df)} eligible videos.")
    print_status(f"Outputs will be written to {display_path(comments_output_path)} and {display_path(summary_output_path)}.")

    collected_comments: list[dict[str, Any]] = []
    collection_map: dict[str, dict[str, Any]] = {}

    for index, row in enumerate(selected_df.to_dict(orient="records"), start=1):
        video_id = str(row["video_id"])
        print_status(f"[{index}/{len(selected_df)}] Collecting comments for {video_id}.")
        comment_rows, status = collect_video_comments(
            youtube=youtube,
            video_id=video_id,
            max_comments=max_comments,
            order=comment_order,
            collection_date=collection_date,
        )
        collected_comments.extend(comment_rows)
        collection_map[video_id] = status

    comments_df = pd.DataFrame(collected_comments, columns=COMMENT_COLUMNS)
    if not comments_df.empty:
        comments_df = comments_df.drop_duplicates(subset=["comment_id"], keep="first").reset_index(drop=True)

    sampling_out_df = sampling_df.copy()
    if "comments_collected" not in sampling_out_df.columns:
        sampling_out_df["comments_collected"] = 0
    sampling_out_df["comments_collected"] = pd.to_numeric(sampling_out_df["comments_collected"], errors="coerce").fillna(0)
    if "collection_error" not in sampling_out_df.columns:
        sampling_out_df["collection_error"] = ""
    sampling_out_df["collection_error"] = sampling_out_df["collection_error"].astype("object")
    sampling_out_df["collection_error"] = sampling_out_df["collection_error"].where(~sampling_out_df["collection_error"].isna(), "")
    if "collection_date" not in sampling_out_df.columns:
        sampling_out_df["collection_date"] = ""
    sampling_out_df["collection_date"] = sampling_out_df["collection_date"].astype("object")
    sampling_out_df["collection_date"] = sampling_out_df["collection_date"].where(~sampling_out_df["collection_date"].isna(), "")

    for index, row in sampling_out_df.iterrows():
        video_id = str(row.get("video_id"))
        if video_id not in collection_map:
            continue
        sampling_out_df.at[index, "comments_collected"] = collection_map[video_id]["comments_collected"]
        sampling_out_df.at[index, "collection_error"] = collection_map[video_id]["collection_error"]
        sampling_out_df.at[index, "collection_date"] = collection_date

    compact_sampling_df = pd.DataFrame(
        [compact_video_row(row) for row in sampling_out_df.to_dict(orient="records")],
        columns=SAMPLING_COLUMNS,
    ).drop_duplicates(subset=["video_id"], keep="first")
    summary_df = build_summary_from_sampling_and_comments(compact_sampling_df, comments_df)

    write_csv_atomic(comments_df, comments_output_path, overwrite=overwrite, dedupe_column="comment_id")
    write_csv_atomic(compact_sampling_df, selected_videos_path, overwrite=overwrite, dedupe_column="video_id")
    write_csv_atomic(summary_df, summary_output_path, overwrite=overwrite)

    print_status(f"Saved raw comments to {display_path(comments_output_path)}.")
    print_status(f"Updated sampling file at {display_path(selected_videos_path)}.")
    print_status(f"Saved summary file to {display_path(summary_output_path)}.")


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()

    if args.command == "discover":
        discover_candidate_videos(
            config_path=resolve_path(args.config),
            output_path=resolve_path(args.output),
            overwrite=args.overwrite,
        )
        return

    if args.command == "filter-eligible":
        filter_eligible_videos(
            config_path=resolve_path(args.config),
            candidates_path=resolve_path(args.candidates),
            output_path=resolve_path(args.output),
            overwrite=args.overwrite,
            write_audit_splits=args.write_audit_splits,
        )
        return

    if args.command == "collect":
        collect_comments_for_videos(
            config_path=resolve_path(args.config),
            selected_videos_path=resolve_path(args.selected),
            comments_output_path=resolve_path(args.comments_output),
            summary_output_path=resolve_path(args.summary_output),
            max_comments_override=args.max_comments,
            overwrite=args.overwrite,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()