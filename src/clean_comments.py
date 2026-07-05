"""Preflight checks and light preprocessing for YouTube comments."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from config import OUTPUTS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, ROOT_DIR


DEFAULT_COMMENTS_PATH = RAW_DATA_DIR / "youtube_comments_pilot.csv"
DEFAULT_VIDEOS_PATH = ROOT_DIR / "metadata" / "video_sampling_pilot.csv"
DEFAULT_OUTPUT_PATH = PROCESSED_DATA_DIR / "comments_processed.csv"
DEFAULT_SUMMARY_PATH = OUTPUTS_DIR / "tables" / "preprocessing_summary.csv"
AUTHOR_SALT = "sma_world_cup_2026"

ENGLISH_FUNCTION_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "for",
    "from",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "our",
    "that",
    "the",
    "their",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "with",
    "you",
    "your",
}

FOOTBALL_COMMENT_TERMS = {
    "argentina",
    "ball",
    "defence",
    "england",
    "fifa",
    "final",
    "football",
    "france",
    "goal",
    "goat",
    "keeper",
    "match",
    "messi",
    "player",
    "penalty",
    "ronaldo",
    "striker",
    "team",
    "var",
    "win",
    "world",
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

VIDEO_REQUIRED_COLUMNS = [
    "video_id",
    "title",
    "channel_title",
    "primary_category",
    "video_context_type",
    "eligibility_status",
    "exclusion_reason",
    "comment_count",
    "default_language",
    "default_audio_language",
]

COMMENT_REQUIRED_COLUMNS = [
    "comment_id",
    "video_id",
    "author_channel_id",
    "comment_text",
    "published_at",
    "updated_at",
    "like_count",
    "reply_count",
    "collection_date",
]

PROCESSED_COLUMNS = [
    "comment_id",
    "video_id",
    "author_id",
    "original_text",
    "clean_text",
    "published_at",
    "updated_at",
    "like_count",
    "reply_count",
    "collection_date",
    "text_length",
    "likely_english",
    "title",
    "channel_title",
    "primary_category",
    "video_context_type",
]

SUMMARY_METRICS_ORDER = [
    "raw_comments",
    "duplicate_comment_ids_before_dedup",
    "duplicate_comments_removed",
    "empty_comments_removed",
    "short_comments_removed",
    "non_english_comments_removed",
    "processed_comments",
    "unique_users_processed",
    "unique_videos_processed",
    "users_with_1_video_raw",
    "users_with_at_least_2_videos_raw",
    "users_with_at_least_3_videos_raw",
    "eligible_videos_with_comments",
    "eligible_videos_without_comments",
    "min_comments_per_video_raw",
    "mean_comments_per_video_raw",
    "max_comments_per_video_raw",
    "min_text_length_processed",
    "mean_text_length_processed",
    "max_text_length_processed",
    "comments_by_primary_category_processed",
    "comments_by_video_context_type_processed",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Preflight checks and light preprocessing for YouTube comments.")
    parser.add_argument("--comments", default=str(DEFAULT_COMMENTS_PATH.relative_to(ROOT_DIR)), help="Raw comments CSV path.")
    parser.add_argument("--videos", default=str(DEFAULT_VIDEOS_PATH.relative_to(ROOT_DIR)), help="Video sampling CSV path.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH.relative_to(ROOT_DIR)), help="Processed comments CSV path.")
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY_PATH.relative_to(ROOT_DIR)), help="Preprocessing summary CSV path.")
    parser.add_argument("--min-chars", type=int, default=15, help="Minimum clean-text length to keep.")
    parser.add_argument("--keep-non-english", action="store_true", help="Keep comments even when likely_english is false.")
    parser.set_defaults(overwrite=True)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite generated outputs. This is the default behavior.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Stop instead of overwriting existing generated outputs.")
    return parser.parse_args()


def display_path(path: Path) -> str:
    """Render a path relative to the repository root when possible."""
    if path.is_relative_to(ROOT_DIR):
        return str(path.relative_to(ROOT_DIR))
    return str(path)


def resolve_path(path_text: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def print_status(message: str) -> None:
    """Print a terminal status message."""
    print(f"[clean_comments] {message}")


def write_csv_atomic(dataframe: pd.DataFrame, output_path: Path, overwrite: bool) -> None:
    """Write a CSV atomically, overwriting by default."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {display_path(output_path)}")

    temp_path = output_path.parent / f".{output_path.name}.tmp"
    if temp_path.exists():
        temp_path.unlink()

    if output_path.exists():
        print_status(f"Overwriting existing file: {display_path(output_path)}")
    else:
        print_status(f"Writing new file: {display_path(output_path)}")

    dataframe.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file."""
    return pd.read_csv(path)


def require_columns(dataframe: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Validate required columns."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_label}: {', '.join(missing)}")


def normalize_language_code(value: Any) -> str:
    """Normalize a language code such as en-US to en-us."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().lower()


def is_explicit_non_english(language_code: str) -> bool:
    """Check whether a language code is explicitly non-English."""
    if not language_code:
        return False
    if language_code == "en" or language_code.startswith("en-"):
        return False
    return language_code.split("-", 1)[0] in NON_ENGLISH_PREFIXES


def ascii_letter_ratio(text: str) -> float:
    """Estimate the share of alphabetic text that is ASCII."""
    letters = re.findall("[A-Za-z\u00C0-\u00FF]", text)
    if not letters:
        return 0.0
    ascii_letters = re.findall(r"[A-Za-z]", text)
    return len(ascii_letters) / len(letters)


def likely_english_comment(text: str) -> bool:
    """Flag likely English comments with a transparent rule-based heuristic.

    This is not full language detection. It uses a simple combination of
    ASCII/Latin-letter ratio, common English function words, football-related
    terms, and token count so the filter stays interpretable and easy to
    justify in the project report.
    """
    normalized = text.strip()
    if not normalized:
        return False

    tokens = re.findall(r"[A-Za-z]+", normalized.lower())
    if not tokens:
        return False

    ascii_ratio = ascii_letter_ratio(normalized)
    function_word_hits = len(set(tokens).intersection(ENGLISH_FUNCTION_WORDS))
    football_hits = len(set(tokens).intersection(FOOTBALL_COMMENT_TERMS))

    if ascii_ratio < 0.55:
        return False
    if function_word_hits >= 1:
        return True
    if ascii_ratio >= 0.9 and football_hits >= 1:
        return True
    if ascii_ratio >= 0.95 and len(tokens) >= 4:
        return True
    return False


def clean_comment_text(text: str) -> str:
    """Apply light transparent cleaning to comment text."""
    cleaned = html.unescape(text)
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    cleaned = cleaned.replace("\r", " ").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def anonymize_author(author_channel_id: Any) -> str:
    """Create a stable pseudonymous author ID from author_channel_id.

    The same YouTube channel ID always maps to the same SHA-256-based
    anonymous ID. This preserves repeat participation for later network
    analysis while keeping raw channel IDs and display names out of the
    processed dataset.
    """
    source = "" if author_channel_id is None or pd.isna(author_channel_id) else str(author_channel_id)
    return hashlib.sha256(f"{AUTHOR_SALT}:{source}".encode("utf-8")).hexdigest()


def counts_to_json(series: pd.Series) -> str:
    """Convert value counts to a compact JSON string."""
    clean_series = series.fillna("").astype(str)
    clean_series = clean_series[clean_series != ""]
    return json.dumps(dict(sorted(clean_series.value_counts().to_dict().items())), ensure_ascii=False, sort_keys=True)


def preflight_checks(videos_df: pd.DataFrame, comments_df: pd.DataFrame) -> tuple[dict[str, Any], list[str]]:
    """Run compact preflight checks and return summary metrics plus warnings."""
    warnings: list[str] = []

    total_videos = len(videos_df)
    eligible_videos_df = videos_df[videos_df["eligibility_status"] == "eligible"].copy()
    raw_comments = len(comments_df)
    unique_commenters = int(comments_df["author_channel_id"].dropna().nunique())

    if raw_comments == 0:
        raise ValueError("Zero raw comments found.")
    if eligible_videos_df.empty:
        raise ValueError("Zero eligible videos found.")
    if unique_commenters == 0:
        raise ValueError("Zero unique commenters found.")

    duplicate_video_rows = int(videos_df["video_id"].duplicated().sum())
    duplicate_comment_ids = int(comments_df["comment_id"].duplicated().sum())
    comments_per_video = comments_df.groupby("video_id").size()

    eligible_video_ids = set(eligible_videos_df["video_id"].astype(str))
    videos_with_comments = set(comments_df["video_id"].astype(str))
    eligible_with_comments = len(eligible_video_ids.intersection(videos_with_comments))
    eligible_without_comments = len(eligible_video_ids.difference(videos_with_comments))

    user_video_counts = comments_df.dropna(subset=["author_channel_id"]).groupby("author_channel_id")["video_id"].nunique()
    users_with_1_video = int((user_video_counts == 1).sum())
    users_with_at_least_2_videos = int((user_video_counts >= 2).sum())
    users_with_at_least_3_videos = int((user_video_counts >= 3).sum())

    video_metadata_lookup = videos_df[["video_id", "primary_category", "video_context_type"]].drop_duplicates(subset=["video_id"])
    comments_with_video = comments_df.merge(video_metadata_lookup, on="video_id", how="left")

    eligible_by_primary_category = counts_to_json(eligible_videos_df["primary_category"])
    comments_by_primary_category = counts_to_json(comments_with_video["primary_category"])
    eligible_by_video_context_type = counts_to_json(eligible_videos_df["video_context_type"])
    comments_by_video_context_type = counts_to_json(comments_with_video["video_context_type"])

    default_language_counts = counts_to_json(eligible_videos_df["default_language"])
    default_audio_language_counts = counts_to_json(eligible_videos_df["default_audio_language"])

    eligible_videos_df["default_language_norm"] = eligible_videos_df["default_language"].apply(normalize_language_code)
    eligible_videos_df["default_audio_language_norm"] = eligible_videos_df["default_audio_language"].apply(normalize_language_code)

    if len(eligible_videos_df) > 0 and eligible_with_comments / len(eligible_videos_df) < 0.7:
        warnings.append("Fewer than 70% of eligible videos have at least one collected comment.")
    if eligible_without_comments > 0:
        warnings.append(f"{eligible_without_comments} eligible videos have zero collected comments.")
    if duplicate_comment_ids > 0:
        warnings.append(f"{duplicate_comment_ids} duplicate comment IDs detected before deduplication.")
    if users_with_at_least_2_videos < 50:
        warnings.append("Users with at least 2 videos are below 50, so the co-commenting network may be weak.")

    primary_comment_counts = comments_with_video["primary_category"].fillna("").replace("", "uncertain").value_counts(normalize=True)
    if not primary_comment_counts.empty and float(primary_comment_counts.iloc[0]) > 0.6:
        warnings.append("One primary_category accounts for more than 60% of raw comments.")

    audio_language_issues = eligible_videos_df["default_audio_language_norm"].apply(is_explicit_non_english)
    if bool(audio_language_issues.any()):
        warnings.append("Some eligible videos have explicitly non-English default_audio_language.")

    default_language_only_issues = (
        eligible_videos_df["default_audio_language_norm"].eq("")
        & eligible_videos_df["default_language_norm"].apply(is_explicit_non_english)
    )
    if bool(default_language_only_issues.any()):
        warnings.append("Some eligible videos have missing default_audio_language but explicitly non-English default_language.")

    metrics = {
        "raw_comments": raw_comments,
        "duplicate_comment_ids_before_dedup": duplicate_comment_ids,
        "eligible_videos_with_comments": eligible_with_comments,
        "eligible_videos_without_comments": eligible_without_comments,
        "users_with_1_video_raw": users_with_1_video,
        "users_with_at_least_2_videos_raw": users_with_at_least_2_videos,
        "users_with_at_least_3_videos_raw": users_with_at_least_3_videos,
        "min_comments_per_video_raw": int(comments_per_video.min()) if not comments_per_video.empty else 0,
        "mean_comments_per_video_raw": round(float(comments_per_video.mean()), 2) if not comments_per_video.empty else 0,
        "max_comments_per_video_raw": int(comments_per_video.max()) if not comments_per_video.empty else 0,
    }

    print_status(f"Preflight: {total_videos} videos, {len(eligible_videos_df)} eligible, {raw_comments} raw comments, {unique_commenters} unique commenters.")
    print_status(f"Preflight: users with >=2 videos = {users_with_at_least_2_videos}, users with >=3 videos = {users_with_at_least_3_videos}.")
    print_status(f"Preflight: eligible videos by primary_category = {eligible_by_primary_category}")
    print_status(f"Preflight: raw comments by primary_category = {comments_by_primary_category}")
    print_status(f"Preflight: eligible videos by video_context_type = {eligible_by_video_context_type}")
    print_status(f"Preflight: raw comments by video_context_type = {comments_by_video_context_type}")
    print_status(f"Preflight: eligible default_language counts = {default_language_counts}")
    print_status(f"Preflight: eligible default_audio_language counts = {default_audio_language_counts}")
    if duplicate_video_rows > 0:
        print_status(f"WARNING: {duplicate_video_rows} duplicate video_id rows detected in the sampling file.")
    for warning in warnings:
        print_status(f"WARNING: {warning}")

    return metrics, warnings


def preprocess_comments(
    comments_df: pd.DataFrame,
    videos_df: pd.DataFrame,
    min_chars: int,
    keep_non_english: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run light preprocessing and return the processed dataframe plus summary metrics."""
    duplicate_comment_ids = int(comments_df["comment_id"].duplicated().sum())
    comments_df = comments_df.drop_duplicates(subset=["comment_id"], keep="first").copy()

    before_empty = len(comments_df)
    comments_df["comment_text"] = comments_df["comment_text"].fillna("").astype(str)
    comments_df = comments_df[comments_df["comment_text"].str.strip() != ""].copy()
    empty_comments_removed = before_empty - len(comments_df)

    comments_df["original_text"] = comments_df["comment_text"]
    comments_df["clean_text"] = comments_df["original_text"].apply(clean_comment_text)
    comments_df["text_length"] = comments_df["clean_text"].str.len()

    before_short = len(comments_df)
    comments_df = comments_df[comments_df["text_length"] >= min_chars].copy()
    short_comments_removed = before_short - len(comments_df)

    comments_df["likely_english"] = comments_df["clean_text"].apply(likely_english_comment)
    before_language = len(comments_df)
    if not keep_non_english:
        comments_df = comments_df[comments_df["likely_english"]].copy()
    non_english_comments_removed = before_language - len(comments_df)

    comments_df["author_id"] = comments_df["author_channel_id"].apply(anonymize_author)

    # Keep only compact metadata needed for later interpretation of eligible videos.
    eligible_videos_df = videos_df[videos_df["eligibility_status"] == "eligible"][
        ["video_id", "title", "channel_title", "primary_category", "video_context_type", "eligibility_status"]
    ].drop_duplicates(subset=["video_id"])
    comments_df = comments_df.merge(eligible_videos_df, on="video_id", how="inner")

    processed_df = comments_df[PROCESSED_COLUMNS].copy()
    summary = {
        "duplicate_comments_removed": duplicate_comment_ids,
        "empty_comments_removed": empty_comments_removed,
        "short_comments_removed": short_comments_removed,
        "non_english_comments_removed": non_english_comments_removed,
        "processed_comments": len(processed_df),
        "unique_users_processed": int(processed_df["author_id"].nunique()) if not processed_df.empty else 0,
        "unique_videos_processed": int(processed_df["video_id"].nunique()) if not processed_df.empty else 0,
        "min_text_length_processed": int(processed_df["text_length"].min()) if not processed_df.empty else 0,
        "mean_text_length_processed": round(float(processed_df["text_length"].mean()), 2) if not processed_df.empty else 0,
        "max_text_length_processed": int(processed_df["text_length"].max()) if not processed_df.empty else 0,
        "comments_by_primary_category_processed": counts_to_json(processed_df["primary_category"]),
        "comments_by_video_context_type_processed": counts_to_json(processed_df["video_context_type"]),
    }
    return processed_df, summary


def build_summary_rows(preflight_metrics: dict[str, Any], preprocessing_metrics: dict[str, Any], warnings: list[str]) -> pd.DataFrame:
    """Build simple key-value summary rows."""
    merged = dict(preflight_metrics)
    merged.update(preprocessing_metrics)
    rows = [{"metric": metric, "value": merged[metric]} for metric in SUMMARY_METRICS_ORDER]
    rows.append({"metric": "warnings", "value": " | ".join(warnings)})
    return pd.DataFrame(rows)


def main() -> None:
    """Run the preprocessing CLI."""
    args = parse_args()
    comments_path = resolve_path(args.comments)
    videos_path = resolve_path(args.videos)
    output_path = resolve_path(args.output)
    summary_path = resolve_path(args.summary)

    if not comments_path.exists():
        raise FileNotFoundError(f"Missing required file: {display_path(comments_path)}")
    if not videos_path.exists():
        raise FileNotFoundError(f"Missing required file: {display_path(videos_path)}")

    comments_df = load_csv(comments_path)
    videos_df = load_csv(videos_path)
    require_columns(comments_df, COMMENT_REQUIRED_COLUMNS, display_path(comments_path))
    require_columns(videos_df, VIDEO_REQUIRED_COLUMNS, display_path(videos_path))

    preflight_metrics, warnings = preflight_checks(videos_df, comments_df)
    processed_df, preprocessing_metrics = preprocess_comments(
        comments_df=comments_df,
        videos_df=videos_df,
        min_chars=args.min_chars,
        keep_non_english=args.keep_non_english,
    )
    summary_df = build_summary_rows(preflight_metrics, preprocessing_metrics, warnings)

    write_csv_atomic(processed_df, output_path, overwrite=args.overwrite)
    write_csv_atomic(summary_df, summary_path, overwrite=args.overwrite)

    print_status(
        f"Processed {preprocessing_metrics['processed_comments']} comments from {preprocessing_metrics['unique_users_processed']} users across {preprocessing_metrics['unique_videos_processed']} videos."
    )


if __name__ == "__main__":
    main()