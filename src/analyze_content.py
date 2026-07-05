"""Profile detected participation communities using comment text."""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from collections import Counter, defaultdict
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer

from config import FIGURES_DIR, PROCESSED_DATA_DIR, ROOT_DIR, TABLES_DIR


DEFAULT_COMMENTS_PATH = PROCESSED_DATA_DIR / "comments_processed.csv"
DEFAULT_COMMUNITIES_PATH = PROCESSED_DATA_DIR / "user_communities.csv"
DEFAULT_COMMUNITY_SUMMARY_PATH = PROCESSED_DATA_DIR / "community_summary.csv"
DEFAULT_OUTPUT_COMMENT_FEATURES = PROCESSED_DATA_DIR / "comment_content_features.csv"
DEFAULT_OUTPUT_COMMUNITY_SUMMARY = PROCESSED_DATA_DIR / "community_content_summary.csv"
DEFAULT_OUTPUT_TOP_TERMS = PROCESSED_DATA_DIR / "community_top_terms.csv"
DEFAULT_OUTPUT_ENTITIES = PROCESSED_DATA_DIR / "community_entities.csv"
DEFAULT_OUTPUT_REPRESENTATIVE_COMMENTS = PROCESSED_DATA_DIR / "community_representative_comments.csv"
DEFAULT_SUMMARY_OUTPUT = TABLES_DIR / "content_analysis_summary.csv"
DEFAULT_FIGURES_DIR = FIGURES_DIR
DEFAULT_ENTITY_DICTIONARY_PATH = ROOT_DIR / "config" / "entity_dictionary.csv"
DEFAULT_MIN_COMMUNITY_USERS = 20
DEFAULT_TOP_TERMS = 20
DEFAULT_REPRESENTATIVE_COMMENTS = 5
DEFAULT_MAX_FEATURES = 5000

COMMENT_REQUIRED_COLUMNS = [
    "comment_id",
    "video_id",
    "author_id",
    "original_text",
    "clean_text",
    "like_count",
    "reply_count",
    "text_length",
    "primary_category",
    "video_context_type",
]
COMMUNITY_REQUIRED_COLUMNS = ["author_id", "community_id", "community_label"]
COMMUNITY_SUMMARY_REQUIRED_COLUMNS = [
    "community_id",
    "community_label",
    "users",
    "dominant_primary_category",
    "dominant_video_context_type",
    "top_video_titles",
    "top_channels",
]
ENTITY_DICTIONARY_REQUIRED_COLUMNS = ["pattern", "normalized_entity", "entity_type"]

EMOTION_ORDER = [
    "anger",
    "joy",
    "sadness",
    "fear",
    "surprise",
    "trust",
    "disgust",
    "anticipation",
]

COMMENT_FEATURE_COLUMNS = [
    "comment_id",
    "video_id",
    "author_id",
    "community_id",
    "community_label",
    "clean_text",
    "original_text",
    "like_count",
    "reply_count",
    "text_length",
    "primary_category",
    "video_context_type",
    "vader_compound",
    "vader_label",
    "emotion_top",
    "emotion_anger",
    "emotion_joy",
    "emotion_sadness",
    "emotion_fear",
    "emotion_surprise",
    "emotion_trust",
    "emotion_disgust",
    "emotion_anticipation",
]

COMMUNITY_CONTENT_COLUMNS = [
    "community_id",
    "community_label",
    "users",
    "comments",
    "videos",
    "mean_vader_compound",
    "share_positive",
    "share_neutral",
    "share_negative",
    "dominant_emotion",
    "emotion_anger_share",
    "emotion_joy_share",
    "emotion_sadness_share",
    "emotion_fear_share",
    "emotion_surprise_share",
    "emotion_trust_share",
    "emotion_disgust_share",
    "emotion_anticipation_share",
    "top_terms",
    "top_entities",
    "representative_comments",
]

TOP_TERMS_COLUMNS = [
    "community_id",
    "community_label",
    "term",
    "ngram_type",
    "tfidf_score",
    "raw_count",
    "document_frequency",
    "rank",
]

ENTITY_COLUMNS = [
    "community_id",
    "community_label",
    "entity",
    "normalized_entity",
    "entity_type",
    "mention_count",
    "comments_with_entity",
    "source",
    "rank",
]

REPRESENTATIVE_COMMENT_COLUMNS = [
    "community_id",
    "community_label",
    "comment_id",
    "video_id",
    "clean_text",
    "original_text",
    "like_count",
    "vader_compound",
    "vader_label",
    "emotion_top",
    "selection_reason",
]

SUMMARY_METRIC_ORDER = [
    "sentiment_method",
    "emotion_method",
    "entity_method",
    "full_nel_ned_used",
    "entity_normalization_used",
    "spacy_ner_used",
    "comments_analyzed",
    "main_communities_analyzed",
    "tfidf_terms_generated",
    "entities_extracted",
    "representative_comments_selected",
    "caveats",
]

CUSTOM_FOOTBALL_STOPWORDS = {
    "cup",
    "cups",
    "fifa",
    "football",
    "game",
    "games",
    "goal",
    "goals",
    "highlight",
    "highlights",
    "match",
    "matches",
    "player",
    "players",
    "soccer",
    "team",
    "teams",
    "tournament",
    "video",
    "videos",
    "world",
}

GENERIC_TERM_BLACKLIST = {
    "2026",
    "fifa world",
    "world cup",
}

SPACY_ENTITY_LABELS = {
    "EVENT": "COMPETITION",
    "GPE": "GPE",
    "NORP": "GPE",
    "ORG": "ORG",
    "PERSON": "PERSON",
}

EMOTION_COLORS = {
    "anger": "#B23A48",
    "joy": "#F4A259",
    "sadness": "#4F6D7A",
    "fear": "#7B6D8D",
    "surprise": "#F6BD60",
    "trust": "#5E8C61",
    "disgust": "#6C584C",
    "anticipation": "#5B8E7D",
    "other": "#C7CED6",
}

SENTIMENT_COLORS = {
    "positive": "#5E8C61",
    "neutral": "#C7CED6",
    "negative": "#B23A48",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Profile detected World Cup communities using comment text.")
    parser.add_argument("--comments", default=str(DEFAULT_COMMENTS_PATH.relative_to(ROOT_DIR)), help="Processed comments CSV path.")
    parser.add_argument("--communities", default=str(DEFAULT_COMMUNITIES_PATH.relative_to(ROOT_DIR)), help="User-community CSV path.")
    parser.add_argument(
        "--community-summary",
        default=str(DEFAULT_COMMUNITY_SUMMARY_PATH.relative_to(ROOT_DIR)),
        help="Detected community summary CSV path.",
    )
    parser.add_argument(
        "--output-comment-features",
        default=str(DEFAULT_OUTPUT_COMMENT_FEATURES.relative_to(ROOT_DIR)),
        help="Output CSV for one-row-per-comment content features.",
    )
    parser.add_argument(
        "--output-community-summary",
        default=str(DEFAULT_OUTPUT_COMMUNITY_SUMMARY.relative_to(ROOT_DIR)),
        help="Output CSV for one-row-per-community content summaries.",
    )
    parser.add_argument(
        "--output-top-terms",
        default=str(DEFAULT_OUTPUT_TOP_TERMS.relative_to(ROOT_DIR)),
        help="Output CSV for community TF-IDF terms.",
    )
    parser.add_argument(
        "--output-entities",
        default=str(DEFAULT_OUTPUT_ENTITIES.relative_to(ROOT_DIR)),
        help="Output CSV for aggregated community entities.",
    )
    parser.add_argument(
        "--output-representative-comments",
        default=str(DEFAULT_OUTPUT_REPRESENTATIVE_COMMENTS.relative_to(ROOT_DIR)),
        help="Output CSV for representative community comments.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_OUTPUT.relative_to(ROOT_DIR)),
        help="Compact content-analysis key-value summary CSV path.",
    )
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR.relative_to(ROOT_DIR)), help="Directory for PNG figures.")
    parser.add_argument("--min-community-users", type=int, default=DEFAULT_MIN_COMMUNITY_USERS, help="Minimum users for main interpretive communities.")
    parser.add_argument("--top-terms", type=int, default=DEFAULT_TOP_TERMS, help="Top TF-IDF terms to keep per community.")
    parser.add_argument(
        "--representative-comments-per-community",
        type=int,
        default=DEFAULT_REPRESENTATIVE_COMMENTS,
        help="Representative comments to export per main community.",
    )
    parser.add_argument(
        "--write-detail-tables",
        action="store_true",
        help="Write detailed comment, term, entity, and representative-comment CSV tables in addition to the default summary outputs.",
    )
    parser.add_argument(
        "--debug-nrclex",
        action="store_true",
        help="Print NRCLex API diagnostics for a few sample texts before running the main pipeline.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", help="Overwrite generated outputs. This is the default behavior.")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false", help="Stop instead of overwriting existing generated outputs.")
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    """Resolve a path relative to the repository root."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def display_path(path: Path) -> str:
    """Render a repository-relative path when possible."""
    if path.is_relative_to(ROOT_DIR):
        return str(path.relative_to(ROOT_DIR))
    return str(path)


def print_status(message: str) -> None:
    """Print a compact terminal status message."""
    print(f"[analyze_content] {message}")


def require_columns(dataframe: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Validate that required columns are present."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_label}: {', '.join(missing)}")


def write_csv_atomic(dataframe: pd.DataFrame, output_path: Path, overwrite: bool) -> None:
    """Write a CSV atomically."""
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


def save_figure_atomic(figure: plt.Figure, output_path: Path, overwrite: bool) -> None:
    """Write a PNG figure atomically."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {display_path(output_path)}")

    temp_path = output_path.parent / f".{output_path.stem}.tmp{output_path.suffix}"
    if temp_path.exists():
        temp_path.unlink()

    if output_path.exists():
        print_status(f"Overwriting existing file: {display_path(output_path)}")
    else:
        print_status(f"Writing new file: {display_path(output_path)}")

    figure.savefig(temp_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    temp_path.replace(output_path)


def remove_file_if_exists(path: Path) -> bool:
    """Remove a file if it exists and return whether anything was removed."""
    if path.exists():
        path.unlink()
        return True
    return False


def round_metric(value: Any, digits: int = 4) -> float:
    """Round a numeric value for compact outputs."""
    return round(float(value), digits)


def tokenize_text(text: str) -> list[str]:
    """Tokenize text for lexicon-based methods."""
    return re.findall(r"[A-Za-z][A-Za-z'\-]+", text.lower())


def username_safe_text(text: str) -> str:
    """Mask @mentions in representative comment outputs."""
    return re.sub(r"@\w+", "@user", text)


def normalize_label(value: Any, fallback: str = "unknown") -> str:
    """Convert missing values into a stable string label."""
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def normalize_entity_text(text: str) -> str:
    """Normalize entity surface text while preserving short acronyms."""
    cleaned = re.sub(r"\s+", " ", text.strip(" \t\r\n\"'.,!?;:()[]{}"))
    if not cleaned:
        return ""
    if cleaned.isupper() and len(cleaned) <= 8:
        return cleaned
    parts = []
    for part in cleaned.split(" "):
        parts.append(part if part.isupper() else part.capitalize())
    return " ".join(parts)


def resolve_entity_source(source_values: Iterable[str]) -> str:
    """Collapse entity sources into dictionary, spacy, or dictionary+spacy."""
    parts: set[str] = set()
    for value in source_values:
        if not value:
            continue
        for item in str(value).split("+"):
            if item:
                parts.add(item)
    if parts == {"dictionary", "spacy"}:
        return "dictionary+spacy"
    if "dictionary" in parts and "spacy" in parts:
        return "dictionary+spacy"
    if "dictionary" in parts:
        return "dictionary"
    if "spacy" in parts:
        return "spacy"
    return "dictionary"


def most_common_value(series: pd.Series, fallback: str = "") -> str:
    """Return the most common non-empty value in a series."""
    counts = Counter(normalize_label(value, fallback="") for value in series if normalize_label(value, fallback="") != "")
    if not counts:
        return fallback
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))[0][0]


def compile_phrase_pattern(phrase: str) -> re.Pattern[str]:
    """Compile a case-insensitive boundary-aware phrase pattern."""
    escaped = re.escape(phrase.strip())
    escaped = escaped.replace(r"\ ", r"\s+")
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def normalize_pattern_key(value: str) -> str:
    """Normalize a dictionary alias key for case-insensitive lookups."""
    return re.sub(r"\s+", " ", value.strip().casefold())


def spans_overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    """Return True when two text spans overlap."""
    return left[0] < right[1] and right[0] < left[1]


def load_comments(path: Path) -> pd.DataFrame:
    """Load processed comments used for content analysis."""
    comments_df = pd.read_csv(path)
    require_columns(comments_df, COMMENT_REQUIRED_COLUMNS, display_path(path))
    comments_df["author_id"] = comments_df["author_id"].astype(str)
    comments_df["comment_id"] = comments_df["comment_id"].astype(str)
    comments_df["video_id"] = comments_df["video_id"].astype(str)
    comments_df["clean_text"] = comments_df["clean_text"].fillna("").astype(str)
    comments_df["original_text"] = comments_df["original_text"].fillna("").astype(str)
    comments_df["like_count"] = pd.to_numeric(comments_df["like_count"], errors="coerce").fillna(0).astype(int)
    comments_df["reply_count"] = pd.to_numeric(comments_df["reply_count"], errors="coerce").fillna(0).astype(int)
    comments_df["text_length"] = pd.to_numeric(comments_df["text_length"], errors="coerce").fillna(0).astype(int)
    comments_df["primary_category"] = comments_df["primary_category"].apply(normalize_label)
    comments_df["video_context_type"] = comments_df["video_context_type"].apply(normalize_label)
    return comments_df


def load_user_communities(path: Path) -> pd.DataFrame:
    """Load user-community assignments."""
    communities_df = pd.read_csv(path)
    require_columns(communities_df, COMMUNITY_REQUIRED_COLUMNS, display_path(path))
    communities_df["author_id"] = communities_df["author_id"].astype(str)
    communities_df["community_id"] = pd.to_numeric(communities_df["community_id"], errors="raise").astype(int)
    communities_df["community_label"] = communities_df["community_label"].astype(str)
    return communities_df[COMMUNITY_REQUIRED_COLUMNS].drop_duplicates(subset=["author_id"])


def load_detected_community_summary(path: Path) -> pd.DataFrame:
    """Load the existing structural community summary."""
    summary_df = pd.read_csv(path)
    require_columns(summary_df, COMMUNITY_SUMMARY_REQUIRED_COLUMNS, display_path(path))
    summary_df["community_id"] = pd.to_numeric(summary_df["community_id"], errors="raise").astype(int)
    summary_df["community_label"] = summary_df["community_label"].astype(str)
    summary_df["users"] = pd.to_numeric(summary_df["users"], errors="coerce").fillna(0).astype(int)
    for column in ["dominant_primary_category", "dominant_video_context_type", "top_video_titles", "top_channels"]:
        summary_df[column] = summary_df[column].fillna("").astype(str)
    return summary_df.sort_values(["community_id", "community_label"], kind="stable").reset_index(drop=True)


def load_entity_dictionary(path: Path) -> dict[str, Any]:
    """Load and compile the rule-based entity dictionary plus alias lookups."""
    dictionary_df = pd.read_csv(path)
    require_columns(dictionary_df, ENTITY_DICTIONARY_REQUIRED_COLUMNS, display_path(path))

    compiled_entries: list[dict[str, Any]] = []
    alias_lookup: dict[str, dict[str, str]] = {}
    for row in dictionary_df.itertuples(index=False):
        pattern = normalize_label(row.pattern, fallback="")
        normalized_entity = normalize_entity_text(normalize_label(row.normalized_entity, fallback=""))
        entity_type = normalize_label(row.entity_type, fallback="OTHER").upper()
        if not pattern or not normalized_entity:
            continue
        pattern_key = normalize_pattern_key(pattern)
        compiled_entries.append(
            {
                "pattern": pattern,
                "normalized_entity": normalized_entity,
                "entity_type": entity_type,
                "regex": compile_phrase_pattern(pattern),
                "pattern_length": len(pattern),
                "token_length": len(pattern.split()),
            }
        )
        alias_lookup[pattern_key] = {
            "normalized_entity": normalized_entity,
            "entity_type": entity_type,
        }
    if not compiled_entries:
        raise ValueError(f"Entity dictionary is empty: {display_path(path)}")
    compiled_entries = sorted(
        compiled_entries,
        key=lambda entry: (-entry["pattern_length"], -entry["token_length"], entry["pattern"].casefold()),
    )
    return {"entries": compiled_entries, "alias_lookup": alias_lookup}


def prepare_comments_with_communities(comments_df: pd.DataFrame, communities_df: pd.DataFrame) -> pd.DataFrame:
    """Keep only processed comments written by users assigned to a detected community."""
    merged = comments_df.merge(
        communities_df,
        on="author_id",
        how="inner",
        validate="many_to_one",
    )
    if merged.empty:
        raise ValueError("No processed comments matched detected community assignments.")
    return merged.sort_values(["community_id", "comment_id"], kind="stable").reset_index(drop=True)


def initialize_vader_analyzer() -> tuple[Any, str]:
    """Initialize the vaderSentiment analyzer."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        return SentimentIntensityAnalyzer(), "vaderSentiment VADER"
    except Exception as exc:  # pragma: no cover - import path depends on environment
        raise RuntimeError("Install vaderSentiment with: pip install vaderSentiment") from exc


def vader_label_from_compound(compound: float) -> str:
    """Convert a VADER compound score into a sentiment label."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def compute_vader_features(comments_df: pd.DataFrame, analyzer: Any) -> pd.DataFrame:
    """Score each comment with VADER."""
    compounds = []
    labels = []
    for text in comments_df["clean_text"].astype(str):
        score = analyzer.polarity_scores(text)
        compound = round_metric(score.get("compound", 0.0), digits=4)
        compounds.append(compound)
        labels.append(vader_label_from_compound(compound))

    output_df = comments_df.copy()
    output_df["vader_compound"] = compounds
    output_df["vader_label"] = labels
    return output_df


def tokenize_for_nrclex(text: str) -> list[str]:
    """Tokenize text deterministically for NRCLex."""
    return [token for token in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text.lower()) if token]


def coerce_nrclex_scores(raw_scores: dict[str, Any]) -> dict[str, float]:
    """Map NRCLex scores into the project's eight target emotions."""
    scores = {emotion: 0.0 for emotion in EMOTION_ORDER}
    key_map = {"anticip": "anticipation"}
    for key, value in raw_scores.items():
        normalized_key = key_map.get(str(key).lower(), str(key).lower())
        if normalized_key in scores:
            scores[normalized_key] = float(value)
    return scores


def compute_scores_from_affect_dict(affect_dict: dict[str, list[str]]) -> dict[str, float]:
    """Compute emotion counts manually from NRCLex affect_dict."""
    counts = {emotion: 0.0 for emotion in EMOTION_ORDER}
    for emotions in affect_dict.values():
        for emotion in emotions:
            normalized = "anticipation" if emotion == "anticip" else str(emotion).lower()
            if normalized in counts:
                counts[normalized] += 1.0
    return counts


def inspect_nrclex_backend() -> dict[str, Any]:
    """Inspect the installed NRCLex package and its working API surface."""
    try:
        import nrclex
        from nrclex import NRCLex
    except Exception as exc:  # pragma: no cover - import path depends on environment
        raise RuntimeError(
            "NRCLex is required for emotion analysis. Install it with '.\\.venv\\Scripts\\python -m pip install NRCLex'."
        ) from exc

    try:
        distribution_version = metadata.version("NRCLex")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError(
            "NRCLex is required for emotion analysis, but its package metadata is missing. "
            "Reinstall it with '.\\.venv\\Scripts\\python -m pip install --upgrade --force-reinstall NRCLex'."
        ) from exc

    info: dict[str, Any] = {
        "module_path": getattr(nrclex, "__file__", ""),
        "import_name": NRCLex.__module__,
        "distribution_version": distribution_version,
    }

    probe = NRCLex()
    info["has_load_raw_text"] = hasattr(probe, "load_raw_text")
    info["has_load_token_list"] = hasattr(probe, "load_token_list")
    if not info["has_load_token_list"]:
        raise RuntimeError(
            "Installed NRCLex is incompatible: NRCLex.load_token_list() is missing. "
            "Reinstall NRCLex with '.\\.venv\\Scripts\\python -m pip install --upgrade --force-reinstall NRCLex'."
        )

    sample_tokens = tokenize_for_nrclex("This referee robbed us, disgraceful decision")
    probe.load_token_list(sample_tokens)
    info["attributes_after_load_token_list"] = [name for name in dir(probe) if not name.startswith("__")]
    info["has_raw_emotion_scores"] = hasattr(probe, "raw_emotion_scores")
    info["has_affect_frequencies"] = hasattr(probe, "affect_frequencies")
    info["has_affect_dict"] = hasattr(probe, "affect_dict")
    info["has_top_emotions"] = hasattr(probe, "top_emotions")

    if not any([info["has_raw_emotion_scores"], info["has_affect_frequencies"], info["has_affect_dict"]]):
        raise RuntimeError(
            "Installed NRCLex API is incompatible: no raw_emotion_scores, affect_frequencies, or affect_dict after "
            "load_token_list(). Reinstall NRCLex with '.\\.venv\\Scripts\\python -m pip install --upgrade --force-reinstall NRCLex'."
        )
    return info


def debug_nrclex_samples() -> None:
    """Print a concise NRCLex API diagnostic behind the debug flag."""
    from nrclex import NRCLex

    diagnostics = inspect_nrclex_backend()
    print_status(f"NRCLex import path: {diagnostics['module_path']}")
    print_status(f"NRCLex version: {diagnostics['distribution_version']}")
    print_status(f"NRCLex imported from: {diagnostics['import_name']}")
    print_status(f"NRCLex has load_raw_text: {diagnostics['has_load_raw_text']}")
    print_status(f"NRCLex has load_token_list: {diagnostics['has_load_token_list']}")

    samples = [
        "This referee robbed us, disgraceful decision",
        "Amazing goal, proud of this team",
        "What a shocking and unbelievable match",
    ]
    for sample in samples:
        tokens = tokenize_for_nrclex(sample)
        probe = NRCLex()
        probe.load_token_list(tokens)
        print_status(f"NRCLex sample: {sample}")
        print_status(f"  tokens={tokens}")
        print_status(f"  dir={diagnostics['attributes_after_load_token_list']}")
        print_status(f"  raw_emotion_scores={getattr(probe, 'raw_emotion_scores', 'MISSING')}")
        print_status(f"  affect_frequencies={getattr(probe, 'affect_frequencies', 'MISSING')}")
        print_status(f"  top_emotions={getattr(probe, 'top_emotions', 'MISSING')}")
        print_status(f"  affect_dict={getattr(probe, 'affect_dict', 'MISSING')}")


def initialize_emotion_backend() -> tuple[dict[str, Any], str]:
    """Initialize the required NRCLex emotion backend."""
    diagnostics = inspect_nrclex_backend()
    from nrclex import NRCLex

    return {"backend": "nrclex", "object": NRCLex, "diagnostics": diagnostics}, "NRCLex / NRC Emotion Lexicon"


def pick_top_emotion(counts: dict[str, float]) -> str:
    """Pick the highest-scoring emotion, or none when no emotion word is found."""
    best_emotion = "none"
    best_count = 0.0
    for emotion in EMOTION_ORDER:
        current = float(counts.get(emotion, 0.0))
        if current > best_count:
            best_count = current
            best_emotion = emotion
    return best_emotion if best_count > 0 else "none"


def get_nrclex_emotion_scores(text: str, nrclex_class: Any) -> dict[str, float]:
    """Return NRCLex-based scores for the project's eight emotion categories."""
    tokens = tokenize_for_nrclex(text)
    probe = nrclex_class()
    probe.load_token_list(tokens)

    if hasattr(probe, "raw_emotion_scores"):
        return coerce_nrclex_scores(getattr(probe, "raw_emotion_scores"))
    if hasattr(probe, "affect_frequencies"):
        return coerce_nrclex_scores(getattr(probe, "affect_frequencies"))
    if hasattr(probe, "affect_dict"):
        return compute_scores_from_affect_dict(getattr(probe, "affect_dict"))

    raise RuntimeError(
        "Installed NRCLex API is incompatible after load_token_list(): no raw_emotion_scores, affect_frequencies, or "
        "affect_dict were available. Reinstall NRCLex with '.\\.venv\\Scripts\\python -m pip install --upgrade --force-reinstall NRCLex'."
    )


def compute_emotion_features(comments_df: pd.DataFrame, emotion_backend: dict[str, Any]) -> pd.DataFrame:
    """Compute per-comment emotion features and top emotion labels."""
    output_df = comments_df.copy()
    top_emotions = []

    for emotion in EMOTION_ORDER:
        output_df[f"_emotion_count_{emotion}"] = 0.0
        output_df[f"_emotion_hit_{emotion}"] = 0
        output_df[f"emotion_{emotion}"] = 0.0

    for index, text in output_df["clean_text"].astype(str).items():
        scores = get_nrclex_emotion_scores(text, emotion_backend["object"])
        top_emotions.append(pick_top_emotion(scores))
        for emotion in EMOTION_ORDER:
            value = float(scores[emotion])
            output_df.at[index, f"_emotion_count_{emotion}"] = value
            output_df.at[index, f"_emotion_hit_{emotion}"] = int(value > 0)
            output_df.at[index, f"emotion_{emotion}"] = round_metric(value, digits=6)

    output_df["emotion_top"] = top_emotions
    return output_df


def initialize_spacy_backend() -> tuple[Any | None, str, list[str]]:
    """Initialize optional spaCy NER without failing when it is unavailable."""
    warnings: list[str] = []
    try:
        import spacy
    except Exception as exc:  # pragma: no cover - import path depends on environment
        warnings.append(f"spaCy unavailable ({exc.__class__.__name__}); using dictionary-only entity extraction.")
        return None, "dictionary only", warnings

    try:
        nlp = spacy.load(
            "en_core_web_sm",
            disable=["attribute_ruler", "lemmatizer", "parser", "tagger", "textcat"],
        )
        return nlp, "dictionary + spaCy en_core_web_sm", warnings
    except Exception as exc:  # pragma: no cover - model availability depends on environment
        warnings.append(f"spaCy model en_core_web_sm unavailable ({exc.__class__.__name__}); using dictionary-only entity extraction.")
        return None, "dictionary only", warnings


def normalize_entity_alias(entity_text: str, alias_lookup: dict[str, dict[str, str]]) -> tuple[str, str] | None:
    """Normalize an extracted entity mention through the dictionary alias map."""
    lookup = alias_lookup.get(normalize_pattern_key(entity_text))
    if lookup is None:
        return None
    return lookup["normalized_entity"], lookup["entity_type"]


def extract_dictionary_entities(text: str, compiled_dictionary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract case-insensitive dictionary entities from one comment."""
    records: dict[str, dict[str, Any]] = {}
    occupied_spans: list[tuple[int, int]] = []
    for entry in compiled_dictionary["entries"]:
        matches = list(entry["regex"].finditer(text))
        if not matches:
            continue
        for match in matches:
            span = match.span()
            if any(spans_overlap(span, other_span) for other_span in occupied_spans):
                continue
            occupied_spans.append(span)
            key = entry["normalized_entity"].casefold()
            if key not in records:
                records[key] = {
                    "entity": match.group(0),
                    "normalized_entity": entry["normalized_entity"],
                    "entity_type": entry["entity_type"],
                    "mention_count": 0,
                    "sources": {"dictionary"},
                }
            records[key]["mention_count"] += 1
            records[key]["sources"].add("dictionary")
    return records


def add_spacy_entities_to_record(records: dict[str, dict[str, Any]], doc: Any, alias_lookup: dict[str, dict[str, str]]) -> dict[str, dict[str, Any]]:
    """Add optional spaCy entities while avoiding duplicate counting against the dictionary."""
    if doc is None:
        return records

    for entity in doc.ents:
        if entity.label_ not in SPACY_ENTITY_LABELS:
            continue
        normalized = normalize_entity_text(entity.text)
        if not normalized:
            continue
        alias_hit = normalize_entity_alias(entity.text, alias_lookup)
        entity_type = SPACY_ENTITY_LABELS[entity.label_]
        if alias_hit is not None:
            normalized, entity_type = alias_hit
        key = normalized.casefold()
        if key in records:
            records[key]["sources"].add("spacy")
            if "dictionary" not in records[key]["sources"]:
                records[key]["mention_count"] += 1
            continue

        records[key] = {
            "entity": entity.text,
            "normalized_entity": normalized,
            "entity_type": entity_type,
            "mention_count": 1,
            "sources": {"spacy"},
        }
    return records


def extract_entity_mentions(
    comments_df: pd.DataFrame,
    compiled_dictionary: dict[str, Any],
    spacy_backend: Any | None,
) -> pd.DataFrame:
    """Extract comment-level entity mentions and sources."""
    rows: list[dict[str, Any]] = []
    texts = comments_df["clean_text"].astype(str).tolist()

    if spacy_backend is not None:
        docs = spacy_backend.pipe(texts, batch_size=64)
    else:
        docs = [None] * len(texts)

    for comment_row, doc in zip(comments_df.itertuples(index=False), docs):
        entity_records = extract_dictionary_entities(comment_row.clean_text, compiled_dictionary)
        entity_records = add_spacy_entities_to_record(entity_records, doc, compiled_dictionary["alias_lookup"])
        for record in entity_records.values():
            rows.append(
                {
                    "community_id": int(comment_row.community_id),
                    "community_label": comment_row.community_label,
                    "comment_id": comment_row.comment_id,
                    "entity": normalize_entity_text(record["entity"]),
                    "normalized_entity": record["normalized_entity"],
                    "entity_type": record["entity_type"],
                    "mention_count": int(record["mention_count"]),
                    "source": resolve_entity_source(record["sources"]),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["community_id", "community_label", "comment_id", "entity", "normalized_entity", "entity_type", "mention_count", "source"])

    return pd.DataFrame(rows)


def build_stop_words() -> list[str]:
    """Build the stop-word set used by TF-IDF."""
    return sorted(set(ENGLISH_STOP_WORDS).union(CUSTOM_FOOTBALL_STOPWORDS))


def valid_tfidf_term(term: str) -> bool:
    """Filter out obviously uninformative TF-IDF terms."""
    normalized = term.strip().lower()
    if not normalized:
        return False
    if normalized in GENERIC_TERM_BLACKLIST:
        return False
    if normalized.isdigit():
        return False
    return True


def compute_top_terms_by_community(comments_df: pd.DataFrame, top_terms: int) -> pd.DataFrame:
    """Compute distinctive unigrams and bigrams by community using mean comment-level TF-IDF."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        max_df=0.70,
        max_features=DEFAULT_MAX_FEATURES,
        min_df=3,
        ngram_range=(1, 2),
        stop_words=build_stop_words(),
    )
    tfidf_matrix = vectorizer.fit_transform(comments_df["clean_text"].astype(str))
    terms = np.array(vectorizer.get_feature_names_out())

    count_vectorizer = CountVectorizer(lowercase=True, ngram_range=(1, 2), vocabulary=vectorizer.vocabulary_)
    count_matrix = count_vectorizer.transform(comments_df["clean_text"].astype(str))
    document_frequency = np.asarray((count_matrix > 0).sum(axis=0)).ravel()

    rows: list[dict[str, Any]] = []
    community_groups = comments_df.groupby(["community_id", "community_label"], sort=True).indices
    for (community_id, community_label), row_indexes in community_groups.items():
        row_index_array = np.array(row_indexes)
        mean_scores = np.asarray(tfidf_matrix[row_index_array].mean(axis=0)).ravel()
        raw_counts = np.asarray(count_matrix[row_index_array].sum(axis=0)).ravel()
        candidate_indexes = np.flatnonzero(raw_counts > 0)

        ranked_records: list[dict[str, Any]] = []
        for term_index in candidate_indexes:
            term = str(terms[term_index])
            if not valid_tfidf_term(term):
                continue
            ranked_records.append(
                {
                    "community_id": int(community_id),
                    "community_label": community_label,
                    "term": term,
                    "ngram_type": "bigram" if " " in term else "unigram",
                    "tfidf_score": round_metric(mean_scores[term_index], digits=6),
                    "raw_count": int(raw_counts[term_index]),
                    "document_frequency": int(document_frequency[term_index]),
                }
            )

        ranked_records = sorted(
            ranked_records,
            key=lambda item: (-item["tfidf_score"], -item["raw_count"], item["term"]),
        )[:top_terms]
        for rank, record in enumerate(ranked_records, start=1):
            record["rank"] = rank
            rows.append(record)

    if not rows:
        return pd.DataFrame(columns=TOP_TERMS_COLUMNS)

    return pd.DataFrame(rows)[TOP_TERMS_COLUMNS].sort_values(
        ["community_id", "rank", "term"],
        kind="stable",
    ).reset_index(drop=True)


def aggregate_community_entities(entity_mentions_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate entity mentions to the community level."""
    if entity_mentions_df.empty:
        return pd.DataFrame(columns=ENTITY_COLUMNS)

    grouped = (
        entity_mentions_df.groupby(["community_id", "community_label", "normalized_entity"], as_index=False)
        .agg(
            entity=("entity", most_common_value),
            entity_type=("entity_type", most_common_value),
            mention_count=("mention_count", "sum"),
            comments_with_entity=("comment_id", "nunique"),
            source=("source", lambda values: resolve_entity_source(values)),
        )
        .sort_values(
            ["community_id", "comments_with_entity", "mention_count", "normalized_entity"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    grouped["rank"] = grouped.groupby("community_id").cumcount() + 1
    return grouped[ENTITY_COLUMNS]


def join_preview_values(
    dataframe: pd.DataFrame,
    value_column: str,
    rank_limit: int,
    output_column: str,
    separator: str = "; ",
) -> pd.DataFrame:
    """Create preview strings such as top terms or top entities."""
    if dataframe.empty:
        return pd.DataFrame(columns=["community_id", output_column])
    preview_df = dataframe[dataframe["rank"] <= rank_limit].copy()
    preview_df = (
        preview_df.groupby("community_id", as_index=False)[value_column]
        .agg(lambda values: separator.join(str(value) for value in values))
        .rename(columns={value_column: output_column})
    )
    return preview_df


def determine_dominant_emotion(community_comments_df: pd.DataFrame) -> str:
    """Determine the dominant emotion from comment-level top-emotion shares."""
    if community_comments_df.empty:
        return "none"
    shares = {emotion: float((community_comments_df["emotion_top"] == emotion).mean()) for emotion in EMOTION_ORDER}
    best_emotion = max(EMOTION_ORDER, key=lambda emotion: shares[emotion])
    return best_emotion if shares[best_emotion] > 0 else "none"


def infer_preliminary_narrative_label(row: pd.Series, metadata_lookup: dict[int, dict[str, str]]) -> str:
    """Assign a cautious preliminary label from terms, entities, and community metadata."""
    metadata = metadata_lookup.get(int(row["community_id"]), {})
    dominant_context = normalize_label(metadata.get("dominant_video_context_type", ""), fallback="").lower()
    dominant_category = normalize_label(metadata.get("dominant_primary_category", ""), fallback="").lower()
    evidence = " ".join(
        [
            dominant_context,
            dominant_category,
            normalize_label(metadata.get("top_video_titles", ""), fallback=""),
            normalize_label(row.get("top_terms_preview", ""), fallback=""),
            normalize_label(row.get("top_entities_preview", ""), fallback=""),
        ]
    ).lower()

    controversy_terms = ["var", "referee", "offside", "penalty", "disallowed goal", "red card", "handball"]
    reaction_terms = ["ronaldo", "messi", "mbappe", "kane", "legacy", "goat"]
    tactical_terms = ["manager", "clarke", "ordinary", "disaster", "blame", "defence", "defense"]

    if dominant_context == "var_referee_controversy" or "var_referee_controvers" in dominant_category:
        return "VAR/referee controversy"
    if dominant_context == "tactical_analysis" or "tactical_blame" in dominant_category:
        return "tactical blame / national-team criticism"
    if dominant_context == "upset_underdog" or "upsets_underdogs" in dominant_category:
        return "upset / underdog reaction"
    if dominant_context in {"post_match_reaction", "star_player_legacy"} or dominant_category in {"post_match_reactions", "star_player_legacy"}:
        return "post-match reaction / star-player debate"
    if dominant_context == "match_highlights" or dominant_category == "match_highlights":
        return "match highlights / general tournament viewing"
    if any(term in evidence for term in controversy_terms):
        return "VAR/referee controversy"
    if "tactical" in evidence or any(term in evidence for term in tactical_terms):
        return "tactical blame / national-team criticism"
    if "underdog" in evidence or any(term in evidence for term in ["upset", "shock", "surprise win", "penalties exit"]):
        return "upset / underdog reaction"
    if "reaction" in evidence or "star_player" in evidence or any(term in evidence for term in reaction_terms):
        return "post-match reaction / star-player debate"
    if "highlight" in evidence or "match_highlights" in evidence:
        return "match highlights / general tournament viewing"
    return "mixed discussion community"


def build_representative_comments_preview(representative_comments_df: pd.DataFrame) -> pd.DataFrame:
    """Build a compact representative-comment preview per community."""
    if representative_comments_df.empty:
        return pd.DataFrame(columns=["community_id", "representative_comments"])
    preview_df = representative_comments_df.copy()
    preview_df["comment_preview"] = preview_df["clean_text"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    preview_df = (
        preview_df.groupby("community_id", as_index=False)["comment_preview"]
        .agg(lambda values: " || ".join(str(value) for value in values))
        .rename(columns={"comment_preview": "representative_comments"})
    )
    return preview_df


def build_community_content_summary(
    comment_features_df: pd.DataFrame,
    detected_summary_df: pd.DataFrame,
    top_terms_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    representative_comments_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build one content summary row per community."""
    user_lookup = dict(detected_summary_df[["community_id", "users"]].itertuples(index=False, name=None))

    rows: list[dict[str, Any]] = []
    grouped = comment_features_df.groupby(["community_id", "community_label"], sort=True)
    for (community_id, community_label), community_comments_df in grouped:
        comments_count = len(community_comments_df)
        row = {
            "community_id": int(community_id),
            "community_label": community_label,
            "users": int(user_lookup.get(int(community_id), 0)),
            "comments": comments_count,
            "videos": int(community_comments_df["video_id"].nunique()),
            "mean_vader_compound": round_metric(community_comments_df["vader_compound"].mean()) if comments_count else 0.0,
            "share_positive": round_metric((community_comments_df["vader_label"] == "positive").mean()) if comments_count else 0.0,
            "share_neutral": round_metric((community_comments_df["vader_label"] == "neutral").mean()) if comments_count else 0.0,
            "share_negative": round_metric((community_comments_df["vader_label"] == "negative").mean()) if comments_count else 0.0,
            "dominant_emotion": determine_dominant_emotion(community_comments_df),
        }

        for emotion in EMOTION_ORDER:
            row[f"emotion_{emotion}_share"] = round_metric((community_comments_df["emotion_top"] == emotion).mean()) if comments_count else 0.0
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    terms_preview_df = join_preview_values(top_terms_df, value_column="term", rank_limit=5, output_column="top_terms")
    entities_preview_df = join_preview_values(
        entities_df,
        value_column="normalized_entity",
        rank_limit=5,
        output_column="top_entities",
    )
    representative_preview_df = build_representative_comments_preview(representative_comments_df)
    summary_df = summary_df.merge(terms_preview_df, on="community_id", how="left")
    summary_df = summary_df.merge(entities_preview_df, on="community_id", how="left")
    summary_df = summary_df.merge(representative_preview_df, on="community_id", how="left")
    summary_df["top_terms"] = summary_df["top_terms"].fillna("")
    summary_df["top_entities"] = summary_df["top_entities"].fillna("")
    summary_df["representative_comments"] = summary_df["representative_comments"].fillna("")
    return summary_df[COMMUNITY_CONTENT_COLUMNS].sort_values(["community_id", "community_label"], kind="stable").reset_index(drop=True)


def build_term_pattern_cache(top_terms_df: pd.DataFrame, rank_limit: int = 10) -> dict[int, list[tuple[str, re.Pattern[str]]]]:
    """Compile top terms into regex patterns for representative comment scoring."""
    cache: dict[int, list[tuple[str, re.Pattern[str]]]] = defaultdict(list)
    for row in top_terms_df[top_terms_df["rank"] <= rank_limit].itertuples(index=False):
        cache[int(row.community_id)].append((row.term, compile_phrase_pattern(row.term)))
    return dict(cache)


def build_comment_entity_lookup(entity_mentions_df: pd.DataFrame) -> dict[str, set[str]]:
    """Map each comment to the normalized entities it contains."""
    if entity_mentions_df.empty or "comment_id" not in entity_mentions_df.columns or "normalized_entity" not in entity_mentions_df.columns:
        return {}
    grouped = entity_mentions_df.groupby("comment_id")["normalized_entity"].agg(lambda values: set(values))
    return grouped.to_dict()


def select_representative_comments(
    comment_features_df: pd.DataFrame,
    top_terms_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    main_community_ids: list[int],
    comments_per_community: int,
) -> pd.DataFrame:
    """Select representative comments for the main interpretive communities."""
    if not main_community_ids:
        return pd.DataFrame(columns=REPRESENTATIVE_COMMENT_COLUMNS)

    term_pattern_cache = build_term_pattern_cache(top_terms_df, rank_limit=10)
    entity_mentions_by_comment = build_comment_entity_lookup(entities_df if "comment_id" in entities_df.columns else pd.DataFrame())

    selected_rows: list[dict[str, Any]] = []
    for community_id in main_community_ids:
        community_df = comment_features_df[comment_features_df["community_id"] == community_id].copy()
        community_df = community_df[(community_df["text_length"] >= 40) & (community_df["text_length"] <= 280)].copy()
        community_df = community_df.drop_duplicates(subset=["clean_text"], keep="first").copy()
        if community_df.empty:
            continue

        compiled_terms = term_pattern_cache.get(community_id, [])
        like_threshold = float(community_df["like_count"].quantile(0.75)) if len(community_df) >= 4 else float(community_df["like_count"].median())
        candidate_records: list[dict[str, Any]] = []

        for row in community_df.itertuples(index=False):
            matched_terms = [term for term, pattern in compiled_terms if pattern.search(row.clean_text)]
            matched_entities = sorted(entity_mentions_by_comment.get(row.comment_id, set()))
            length_balance = max(0.0, 1.0 - (abs(int(row.text_length) - 140) / 140))
            score = (
                (2.5 * len(matched_terms))
                + (1.75 * len(matched_entities))
                + math.log1p(max(int(row.like_count), 0))
                + length_balance
            )
            candidate_records.append(
                {
                    "community_id": int(row.community_id),
                    "community_label": row.community_label,
                    "comment_id": row.comment_id,
                    "video_id": row.video_id,
                    "clean_text": row.clean_text,
                    "original_text": row.original_text,
                    "like_count": int(row.like_count),
                    "text_length": int(row.text_length),
                    "vader_compound": float(row.vader_compound),
                    "vader_label": row.vader_label,
                    "emotion_top": row.emotion_top,
                    "term_hits": len(matched_terms),
                    "entity_hits": len(matched_entities),
                    "high_like": bool(int(row.like_count) >= like_threshold),
                    "score": score,
                }
            )

        candidate_df = pd.DataFrame(candidate_records).sort_values(
            ["score", "like_count", "text_length"],
            ascending=[False, False, True],
            kind="stable",
        )
        if candidate_df.empty:
            continue

        used_comment_ids: set[str] = set()
        used_videos: set[str] = set()
        used_sentiments: set[str] = set()
        used_emotions: set[str] = set()

        def append_selected(record: pd.Series, diversity_reason: str | None = None) -> None:
            reasons = []
            if record["term_hits"] > 0:
                reasons.append("contains_top_term")
            if record["entity_hits"] > 0:
                reasons.append("contains_entity")
            if bool(record["high_like"]):
                reasons.append("high_like_count")
            if diversity_reason:
                reasons.append(diversity_reason)
            if not reasons:
                reasons.append("balanced_length")

            selected_rows.append(
                {
                    "community_id": int(record["community_id"]),
                    "community_label": record["community_label"],
                    "comment_id": record["comment_id"],
                    "video_id": record["video_id"],
                    "clean_text": username_safe_text(record["clean_text"]),
                    "original_text": username_safe_text(record["original_text"]),
                    "like_count": int(record["like_count"]),
                    "vader_compound": round_metric(record["vader_compound"]),
                    "vader_label": record["vader_label"],
                    "emotion_top": record["emotion_top"],
                    "selection_reason": "; ".join(reasons),
                }
            )
            used_comment_ids.add(record["comment_id"])
            used_videos.add(record["video_id"])
            used_sentiments.add(record["vader_label"])
            if record["emotion_top"] != "none":
                used_emotions.add(record["emotion_top"])

        for _, record in candidate_df.iterrows():
            if len([row for row in selected_rows if row["community_id"] == community_id]) >= comments_per_community:
                break
            if record["comment_id"] in used_comment_ids:
                continue
            if record["video_id"] not in used_videos or record["vader_label"] not in used_sentiments:
                append_selected(record, diversity_reason="adds_discussion_variety")

        for _, record in candidate_df.iterrows():
            if len([row for row in selected_rows if row["community_id"] == community_id]) >= comments_per_community:
                break
            if record["comment_id"] in used_comment_ids:
                continue
            if record["emotion_top"] != "none" and record["emotion_top"] not in used_emotions:
                append_selected(record, diversity_reason="adds_emotion_variety")

        for _, record in candidate_df.iterrows():
            if len([row for row in selected_rows if row["community_id"] == community_id]) >= comments_per_community:
                break
            if record["comment_id"] in used_comment_ids:
                continue
            append_selected(record)

    if not selected_rows:
        return pd.DataFrame(columns=REPRESENTATIVE_COMMENT_COLUMNS)

    selected_df = pd.DataFrame(selected_rows)
    return selected_df[REPRESENTATIVE_COMMENT_COLUMNS].sort_values(
        ["community_id", "like_count", "comment_id"],
        ascending=[True, False, True],
        kind="stable",
    ).reset_index(drop=True)


def build_summary_table(
    comment_features_df: pd.DataFrame,
    main_community_ids: list[int],
    top_terms_df: pd.DataFrame,
    entities_df: pd.DataFrame,
    representative_comments_df: pd.DataFrame,
    sentiment_method: str,
    emotion_method: str,
    entity_method: str,
    spacy_ner_used: bool,
    runtime_warnings: list[str],
) -> pd.DataFrame:
    """Build a compact key-value summary table."""
    caveats = (
        "Sentiment and emotion scores are lexicon-based and approximate; sarcasm, football slang, and imperfect entity matching remain possible."
    )
    if runtime_warnings:
        caveats = caveats + " Runtime warnings: " + " | ".join(runtime_warnings)

    metrics = {
        "sentiment_method": sentiment_method,
        "emotion_method": emotion_method,
        "entity_method": entity_method,
        "full_nel_ned_used": "false",
        "entity_normalization_used": "true",
        "spacy_ner_used": str(spacy_ner_used).lower(),
        "comments_analyzed": len(comment_features_df),
        "main_communities_analyzed": len(main_community_ids),
        "tfidf_terms_generated": len(top_terms_df),
        "entities_extracted": len(entities_df),
        "representative_comments_selected": len(representative_comments_df),
        "caveats": caveats,
    }
    return pd.DataFrame([{"metric": metric, "value": metrics[metric]} for metric in SUMMARY_METRIC_ORDER])


def main_communities_from_summary(summary_df: pd.DataFrame, min_community_users: int) -> list[int]:
    """Return the community IDs used for main interpretation."""
    main_df = summary_df[summary_df["users"] >= min_community_users].copy()
    if main_df.empty:
        main_df = summary_df.head(min(6, len(summary_df))).copy()
    return main_df["community_id"].astype(int).tolist()


def apply_matplotlib_style() -> None:
    """Apply a clean matplotlib style for report-ready PNGs."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "axes.edgecolor": "#D1D5DB",
            "axes.labelcolor": "#111827",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "font.size": 11,
            "savefig.facecolor": "white",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
        }
    )


def add_title_and_subtitle(axis: plt.Axes, title: str, subtitle: str) -> None:
    """Add a left-aligned title and subtitle."""
    axis.set_title(title, loc="left", pad=24)
    axis.text(0.0, 1.03, subtitle, transform=axis.transAxes, ha="left", va="bottom", fontsize=10, color="#4B5563")


def clean_axis(axis: plt.Axes, show_y_grid: bool = True) -> None:
    """Reduce chart clutter while keeping guides readable."""
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, alpha=0.8) if show_y_grid else axis.grid(False, axis="y")
    axis.grid(False, axis="x")


def wrap_axis_label(text: str, width: int = 16) -> str:
    """Wrap a long x-axis label."""
    return "\n".join(textwrap.wrap(text, width=width)) if len(text) > width else text


def create_sentiment_profile_figure(community_content_df: pd.DataFrame, main_community_ids: list[int]) -> plt.Figure:
    """Create a 100% stacked community sentiment profile figure."""
    chart_df = community_content_df[community_content_df["community_id"].isin(main_community_ids)].copy()
    chart_df = chart_df.sort_values("community_id", kind="stable")
    labels = chart_df["community_label"].tolist()

    figure, axis = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(chart_df))
    for sentiment in ["positive", "neutral", "negative"]:
        values = chart_df[f"share_{sentiment}"].to_numpy(dtype=float)
        axis.bar(
            labels,
            values,
            bottom=bottom,
            color=SENTIMENT_COLORS[sentiment],
            edgecolor="white",
            linewidth=0.6,
            width=0.72,
            label=sentiment.title(),
        )
        bottom += values

    add_title_and_subtitle(
        axis,
        "Community Sentiment Profile",
        "Share of positive, neutral, and negative comments in the six main participation communities",
    )
    axis.set_xlabel("Community")
    axis.set_ylabel("Share of comments")
    axis.set_ylim(0, 1.0)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], labels=["0%", "25%", "50%", "75%", "100%"])
    clean_axis(axis, show_y_grid=True)
    axis.legend(frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    figure.tight_layout()
    return figure


def build_emotion_profile_for_plot(comment_features_df: pd.DataFrame, main_community_ids: list[int]) -> tuple[pd.DataFrame, list[str]]:
    """Build a normalized mean emotion profile for stacked plotting."""
    chart_df = (
        comment_features_df[comment_features_df["community_id"].isin(main_community_ids)]
        .groupby(["community_id", "community_label"], as_index=False)
        .agg({f"emotion_{emotion}": "mean" for emotion in EMOTION_ORDER})
        .sort_values("community_id", kind="stable")
    )

    overall = {emotion: float(chart_df[f"emotion_{emotion}"].sum()) for emotion in EMOTION_ORDER}
    ordered_emotions = sorted(EMOTION_ORDER, key=lambda emotion: (-overall[emotion], EMOTION_ORDER.index(emotion)))
    kept_emotions = ordered_emotions[:6]
    other_emotions = [emotion for emotion in EMOTION_ORDER if emotion not in kept_emotions]

    if other_emotions:
        chart_df["emotion_other"] = chart_df[[f"emotion_{emotion}" for emotion in other_emotions]].sum(axis=1)
        kept_columns = [f"emotion_{emotion}" for emotion in kept_emotions] + ["emotion_other"]
        emotion_labels = kept_emotions + ["other"]
    else:
        kept_columns = [f"emotion_{emotion}" for emotion in kept_emotions]
        emotion_labels = kept_emotions

    row_totals = chart_df[kept_columns].sum(axis=1).replace(0, np.nan)
    chart_df[kept_columns] = chart_df[kept_columns].div(row_totals, axis=0).fillna(0.0)
    return chart_df, emotion_labels


def create_emotion_profile_figure(comment_features_df: pd.DataFrame, main_community_ids: list[int]) -> plt.Figure:
    """Create a normalized mean emotion profile figure."""
    chart_df, emotion_labels = build_emotion_profile_for_plot(comment_features_df, main_community_ids)

    figure, axis = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(chart_df))
    for emotion in emotion_labels:
        column = f"emotion_{emotion}"
        values = chart_df[column].to_numpy(dtype=float)
        axis.bar(
            chart_df["community_label"].tolist(),
            values,
            bottom=bottom,
            color=EMOTION_COLORS[emotion],
            edgecolor="white",
            linewidth=0.6,
            width=0.72,
            label=emotion.title(),
        )
        bottom += values

    add_title_and_subtitle(
        axis,
        "Community Emotion Profile",
        "Normalized mean emotion scores by community; less frequent emotions are grouped into Other when needed",
    )
    axis.set_xlabel("Community")
    axis.set_ylabel("Normalized emotion profile")
    axis.set_ylim(0, 1.0)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], labels=["0%", "25%", "50%", "75%", "100%"])
    clean_axis(axis, show_y_grid=True)
    axis.legend(frameon=False, bbox_to_anchor=(1.01, 1.0), loc="upper left")
    figure.tight_layout()
    return figure


def create_top_terms_figure(top_terms_df: pd.DataFrame, main_community_ids: list[int]) -> plt.Figure:
    """Create small multiples of top TF-IDF terms for the main communities."""
    main_terms_df = top_terms_df[(top_terms_df["community_id"].isin(main_community_ids)) & (top_terms_df["rank"] <= 8)].copy()
    communities = sorted(main_terms_df["community_id"].unique())
    labels = {
        int(row.community_id): row.community_label
        for row in main_terms_df[["community_id", "community_label"]].drop_duplicates().itertuples(index=False)
    }
    n_panels = max(len(communities), 1)
    ncols = 3 if n_panels > 4 else 2
    nrows = math.ceil(n_panels / ncols)
    figure, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 4.5 * nrows))
    axes_array = np.atleast_1d(axes).ravel()

    for axis in axes_array[n_panels:]:
        axis.set_visible(False)

    for axis, community_id in zip(axes_array, communities):
        subset = main_terms_df[main_terms_df["community_id"] == community_id].sort_values("tfidf_score", kind="stable").copy()
        axis.barh(subset["term"], subset["tfidf_score"], color="#496A81")
        axis.set_title(labels.get(community_id, f"C{community_id}"), loc="left")
        axis.set_xlabel("Mean TF-IDF")
        clean_axis(axis, show_y_grid=False)
        axis.grid(axis="x", color="#E5E7EB", linewidth=0.8, alpha=0.8)

    figure.suptitle("Community Top Terms", x=0.06, ha="left", y=0.99, fontsize=16, fontweight="bold")
    figure.text(
        0.06,
        0.965,
        "Top eight distinctive TF-IDF unigrams and bigrams for each main community",
        ha="left",
        va="top",
        fontsize=10,
        color="#4B5563",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.95])
    return figure


def create_entity_profile_figure(
    entities_df: pd.DataFrame,
    community_content_df: pd.DataFrame,
    main_community_ids: list[int],
) -> plt.Figure:
    """Create a heatmap of normalized entity intensity across the main communities."""
    main_entities_df = entities_df[entities_df["community_id"].isin(main_community_ids)].copy()
    if main_entities_df.empty:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.text(0.5, 0.5, "No entity mentions available", ha="center", va="center")
        axis.set_axis_off()
        return figure

    entity_order = (
        main_entities_df.groupby("normalized_entity", as_index=False)
        .agg(total_comments_with_entity=("comments_with_entity", "sum"), total_mentions=("mention_count", "sum"))
        .sort_values(["total_comments_with_entity", "total_mentions", "normalized_entity"], ascending=[False, False, True], kind="stable")
        .head(18)["normalized_entity"]
        .tolist()
    )
    community_order_df = community_content_df[community_content_df["community_id"].isin(main_community_ids)].sort_values("community_id", kind="stable")
    community_labels = community_order_df["community_label"].tolist()
    comments_lookup = dict(community_order_df[["community_id", "comments"]].itertuples(index=False, name=None))

    matrix = np.zeros((len(community_order_df), len(entity_order)))
    lookup = {
        (int(row.community_id), row.normalized_entity): float(row.mention_count)
        for row in main_entities_df.itertuples(index=False)
    }
    for row_index, community_id in enumerate(community_order_df["community_id"].tolist()):
        denominator = max(int(comments_lookup.get(int(community_id), 0)), 1)
        for column_index, entity in enumerate(entity_order):
            matrix[row_index, column_index] = (lookup.get((int(community_id), entity), 0.0) / denominator) * 100.0

    figure, axis = plt.subplots(figsize=(16, 6))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    add_title_and_subtitle(
        axis,
        "Community Entity Profile",
        "Normalized entity mention intensity per 100 comments across the six main communities",
    )
    axis.set_xticks(np.arange(len(entity_order)))
    axis.set_xticklabels([wrap_axis_label(entity, width=12) for entity in entity_order], rotation=45, ha="right")
    axis.set_yticks(np.arange(len(community_labels)))
    axis.set_yticklabels(community_labels)
    axis.set_xlabel("Entity")
    axis.set_ylabel("Community")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.03, pad=0.02)
    colorbar.set_label("Mentions per 100 comments")
    figure.tight_layout()
    return figure


def cleanup_stale_detail_tables(detail_paths: list[Path]) -> list[str]:
    """Remove stale default detail tables when they are no longer written by default."""
    removed: list[str] = []
    for path in detail_paths:
        if remove_file_if_exists(path):
            removed.append(display_path(path))
    return removed


def main() -> None:
    """Run the content-analysis CLI."""
    args = parse_args()
    if args.min_community_users < 1:
        raise ValueError("Minimum community users must be a positive integer.")
    if args.top_terms < 1:
        raise ValueError("Top terms must be a positive integer.")
    if args.representative_comments_per_community < 1:
        raise ValueError("Representative comments per community must be a positive integer.")

    apply_matplotlib_style()

    comments_path = resolve_path(args.comments)
    communities_path = resolve_path(args.communities)
    community_summary_path = resolve_path(args.community_summary)
    output_comment_features_path = resolve_path(args.output_comment_features)
    output_community_summary_path = resolve_path(args.output_community_summary)
    output_top_terms_path = resolve_path(args.output_top_terms)
    output_entities_path = resolve_path(args.output_entities)
    output_representative_comments_path = resolve_path(args.output_representative_comments)
    summary_output_path = resolve_path(args.summary_output)
    figures_dir = resolve_path(args.figures_dir)

    required_paths = [
        comments_path,
        communities_path,
        community_summary_path,
        DEFAULT_ENTITY_DICTIONARY_PATH,
    ]

    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required file: {display_path(required_path)}")

    comments_df = load_comments(comments_path)
    communities_df = load_user_communities(communities_path)
    detected_summary_df = load_detected_community_summary(community_summary_path)
    comments_with_communities_df = prepare_comments_with_communities(comments_df, communities_df)
    print_status(
        f"Loaded {len(comments_with_communities_df)} processed comments written by users assigned to detected communities."
    )

    if args.debug_nrclex:
        debug_nrclex_samples()

    vader_analyzer, sentiment_method = initialize_vader_analyzer()
    comment_features_df = compute_vader_features(comments_with_communities_df, vader_analyzer)

    emotion_backend, emotion_method = initialize_emotion_backend()
    comment_features_df = compute_emotion_features(comment_features_df, emotion_backend)

    compiled_dictionary = load_entity_dictionary(DEFAULT_ENTITY_DICTIONARY_PATH)
    spacy_backend, spacy_method, spacy_warnings = initialize_spacy_backend()
    spacy_ner_used = spacy_backend is not None
    entity_mentions_df = extract_entity_mentions(comment_features_df, compiled_dictionary, spacy_backend)
    community_entities_df = aggregate_community_entities(entity_mentions_df)

    top_terms_df = compute_top_terms_by_community(comment_features_df, top_terms=args.top_terms)
    main_community_ids = main_communities_from_summary(detected_summary_df, min_community_users=args.min_community_users)

    representative_comments_df = select_representative_comments(
        comment_features_df=comment_features_df,
        top_terms_df=top_terms_df,
        entities_df=entity_mentions_df,
        main_community_ids=main_community_ids,
        comments_per_community=args.representative_comments_per_community,
    )
    community_content_df = build_community_content_summary(
        comment_features_df=comment_features_df,
        detected_summary_df=detected_summary_df,
        top_terms_df=top_terms_df,
        entities_df=community_entities_df,
        representative_comments_df=representative_comments_df,
    )
    main_community_ids = main_communities_from_summary(community_content_df, min_community_users=args.min_community_users)

    runtime_warnings = spacy_warnings
    summary_table_df = build_summary_table(
        comment_features_df=comment_features_df,
        main_community_ids=main_community_ids,
        top_terms_df=top_terms_df,
        entities_df=community_entities_df,
        representative_comments_df=representative_comments_df,
        sentiment_method=sentiment_method,
        emotion_method=emotion_method,
        entity_method="dictionary normalization + optional spaCy NER",
        spacy_ner_used=spacy_ner_used,
        runtime_warnings=runtime_warnings,
    )

    comment_feature_output_df = comment_features_df[COMMENT_FEATURE_COLUMNS].copy().sort_values(
        ["community_id", "comment_id"],
        kind="stable",
    )
    write_csv_atomic(community_content_df, output_community_summary_path, overwrite=args.overwrite)
    write_csv_atomic(summary_table_df, summary_output_path, overwrite=args.overwrite)

    default_detail_output_paths = [
        DEFAULT_OUTPUT_COMMENT_FEATURES,
        DEFAULT_OUTPUT_TOP_TERMS,
        DEFAULT_OUTPUT_ENTITIES,
        DEFAULT_OUTPUT_REPRESENTATIVE_COMMENTS,
    ]
    if args.write_detail_tables:
        write_csv_atomic(comment_feature_output_df, output_comment_features_path, overwrite=args.overwrite)
        write_csv_atomic(top_terms_df, output_top_terms_path, overwrite=args.overwrite)
        write_csv_atomic(community_entities_df, output_entities_path, overwrite=args.overwrite)
        write_csv_atomic(representative_comments_df, output_representative_comments_path, overwrite=args.overwrite)
    else:
        removed_detail_paths = cleanup_stale_detail_tables(default_detail_output_paths)
        for removed_path in removed_detail_paths:
            print_status(f"Removed stale detail table: {removed_path}")

    save_figure_atomic(
        create_sentiment_profile_figure(community_content_df, main_community_ids),
        figures_dir / "community_sentiment_profile.png",
        overwrite=args.overwrite,
    )
    save_figure_atomic(
        create_emotion_profile_figure(comment_features_df, main_community_ids),
        figures_dir / "community_emotion_profile.png",
        overwrite=args.overwrite,
    )
    save_figure_atomic(
        create_top_terms_figure(top_terms_df, main_community_ids),
        figures_dir / "community_top_terms.png",
        overwrite=args.overwrite,
    )
    save_figure_atomic(
        create_entity_profile_figure(community_entities_df, community_content_df, main_community_ids),
        figures_dir / "community_entity_profile.png",
        overwrite=args.overwrite,
    )

    for warning in runtime_warnings:
        print_status(f"WARNING: {warning}")

    print_status(f"Sentiment backend: {sentiment_method}")
    print_status(f"Emotion backend: {emotion_method}")
    print_status(f"Entity method: dictionary normalization + optional spaCy NER ({spacy_method})")
    print_status(f"spaCy NER used: {str(spacy_ner_used).lower()}")
    print_status(f"Entity normalization used: true")
    print_status("Full NEL/NED used: false")
    print_status(f"Detail tables written: {str(args.write_detail_tables).lower()}")
    print_status(
        f"Generated content analysis for {len(comment_feature_output_df)} comments across {len(main_community_ids)} main communities."
    )
    print_status(
        f"Top terms rows: {len(top_terms_df)} | entity rows: {len(community_entities_df)} | representative comments: {len(representative_comments_df)}."
    )


if __name__ == "__main__":
    main()
