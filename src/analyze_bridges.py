"""Identify bridge users, videos, and entities across detected communities."""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import networkx as nx
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import FIGURES_DIR, PROCESSED_DATA_DIR, ROOT_DIR, TABLES_DIR


DEFAULT_EDGES_PATH = PROCESSED_DATA_DIR / "user_network_edges.csv"
DEFAULT_NODES_PATH = PROCESSED_DATA_DIR / "user_network_nodes.csv"
DEFAULT_USER_VIDEO_PATH = PROCESSED_DATA_DIR / "user_video_edges.csv"
DEFAULT_COMMENTS_PATH = PROCESSED_DATA_DIR / "comments_processed.csv"
DEFAULT_COMMUNITIES_PATH = PROCESSED_DATA_DIR / "user_communities.csv"
DEFAULT_COMMUNITY_SUMMARY_PATH = PROCESSED_DATA_DIR / "community_summary.csv"
DEFAULT_CONTENT_SUMMARY_PATH = PROCESSED_DATA_DIR / "community_content_summary.csv"
DEFAULT_ENTITY_DICTIONARY_PATH = ROOT_DIR / "config" / "entity_dictionary.csv"
DEFAULT_OUTPUT_USERS = PROCESSED_DATA_DIR / "bridge_users.csv"
DEFAULT_OUTPUT_VIDEOS = PROCESSED_DATA_DIR / "bridge_videos.csv"
DEFAULT_OUTPUT_ENTITIES = PROCESSED_DATA_DIR / "bridge_entities.csv"
DEFAULT_SUMMARY_OUTPUT = TABLES_DIR / "bridge_analysis_summary.csv"
DEFAULT_THRESHOLD = 2
DEFAULT_MIN_MAIN_COMMUNITY_USERS = 20
DEFAULT_TOP_N = 10
DEFAULT_SEED = 42

EDGE_REQUIRED_COLUMNS = ["source", "target", "weight", "shared_videos"]
NODE_REQUIRED_COLUMNS = [
    "author_id",
    "total_comments",
    "videos_commented",
    "dominant_primary_category",
    "dominant_video_context_type",
]
USER_VIDEO_REQUIRED_COLUMNS = [
    "author_id",
    "video_id",
    "comment_count",
    "title",
    "channel_title",
    "primary_category",
    "video_context_type",
]
COMMENT_REQUIRED_COLUMNS = ["comment_id", "author_id", "clean_text"]
COMMUNITY_REQUIRED_COLUMNS = ["author_id", "community_id", "community_label"]
COMMUNITY_SUMMARY_REQUIRED_COLUMNS = ["community_id", "community_label", "users"]
ENTITY_DICTIONARY_REQUIRED_COLUMNS = ["pattern", "normalized_entity", "entity_type"]

SUMMARY_METRIC_ORDER = [
    "edge_threshold_used",
    "users_in_bridge_graph",
    "edges_in_bridge_graph",
    "bridge_users_computed",
    "bridge_videos_computed",
    "bridge_entities_computed",
    "top_bridge_user_id",
    "top_bridge_user_community",
    "top_bridge_video_title",
    "top_bridge_video_score",
    "top_bridge_entity",
    "top_bridge_entity_score",
    "bridge_user_method",
    "bridge_video_method",
    "bridge_entity_method",
    "bridge_entity_score_formula",
    "caveats",
]

BAR_COLOR = "#496A81"
ACCENT_COLOR = "#D08C60"
GRID_COLOR = "#D7DEE5"
TEXT_COLOR = "#1F2933"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Identify bridge users, videos, and entities across detected World Cup communities.")
    parser.add_argument("--edges", default=str(DEFAULT_EDGES_PATH.relative_to(ROOT_DIR)), help="User-user edge CSV path.")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES_PATH.relative_to(ROOT_DIR)), help="User-node CSV path.")
    parser.add_argument("--user-video", default=str(DEFAULT_USER_VIDEO_PATH.relative_to(ROOT_DIR)), help="User-video edge CSV path.")
    parser.add_argument("--comments", default=str(DEFAULT_COMMENTS_PATH.relative_to(ROOT_DIR)), help="Processed comments CSV path.")
    parser.add_argument("--communities", default=str(DEFAULT_COMMUNITIES_PATH.relative_to(ROOT_DIR)), help="User-community CSV path.")
    parser.add_argument(
        "--community-summary",
        default=str(DEFAULT_COMMUNITY_SUMMARY_PATH.relative_to(ROOT_DIR)),
        help="Community summary CSV path.",
    )
    parser.add_argument(
        "--content-summary",
        default=str(DEFAULT_CONTENT_SUMMARY_PATH.relative_to(ROOT_DIR)),
        help="Accepted for CLI compatibility; bridge-entity scoring now uses comment-level dictionary matches.",
    )
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="Minimum edge weight for the bridge graph.")
    parser.add_argument(
        "--min-main-community-users",
        type=int,
        default=DEFAULT_MIN_MAIN_COMMUNITY_USERS,
        help="Minimum users for a community to count as a main community.",
    )
    parser.add_argument("--output-users", default=str(DEFAULT_OUTPUT_USERS.relative_to(ROOT_DIR)), help="Bridge users CSV path.")
    parser.add_argument("--output-videos", default=str(DEFAULT_OUTPUT_VIDEOS.relative_to(ROOT_DIR)), help="Bridge videos CSV path.")
    parser.add_argument("--output-entities", default=str(DEFAULT_OUTPUT_ENTITIES.relative_to(ROOT_DIR)), help="Bridge entities CSV path.")
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_OUTPUT.relative_to(ROOT_DIR)),
        help="Compact bridge-analysis summary CSV path.",
    )
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR.relative_to(ROOT_DIR)), help="Directory for PNG figures.")
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N, help="Top videos to show in the bridge-video figure.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed reserved for reproducible approximations if needed.")
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
    print(f"[analyze_bridges] {message}")


def require_columns(dataframe: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Validate that required columns are present."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_label}: {', '.join(missing)}")


def normalize_label(value: Any, fallback: str = "unknown") -> str:
    """Normalize possibly missing labels into stable strings."""
    if value is None or pd.isna(value):
        return fallback
    text = str(value).strip()
    return text if text else fallback


def round_metric(value: Any, digits: int = 4) -> float:
    """Round numeric metrics for compact outputs."""
    return round(float(value), digits)


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


def apply_matplotlib_style() -> None:
    """Apply a clean matplotlib style for report-ready PNGs."""
    plt.style.use("default")
    plt.rcParams.update(
        {
            "axes.facecolor": "white",
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "figure.facecolor": "white",
            "font.size": 11,
            "savefig.facecolor": "white",
            "xtick.color": "#3D4A57",
            "ytick.color": "#3D4A57",
        }
    )


def load_edges(path: Path, threshold: int) -> pd.DataFrame:
    """Load and filter user-user edges for the bridge graph."""
    edges_df = pd.read_csv(path)
    require_columns(edges_df, EDGE_REQUIRED_COLUMNS, display_path(path))
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)
    edges_df["weight"] = pd.to_numeric(edges_df["weight"], errors="coerce")
    edges_df = edges_df.dropna(subset=["source", "target", "weight"]).copy()
    edges_df["weight"] = edges_df["weight"].astype(int)
    edges_df = edges_df[edges_df["weight"] >= threshold].copy()
    if edges_df.empty:
        raise ValueError(f"No user-user edges remain after filtering to weight >= {threshold}.")
    return edges_df.sort_values(["weight", "source", "target"], ascending=[False, True, True], kind="stable").reset_index(drop=True)


def load_nodes(path: Path) -> pd.DataFrame:
    """Load user-node metadata."""
    nodes_df = pd.read_csv(path)
    require_columns(nodes_df, NODE_REQUIRED_COLUMNS, display_path(path))
    nodes_df["author_id"] = nodes_df["author_id"].astype(str)
    nodes_df["total_comments"] = pd.to_numeric(nodes_df["total_comments"], errors="coerce").fillna(0).astype(int)
    nodes_df["videos_commented"] = pd.to_numeric(nodes_df["videos_commented"], errors="coerce").fillna(0).astype(int)
    nodes_df["dominant_primary_category"] = nodes_df["dominant_primary_category"].apply(normalize_label)
    nodes_df["dominant_video_context_type"] = nodes_df["dominant_video_context_type"].apply(normalize_label)
    return nodes_df[NODE_REQUIRED_COLUMNS].drop_duplicates(subset=["author_id"]).reset_index(drop=True)


def load_user_video_edges(path: Path) -> pd.DataFrame:
    """Load user-video participation edges."""
    user_video_df = pd.read_csv(path)
    require_columns(user_video_df, USER_VIDEO_REQUIRED_COLUMNS, display_path(path))
    user_video_df["author_id"] = user_video_df["author_id"].astype(str)
    user_video_df["video_id"] = user_video_df["video_id"].astype(str)
    user_video_df["comment_count"] = pd.to_numeric(user_video_df["comment_count"], errors="coerce").fillna(0).astype(int)
    for column in ["title", "channel_title", "primary_category", "video_context_type"]:
        user_video_df[column] = user_video_df[column].apply(normalize_label)
    return user_video_df.reset_index(drop=True)


def load_comments(path: Path) -> pd.DataFrame:
    """Load processed comments used for comment-level bridge entities."""
    comments_df = pd.read_csv(path)
    require_columns(comments_df, COMMENT_REQUIRED_COLUMNS, display_path(path))
    comments_df["comment_id"] = comments_df["comment_id"].astype(str)
    comments_df["author_id"] = comments_df["author_id"].astype(str)
    comments_df["clean_text"] = comments_df["clean_text"].fillna("").astype(str)
    return comments_df[COMMENT_REQUIRED_COLUMNS].drop_duplicates(subset=["comment_id"]).reset_index(drop=True)


def load_user_communities(path: Path) -> pd.DataFrame:
    """Load detected user-community assignments."""
    communities_df = pd.read_csv(path)
    require_columns(communities_df, COMMUNITY_REQUIRED_COLUMNS, display_path(path))
    communities_df["author_id"] = communities_df["author_id"].astype(str)
    communities_df["community_id"] = pd.to_numeric(communities_df["community_id"], errors="raise").astype(int)
    communities_df["community_label"] = communities_df["community_label"].astype(str)
    return communities_df.drop_duplicates(subset=["author_id"]).reset_index(drop=True)


def load_community_summary(path: Path) -> pd.DataFrame:
    """Load community summary used to define main communities."""
    community_summary_df = pd.read_csv(path)
    require_columns(community_summary_df, COMMUNITY_SUMMARY_REQUIRED_COLUMNS, display_path(path))
    community_summary_df["community_id"] = pd.to_numeric(community_summary_df["community_id"], errors="raise").astype(int)
    community_summary_df["community_label"] = community_summary_df["community_label"].astype(str)
    community_summary_df["users"] = pd.to_numeric(community_summary_df["users"], errors="coerce").fillna(0).astype(int)
    return community_summary_df.reset_index(drop=True)


def load_entity_type_lookup(path: Path) -> dict[str, str]:
    """Load normalized entity to entity-type mappings."""
    if not path.exists():
        return {}
    dictionary_df = pd.read_csv(path)
    require_columns(dictionary_df, ENTITY_DICTIONARY_REQUIRED_COLUMNS, display_path(path))
    lookup: dict[str, str] = {}
    for row in dictionary_df.itertuples(index=False):
        normalized_entity = normalize_label(row.normalized_entity, fallback="")
        entity_type = normalize_label(row.entity_type, fallback="")
        if normalized_entity and entity_type and normalized_entity not in lookup:
            lookup[normalized_entity] = entity_type
    return lookup


def normalize_entity_text(text: str) -> str:
    """Normalize entity surface text while preserving short acronyms."""
    cleaned = re.sub(r"\s+", " ", str(text).strip(" \t\r\n\"'.,!?;:()[]{}"))
    if not cleaned:
        return ""
    if cleaned.isupper() and len(cleaned) <= 8:
        return cleaned
    parts = []
    for part in cleaned.split(" "):
        parts.append(part if part.isupper() else part.capitalize())
    return " ".join(parts)


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


def load_compiled_entity_dictionary(path: Path) -> dict[str, Any]:
    """Load and compile the dictionary-based normalized entity matcher."""
    dictionary_df = pd.read_csv(path)
    require_columns(dictionary_df, ENTITY_DICTIONARY_REQUIRED_COLUMNS, display_path(path))

    compiled_entries: list[dict[str, Any]] = []
    alias_lookup: dict[str, str] = {}
    for row in dictionary_df.itertuples(index=False):
        pattern = normalize_label(row.pattern, fallback="")
        normalized_entity = normalize_entity_text(normalize_label(row.normalized_entity, fallback=""))
        entity_type = normalize_label(row.entity_type, fallback="unknown")
        if not pattern or not normalized_entity:
            continue
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
        alias_lookup[normalize_pattern_key(pattern)] = entity_type
    if not compiled_entries:
        raise ValueError(f"Entity dictionary is empty: {display_path(path)}")
    compiled_entries = sorted(
        compiled_entries,
        key=lambda entry: (-entry["pattern_length"], -entry["token_length"], entry["pattern"].casefold()),
    )
    return {"entries": compiled_entries, "alias_lookup": alias_lookup}


def main_community_lookup(community_summary_df: pd.DataFrame, min_main_community_users: int) -> tuple[set[int], dict[int, str]]:
    """Return main community IDs and a label lookup."""
    label_lookup = dict(zip(community_summary_df["community_id"], community_summary_df["community_label"]))
    main_ids = set(community_summary_df.loc[community_summary_df["users"] >= min_main_community_users, "community_id"].astype(int))
    return main_ids, label_lookup


def community_sort_key(label: str) -> tuple[int, str]:
    """Sort community labels like C1, C2, ..., then fall back alphabetically."""
    label = str(label)
    if label.startswith("C") and label[1:].isdigit():
        return (0, f"{int(label[1:]):06d}")
    return (1, label.casefold())


def normalized_entropy_from_counts(counts: list[float]) -> float:
    """Compute normalized Shannon entropy from non-negative counts."""
    positive_counts = [float(value) for value in counts if float(value) > 0]
    if len(positive_counts) <= 1:
        return 0.0
    total = sum(positive_counts)
    probabilities = [value / total for value in positive_counts]
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0)
    return entropy / math.log(len(positive_counts))


def participation_coefficient_from_counts(counts: list[float]) -> float:
    """Compute a participation-coefficient-style spread metric from community counts."""
    positive_counts = [float(value) for value in counts if float(value) > 0]
    if not positive_counts:
        return 0.0
    total = sum(positive_counts)
    probabilities = [value / total for value in positive_counts]
    return 1.0 - sum(probability * probability for probability in probabilities)


def build_bridge_graph(edges_df: pd.DataFrame) -> nx.Graph:
    """Build an undirected weighted bridge graph."""
    graph = nx.Graph()
    for row in edges_df.itertuples(index=False):
        graph.add_edge(str(row.source), str(row.target), weight=int(row.weight))
    if graph.number_of_nodes() == 0:
        raise ValueError("Bridge graph is empty after loading the filtered edges.")
    return graph


def compute_bridge_users(graph: nx.Graph, nodes_df: pd.DataFrame, communities_df: pd.DataFrame) -> pd.DataFrame:
    """Compute bridge-user metrics on the thresholded co-commenting graph."""
    betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    degree_lookup = dict(graph.degree())
    weighted_degree_lookup = dict(graph.degree(weight="weight"))

    bridge_users_df = pd.DataFrame({"author_id": sorted(graph.nodes())})
    bridge_users_df["degree"] = bridge_users_df["author_id"].map(degree_lookup).fillna(0).astype(int)
    bridge_users_df["weighted_degree"] = bridge_users_df["author_id"].map(weighted_degree_lookup).fillna(0).astype(int)
    bridge_users_df["betweenness_centrality"] = bridge_users_df["author_id"].map(betweenness).fillna(0.0).map(round_metric)

    bridge_users_df = bridge_users_df.merge(
        communities_df[["author_id", "community_id", "community_label"]],
        on="author_id",
        how="left",
        validate="one_to_one",
    )
    bridge_users_df = bridge_users_df.merge(
        nodes_df,
        on="author_id",
        how="left",
        validate="one_to_one",
    )
    bridge_users_df["community_id"] = pd.to_numeric(bridge_users_df["community_id"], errors="coerce").fillna(-1).astype(int)
    bridge_users_df["community_label"] = bridge_users_df["community_label"].fillna("unassigned")
    bridge_users_df["videos_commented"] = pd.to_numeric(bridge_users_df["videos_commented"], errors="coerce").fillna(0).astype(int)
    bridge_users_df["total_comments"] = pd.to_numeric(bridge_users_df["total_comments"], errors="coerce").fillna(0).astype(int)
    bridge_users_df["dominant_primary_category"] = bridge_users_df["dominant_primary_category"].apply(normalize_label)
    bridge_users_df["dominant_video_context_type"] = bridge_users_df["dominant_video_context_type"].apply(normalize_label)

    bridge_users_df = bridge_users_df.sort_values(
        ["betweenness_centrality", "weighted_degree", "degree", "author_id"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    bridge_users_df["bridge_user_rank"] = range(1, len(bridge_users_df) + 1)
    return bridge_users_df[
        [
            "author_id",
            "community_id",
            "community_label",
            "degree",
            "weighted_degree",
            "betweenness_centrality",
            "videos_commented",
            "total_comments",
            "dominant_primary_category",
            "dominant_video_context_type",
            "bridge_user_rank",
        ]
    ]


def most_common_stable(series: pd.Series) -> str:
    """Return the most common non-empty value in a stable, deterministic way."""
    cleaned_values = [normalize_label(value, fallback="") for value in series]
    cleaned_values = [value for value in cleaned_values if value]
    if not cleaned_values:
        return "unknown"
    counts = Counter(cleaned_values)
    return sorted(counts.items(), key=lambda item: (-item[1], item[0].casefold()))[0][0]


def format_mix(mix_records: list[tuple[str, int]]) -> str:
    """Format a compact community mix string."""
    sorted_records = sorted(mix_records, key=lambda item: (-item[1], community_sort_key(item[0])))
    return "; ".join(f"{label}:{count}" for label, count in sorted_records)


def compute_bridge_videos(
    user_video_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    main_community_ids: set[int],
) -> pd.DataFrame:
    """Compute bridge-video metrics from user participation across communities."""
    participation_df = user_video_df.merge(
        communities_df[["author_id", "community_id", "community_label"]],
        on="author_id",
        how="inner",
        validate="many_to_one",
    )
    if participation_df.empty:
        raise ValueError("No user-video rows matched detected communities for bridge-video analysis.")

    rows: list[dict[str, Any]] = []
    for video_id, video_df in participation_df.groupby("video_id", sort=False):
        user_counts = (
            video_df.groupby(["community_id", "community_label"], sort=False)["author_id"]
            .nunique()
            .reset_index(name="community_users")
        )
        total_users = int(video_df["author_id"].nunique())
        total_comments = int(video_df["comment_count"].sum())
        communities_represented = int(len(user_counts))
        main_communities_represented = int(user_counts["community_id"].isin(main_community_ids).sum())
        dominant_row = user_counts.sort_values(
            ["community_users", "community_label"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        dominant_share = float(dominant_row["community_users"]) / total_users if total_users else 0.0
        normalized_entropy = normalized_entropy_from_counts(user_counts["community_users"].tolist())
        bridge_score = normalized_entropy * math.log1p(total_users)
        mix_records = [(str(row.community_label), int(row.community_users)) for row in user_counts.itertuples(index=False)]

        rows.append(
            {
                "video_id": str(video_id),
                "title": most_common_stable(video_df["title"]),
                "channel_title": most_common_stable(video_df["channel_title"]),
                "primary_category": most_common_stable(video_df["primary_category"]),
                "video_context_type": most_common_stable(video_df["video_context_type"]),
                "total_community_users": total_users,
                "total_community_comments": total_comments,
                "communities_represented": communities_represented,
                "main_communities_represented": main_communities_represented,
                "dominant_community": str(dominant_row["community_label"]),
                "dominant_community_share": round_metric(dominant_share),
                "community_entropy": round_metric(normalized_entropy),
                "bridge_video_score": round_metric(bridge_score),
                "community_mix": format_mix(mix_records),
            }
        )

    bridge_videos_df = pd.DataFrame(rows)
    bridge_videos_df = bridge_videos_df.sort_values(
        ["bridge_video_score", "main_communities_represented", "total_community_users", "title"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    bridge_videos_df["bridge_video_rank"] = range(1, len(bridge_videos_df) + 1)
    return bridge_videos_df[
        [
            "video_id",
            "title",
            "channel_title",
            "primary_category",
            "video_context_type",
            "total_community_users",
            "total_community_comments",
            "communities_represented",
            "main_communities_represented",
            "dominant_community",
            "dominant_community_share",
            "community_entropy",
            "bridge_video_score",
            "community_mix",
            "bridge_video_rank",
        ]
    ]


def prepare_comments_with_communities(comments_df: pd.DataFrame, communities_df: pd.DataFrame) -> pd.DataFrame:
    """Attach detected communities to processed comments."""
    merged_df = comments_df.merge(
        communities_df[["author_id", "community_id", "community_label"]],
        on="author_id",
        how="inner",
        validate="many_to_one",
    )
    if merged_df.empty:
        raise ValueError("No processed comments matched detected communities for bridge-entity analysis.")
    return merged_df.sort_values(["community_id", "comment_id"], kind="stable").reset_index(drop=True)


def extract_dictionary_entities(text: str, compiled_dictionary: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Extract normalized dictionary entities from one comment without double-counting."""
    records: dict[str, dict[str, str]] = {}
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
                    "normalized_entity": entry["normalized_entity"],
                    "entity_type": entry["entity_type"],
                }
    return records


def extract_comment_level_entity_rows(
    comments_with_communities_df: pd.DataFrame,
    compiled_dictionary: dict[str, Any],
) -> pd.DataFrame:
    """Return one row per comment-community-entity match."""
    rows: list[dict[str, Any]] = []
    for row in comments_with_communities_df.itertuples(index=False):
        entity_records = extract_dictionary_entities(str(row.clean_text), compiled_dictionary)
        for record in entity_records.values():
            rows.append(
                {
                    "comment_id": row.comment_id,
                    "community_id": int(row.community_id),
                    "community_label": str(row.community_label),
                    "normalized_entity": record["normalized_entity"],
                    "entity_type": record["entity_type"],
                }
            )
    if not rows:
        raise ValueError("No dictionary-based bridge entities were found in the processed comments.")
    return pd.DataFrame(rows)


def compute_bridge_entities(
    comments_df: pd.DataFrame,
    communities_df: pd.DataFrame,
    main_community_ids: set[int],
    compiled_dictionary: dict[str, Any],
) -> tuple[pd.DataFrame, str]:
    """Compute bridge-entity metrics from comment-level dictionary matches across all communities."""
    comments_with_communities_df = prepare_comments_with_communities(comments_df, communities_df)
    entity_matches_df = extract_comment_level_entity_rows(comments_with_communities_df, compiled_dictionary)
    entity_rows_df = (
        entity_matches_df.groupby(["community_id", "community_label", "normalized_entity", "entity_type"], as_index=False)
        .size()
        .rename(columns={"size": "comments_with_entity"})
        .reset_index(drop=True)
    )

    rows: list[dict[str, Any]] = []
    for entity, entity_df in entity_rows_df.groupby("normalized_entity", sort=False):
        total_comments_with_entity = int(entity_df["comments_with_entity"].sum())
        communities_mentioned_in = int(entity_df["community_id"].nunique())
        main_communities_mentioned_in = int(entity_df.loc[entity_df["community_id"].isin(main_community_ids), "community_id"].nunique())
        dominant_row = entity_df.sort_values(
            ["comments_with_entity", "community_label"],
            ascending=[False, True],
            kind="stable",
        ).iloc[0]
        dominant_share = float(dominant_row["comments_with_entity"]) / total_comments_with_entity if total_comments_with_entity else 0.0
        participation_coefficient = participation_coefficient_from_counts(entity_df["comments_with_entity"].tolist())
        bridge_score = participation_coefficient * math.log1p(total_comments_with_entity)
        mix_records = [(str(row.community_label), int(row.comments_with_entity)) for row in entity_df.itertuples(index=False)]
        entity_type = normalize_label(dominant_row["entity_type"], fallback="unknown")

        rows.append(
            {
                "normalized_entity": entity,
                "entity_type": entity_type,
                "total_comments_with_entity": total_comments_with_entity,
                "communities_mentioned_in": communities_mentioned_in,
                "main_communities_mentioned_in": main_communities_mentioned_in,
                "participation_coefficient": round_metric(participation_coefficient),
                "dominant_community": str(dominant_row["community_label"]),
                "dominant_community_share": round_metric(dominant_share),
                "bridge_entity_score": round_metric(bridge_score),
                "community_mix": format_mix(mix_records),
            }
        )

    bridge_entities_df = pd.DataFrame(rows)
    bridge_entities_df = bridge_entities_df.sort_values(
        ["bridge_entity_score", "total_comments_with_entity", "communities_mentioned_in", "normalized_entity"],
        ascending=[False, False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    bridge_entities_df["bridge_entity_rank"] = range(1, len(bridge_entities_df) + 1)
    return (
        bridge_entities_df[
            [
                "normalized_entity",
                "entity_type",
                "total_comments_with_entity",
                "communities_mentioned_in",
                "main_communities_mentioned_in",
                "participation_coefficient",
                "dominant_community",
                "dominant_community_share",
                "bridge_entity_score",
                "community_mix",
                "bridge_entity_rank",
            ]
        ],
        "participation coefficient across all detected communities + log mention support",
    )


def wrapped_label(primary_text: str, secondary_text: str, width: int) -> str:
    """Build a wrapped label for bar charts."""
    combined = f"{primary_text} ({secondary_text})" if secondary_text and secondary_text != "unknown" else primary_text
    safe_text = combined.encode("ascii", "ignore").decode("ascii")
    return textwrap.fill(safe_text, width=width, break_long_words=False)


def create_bridge_videos_figure(bridge_videos_df: pd.DataFrame, top_n: int) -> plt.Figure:
    """Create a horizontal bar chart of the top bridge videos."""
    figure_df = bridge_videos_df.head(top_n).copy()
    if figure_df.empty:
        raise ValueError("No bridge videos are available for plotting.")
    figure_df = figure_df.iloc[::-1].copy()
    labels = [wrapped_label(title, channel, width=42) for title, channel in zip(figure_df["title"], figure_df["channel_title"])]

    figure, axis = plt.subplots(figsize=(12, max(6, 0.7 * len(figure_df) + 1.5)))
    bars = axis.barh(labels, figure_df["bridge_video_score"], color=BAR_COLOR)
    axis.set_title("Top Bridge Videos")
    axis.set_xlabel("Bridge video score")
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)

    xmax = float(figure_df["bridge_video_score"].max()) if len(figure_df) else 0.0
    for bar, main_count in zip(bars, figure_df["main_communities_represented"]):
        axis.text(
            bar.get_width() + max(xmax * 0.02, 0.03),
            bar.get_y() + bar.get_height() / 2,
            f"{int(main_count)} main",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT_COLOR,
        )

    figure.tight_layout()
    return figure


def create_bridge_entities_figure(bridge_entities_df: pd.DataFrame, top_n: int) -> plt.Figure:
    """Create a horizontal bar chart of the top bridge entities."""
    figure_df = bridge_entities_df.head(top_n).copy()
    if figure_df.empty:
        raise ValueError("No bridge entities are available for plotting.")
    figure_df = figure_df.iloc[::-1].copy()

    figure, axis = plt.subplots(figsize=(11, max(6, 0.55 * len(figure_df) + 1.5)))
    bars = axis.barh(figure_df["normalized_entity"], figure_df["bridge_entity_score"], color=ACCENT_COLOR)
    axis.set_title("Top Bridge Entities")
    axis.set_xlabel("Bridge entity score")
    axis.grid(axis="x", color=GRID_COLOR, linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)

    xmax = float(figure_df["bridge_entity_score"].max()) if len(figure_df) else 0.0
    for bar, community_count, main_count in zip(
        bars,
        figure_df["communities_mentioned_in"],
        figure_df["main_communities_mentioned_in"],
    ):
        axis.text(
            bar.get_width() + max(xmax * 0.02, 0.03),
            bar.get_y() + bar.get_height() / 2,
            f"{int(community_count)} communities / {int(main_count)} main communities",
            va="center",
            ha="left",
            fontsize=9,
            color=TEXT_COLOR,
        )

    figure.tight_layout()
    return figure


def build_summary_table(
    threshold: int,
    graph: nx.Graph,
    bridge_users_df: pd.DataFrame,
    bridge_videos_df: pd.DataFrame,
    bridge_entities_df: pd.DataFrame,
    entity_method_detail: str,
) -> pd.DataFrame:
    """Build a compact key-value bridge-analysis summary."""
    top_user = bridge_users_df.iloc[0]
    top_video = bridge_videos_df.iloc[0]
    top_entity = bridge_entities_df.iloc[0]
    metrics = {
        "edge_threshold_used": threshold,
        "users_in_bridge_graph": graph.number_of_nodes(),
        "edges_in_bridge_graph": graph.number_of_edges(),
        "bridge_users_computed": len(bridge_users_df),
        "bridge_videos_computed": len(bridge_videos_df),
        "bridge_entities_computed": len(bridge_entities_df),
        "top_bridge_user_id": top_user["author_id"],
        "top_bridge_user_community": top_user["community_label"],
        "top_bridge_video_title": top_video["title"],
        "top_bridge_video_score": round_metric(top_video["bridge_video_score"]),
        "top_bridge_entity": top_entity["normalized_entity"],
        "top_bridge_entity_score": round_metric(top_entity["bridge_entity_score"]),
        "bridge_user_method": "unweighted betweenness centrality on the thresholded weight>=2 co-commenting graph",
        "bridge_video_method": "normalized community entropy by user mix multiplied by log1p(total community users on video)",
        "bridge_entity_method": entity_method_detail,
        "bridge_entity_score_formula": "participation_coefficient * log1p(total_comments_with_entity)",
        "caveats": (
            "Bridge users are anonymized and indicate structural position, not influence; "
            "bridge videos indicate cross-community participation, not direct interaction; "
            "bridge entities indicate cross-community narrative anchors, not agreement; "
            "results depend on the sampled YouTube corpus and the chosen weight>=2 threshold."
        ),
    }
    return pd.DataFrame([{"metric": metric, "value": metrics[metric]} for metric in SUMMARY_METRIC_ORDER])


def main() -> None:
    """Run the bridge-analysis CLI."""
    args = parse_args()
    if args.threshold < 1:
        raise ValueError("Threshold must be a positive integer.")
    if args.min_main_community_users < 1:
        raise ValueError("Minimum main-community users must be a positive integer.")
    if args.top_n < 1:
        raise ValueError("Top-n must be a positive integer.")

    apply_matplotlib_style()

    edges_path = resolve_path(args.edges)
    nodes_path = resolve_path(args.nodes)
    user_video_path = resolve_path(args.user_video)
    comments_path = resolve_path(args.comments)
    communities_path = resolve_path(args.communities)
    community_summary_path = resolve_path(args.community_summary)
    output_users_path = resolve_path(args.output_users)
    output_videos_path = resolve_path(args.output_videos)
    output_entities_path = resolve_path(args.output_entities)
    summary_output_path = resolve_path(args.summary_output)
    figures_dir = resolve_path(args.figures_dir)

    required_paths = [
        edges_path,
        nodes_path,
        user_video_path,
        comments_path,
        communities_path,
        community_summary_path,
        DEFAULT_ENTITY_DICTIONARY_PATH,
    ]
    for required_path in required_paths:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required file: {display_path(required_path)}")

    edges_df = load_edges(edges_path, threshold=args.threshold)
    nodes_df = load_nodes(nodes_path)
    user_video_df = load_user_video_edges(user_video_path)
    comments_df = load_comments(comments_path)
    communities_df = load_user_communities(communities_path)
    community_summary_df = load_community_summary(community_summary_path)
    compiled_dictionary = load_compiled_entity_dictionary(DEFAULT_ENTITY_DICTIONARY_PATH)

    main_community_ids, _ = main_community_lookup(community_summary_df, min_main_community_users=args.min_main_community_users)
    graph = build_bridge_graph(edges_df)
    bridge_users_df = compute_bridge_users(graph, nodes_df, communities_df)
    bridge_videos_df = compute_bridge_videos(user_video_df, communities_df, main_community_ids)
    bridge_entities_df, entity_method_detail = compute_bridge_entities(comments_df, communities_df, main_community_ids, compiled_dictionary)
    summary_df = build_summary_table(
        threshold=args.threshold,
        graph=graph,
        bridge_users_df=bridge_users_df,
        bridge_videos_df=bridge_videos_df,
        bridge_entities_df=bridge_entities_df,
        entity_method_detail=entity_method_detail,
    )

    write_csv_atomic(bridge_users_df, output_users_path, overwrite=args.overwrite)
    write_csv_atomic(bridge_videos_df, output_videos_path, overwrite=args.overwrite)
    write_csv_atomic(bridge_entities_df, output_entities_path, overwrite=args.overwrite)
    write_csv_atomic(summary_df, summary_output_path, overwrite=args.overwrite)

    bridge_videos_figure_path = figures_dir / "bridge_videos.png"
    bridge_entities_figure_path = figures_dir / "bridge_entities.png"
    save_figure_atomic(create_bridge_videos_figure(bridge_videos_df, top_n=args.top_n), bridge_videos_figure_path, overwrite=args.overwrite)
    save_figure_atomic(
        create_bridge_entities_figure(bridge_entities_df, top_n=max(15, args.top_n)),
        bridge_entities_figure_path,
        overwrite=args.overwrite,
    )

    print_status(f"Bridge graph users: {graph.number_of_nodes()} | edges: {graph.number_of_edges()}")
    print_status(
        f"Bridge users: {len(bridge_users_df)} | bridge videos: {len(bridge_videos_df)} | bridge entities: {len(bridge_entities_df)}"
    )
    print_status(
        f"Top bridge video: {bridge_videos_df.iloc[0]['title']} | score={round_metric(bridge_videos_df.iloc[0]['bridge_video_score'])}"
    )
    print_status(
        f"Top bridge entity: {bridge_entities_df.iloc[0]['normalized_entity']} | score={round_metric(bridge_entities_df.iloc[0]['bridge_entity_score'])}"
    )
    print_status("Bridge entities use comment-level dictionary matching across all detected communities.")
    print_status(f"Generated figures: {display_path(bridge_videos_figure_path)} | {display_path(bridge_entities_figure_path)}")


if __name__ == "__main__":
    main()