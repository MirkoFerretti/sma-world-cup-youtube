"""Build participation networks and threshold diagnostics from processed comments."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import matplotlib
import networkx as nx
import pandas as pd

from config import FIGURES_DIR, PROCESSED_DATA_DIR, ROOT_DIR, TABLES_DIR


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_COMMENTS_PATH = PROCESSED_DATA_DIR / "comments_processed.csv"
DEFAULT_USER_VIDEO_OUTPUT = PROCESSED_DATA_DIR / "user_video_edges.csv"
DEFAULT_EDGES_OUTPUT = PROCESSED_DATA_DIR / "user_network_edges.csv"
DEFAULT_NODES_OUTPUT = PROCESSED_DATA_DIR / "user_network_nodes.csv"
DEFAULT_SUMMARY_OUTPUT = TABLES_DIR / "network_summary.csv"
DEFAULT_THRESHOLDS = [1, 2, 3]

REQUIRED_COLUMNS = [
    "comment_id",
    "video_id",
    "author_id",
    "published_at",
    "title",
    "channel_title",
    "primary_category",
    "video_context_type",
]

OVERALL_METRIC_ORDER = [
    "total_processed_comments",
    "unique_users",
    "unique_videos",
    "user_video_edges",
    "users_with_1_video",
    "users_with_at_least_2_videos",
    "users_with_at_least_3_videos",
    "max_videos_commented_by_user",
    "mean_videos_commented_by_user",
    "median_videos_commented_by_user",
    "total_user_user_edges_weight_ge_1",
    "max_edge_weight",
    "mean_edge_weight",
    "median_edge_weight",
]

THRESHOLD_METRIC_ORDER = [
    "nodes",
    "edges",
    "density",
    "connected_components",
    "largest_component_nodes",
    "largest_component_share",
    "average_degree",
    "median_degree",
    "max_degree",
    "isolated_nodes_after_threshold",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build participation networks and threshold diagnostics.")
    parser.add_argument(
        "--comments",
        default=str(DEFAULT_COMMENTS_PATH.relative_to(ROOT_DIR)),
        help="Processed comments CSV path.",
    )
    parser.add_argument(
        "--user-video-output",
        default=str(DEFAULT_USER_VIDEO_OUTPUT.relative_to(ROOT_DIR)),
        help="User-video edge CSV path.",
    )
    parser.add_argument(
        "--edges-output",
        default=str(DEFAULT_EDGES_OUTPUT.relative_to(ROOT_DIR)),
        help="User-user edge CSV path.",
    )
    parser.add_argument(
        "--nodes-output",
        default=str(DEFAULT_NODES_OUTPUT.relative_to(ROOT_DIR)),
        help="User-node CSV path.",
    )
    parser.add_argument(
        "--summary-output",
        default=str(DEFAULT_SUMMARY_OUTPUT.relative_to(ROOT_DIR)),
        help="Network summary CSV path.",
    )
    parser.add_argument(
        "--figures-dir",
        default=str(FIGURES_DIR.relative_to(ROOT_DIR)),
        help="Directory for diagnostic figures.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=DEFAULT_THRESHOLDS,
        help="Edge-weight thresholds used only for diagnostics.",
    )
    parser.set_defaults(overwrite=True)
    parser.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        help="Overwrite generated outputs. This is the default behavior.",
    )
    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Stop instead of overwriting existing generated outputs.",
    )
    return parser.parse_args()


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
    """Print a terminal status message."""
    print(f"[build_networks] {message}")


def require_columns(dataframe: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Validate required columns in a dataframe."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_label}: {', '.join(missing)}")


def require_non_missing_keys(dataframe: pd.DataFrame, key_columns: list[str], file_label: str) -> None:
    """Stop if essential key columns contain missing or blank values."""
    missing_counts: dict[str, int] = {}
    for column in key_columns:
        missing_mask = dataframe[column].isna() | dataframe[column].astype(str).str.strip().eq("")
        count = int(missing_mask.sum())
        if count > 0:
            missing_counts[column] = count
    if missing_counts:
        detail = ", ".join(f"{column}={count}" for column, count in missing_counts.items())
        raise ValueError(f"Missing values found in required key columns for {file_label}: {detail}")


def load_comments(path: Path) -> pd.DataFrame:
    """Load processed comments and parse timestamps needed for network construction."""
    comments_df = pd.read_csv(path)
    require_columns(comments_df, REQUIRED_COLUMNS, display_path(path))
    require_non_missing_keys(comments_df, ["comment_id", "video_id", "author_id", "published_at"], display_path(path))

    comments_df["video_id"] = comments_df["video_id"].astype(str)
    comments_df["author_id"] = comments_df["author_id"].astype(str)
    comments_df["published_at_ts"] = pd.to_datetime(comments_df["published_at"], utc=True, errors="coerce")

    invalid_timestamps = int(comments_df["published_at_ts"].isna().sum())
    if invalid_timestamps > 0:
        raise ValueError(
            f"Could not parse {invalid_timestamps} published_at values in {display_path(path)}."
        )

    return comments_df


def normalize_label(value: Any) -> str:
    """Convert missing values to empty strings and strip whitespace."""
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def choose_dominant_label(series: pd.Series) -> str:
    """Choose a deterministic dominant label for a user profile field."""
    labels = [normalize_label(value) for value in series]
    labels = [label for label in labels if label]
    if not labels:
        return "unknown"

    counts = Counter(labels)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ordered[0][0]


def count_distinct_non_empty(series: pd.Series) -> int:
    """Count distinct non-empty labels."""
    labels = {normalize_label(value) for value in series if normalize_label(value)}
    return len(labels)


def format_timestamp_column(series: pd.Series) -> pd.Series:
    """Format UTC timestamps for output CSV files."""
    return series.dt.strftime("%Y-%m-%dT%H:%M:%SZ")


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


def save_figure_atomic(figure: plt.Figure, output_path: Path, overwrite: bool) -> None:
    """Write a matplotlib figure atomically, overwriting by default."""
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

    figure.savefig(temp_path, dpi=200, bbox_inches="tight")
    temp_path.replace(output_path)
    plt.close(figure)


def build_user_video_edges(comments_df: pd.DataFrame) -> pd.DataFrame:
    """Build the direct user-video participation network.

    The user-video network is the raw bipartite participation structure in this
    project. One edge means an anonymized user commented on a video discussion.
    """
    grouped = (
        comments_df.groupby(["author_id", "video_id"], as_index=False)
        .agg(
            comment_count=("comment_id", "size"),
            first_comment_at=("published_at_ts", "min"),
            last_comment_at=("published_at_ts", "max"),
            title=("title", "first"),
            channel_title=("channel_title", "first"),
            primary_category=("primary_category", "first"),
            video_context_type=("video_context_type", "first"),
        )
        .sort_values(["author_id", "video_id"], kind="stable")
        .reset_index(drop=True)
    )

    grouped["first_comment_at"] = format_timestamp_column(grouped["first_comment_at"])
    grouped["last_comment_at"] = format_timestamp_column(grouped["last_comment_at"])
    return grouped


def build_user_nodes(comments_df: pd.DataFrame) -> pd.DataFrame:
    """Build user-level participation profiles from processed comments."""
    user_nodes = (
        comments_df.groupby("author_id", as_index=False)
        .agg(
            total_comments=("comment_id", "size"),
            videos_commented=("video_id", "nunique"),
            first_comment_at=("published_at_ts", "min"),
            last_comment_at=("published_at_ts", "max"),
            primary_category_count=("primary_category", count_distinct_non_empty),
            dominant_primary_category=("primary_category", choose_dominant_label),
            video_context_type_count=("video_context_type", count_distinct_non_empty),
            dominant_video_context_type=("video_context_type", choose_dominant_label),
        )
        .sort_values("author_id", kind="stable")
        .reset_index(drop=True)
    )

    user_nodes["first_comment_at"] = format_timestamp_column(user_nodes["first_comment_at"])
    user_nodes["last_comment_at"] = format_timestamp_column(user_nodes["last_comment_at"])
    return user_nodes


def build_user_user_edges(user_video_edges_df: pd.DataFrame) -> pd.DataFrame:
    """Project shared video participation into a user-user co-commenting network.

    This projection does not claim direct interaction, friendship, or agreement.
    It only captures shared participation in the same video discussion spaces.
    Edge weights count repeated shared participation across multiple videos.
    """
    pair_weights: Counter[tuple[str, str]] = Counter()
    pair_videos: defaultdict[tuple[str, str], list[str]] = defaultdict(list)

    for video_id, group in user_video_edges_df.groupby("video_id"):
        users = sorted(group["author_id"].tolist())
        if len(users) < 2:
            continue
        for source, target in combinations(users, 2):
            pair = (source, target)
            pair_weights[pair] += 1
            pair_videos[pair].append(video_id)

    rows: list[dict[str, Any]] = []
    for (source, target), weight in pair_weights.items():
        rows.append(
            {
                "source": source,
                "target": target,
                "weight": weight,
                "shared_videos": "|".join(pair_videos[(source, target)]),
            }
        )

    edges_df = pd.DataFrame(rows, columns=["source", "target", "weight", "shared_videos"])
    if edges_df.empty:
        return edges_df

    return edges_df.sort_values(["weight", "source", "target"], ascending=[False, True, True], kind="stable").reset_index(drop=True)


def round_metric(value: float) -> float:
    """Round float metrics for compact report-ready tables."""
    return round(float(value), 4)


def compute_threshold_diagnostics(
    edges_df: pd.DataFrame,
    all_user_ids: set[str],
    thresholds: list[int],
) -> pd.DataFrame:
    """Compute graph diagnostics for multiple edge-weight thresholds.

    Thresholds are diagnostic only. The full weighted edge list is always saved,
    then filtered temporarily to compare how inclusive or strict later community
    detection inputs would be.
    """
    diagnostics: list[dict[str, Any]] = []

    for threshold in thresholds:
        filtered_edges = edges_df[edges_df["weight"] >= threshold].copy()
        if filtered_edges.empty:
            diagnostics.append(
                {
                    "threshold": threshold,
                    "nodes": 0,
                    "edges": 0,
                    "density": 0.0,
                    "connected_components": 0,
                    "largest_component_nodes": 0,
                    "largest_component_share": 0.0,
                    "average_degree": 0.0,
                    "median_degree": 0.0,
                    "max_degree": 0,
                    "isolated_nodes_after_threshold": len(all_user_ids),
                }
            )
            continue

        graph = nx.Graph()
        graph.add_weighted_edges_from(filtered_edges[["source", "target", "weight"]].itertuples(index=False, name=None))

        node_count = graph.number_of_nodes()
        edge_count = graph.number_of_edges()
        connected_components = nx.number_connected_components(graph)
        component_sizes = [len(component) for component in nx.connected_components(graph)]
        largest_component_nodes = max(component_sizes) if component_sizes else 0
        degrees = [degree for _, degree in graph.degree()]

        diagnostics.append(
            {
                "threshold": threshold,
                "nodes": node_count,
                "edges": edge_count,
                "density": round_metric(nx.density(graph) if node_count > 1 else 0.0),
                "connected_components": connected_components,
                "largest_component_nodes": largest_component_nodes,
                "largest_component_share": round_metric(
                    largest_component_nodes / node_count if node_count > 0 else 0.0
                ),
                "average_degree": round_metric(sum(degrees) / node_count if node_count > 0 else 0.0),
                "median_degree": round_metric(median(degrees) if degrees else 0.0),
                "max_degree": max(degrees) if degrees else 0,
                "isolated_nodes_after_threshold": len(all_user_ids) - node_count,
            }
        )

    return pd.DataFrame(diagnostics)


def build_summary_table(
    comments_df: pd.DataFrame,
    user_video_edges_df: pd.DataFrame,
    user_nodes_df: pd.DataFrame,
    user_user_edges_df: pd.DataFrame,
    threshold_diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a tidy report-ready summary table."""
    videos_per_user = user_nodes_df["videos_commented"]
    edge_weights = user_user_edges_df["weight"] if not user_user_edges_df.empty else pd.Series(dtype="int64")

    overall_metrics: dict[str, Any] = {
        "total_processed_comments": len(comments_df),
        "unique_users": int(user_nodes_df["author_id"].nunique()),
        "unique_videos": int(comments_df["video_id"].nunique()),
        "user_video_edges": len(user_video_edges_df),
        "users_with_1_video": int((videos_per_user == 1).sum()),
        "users_with_at_least_2_videos": int((videos_per_user >= 2).sum()),
        "users_with_at_least_3_videos": int((videos_per_user >= 3).sum()),
        "max_videos_commented_by_user": int(videos_per_user.max()) if not videos_per_user.empty else 0,
        "mean_videos_commented_by_user": round_metric(videos_per_user.mean()) if not videos_per_user.empty else 0.0,
        "median_videos_commented_by_user": round_metric(videos_per_user.median()) if not videos_per_user.empty else 0.0,
        "total_user_user_edges_weight_ge_1": len(user_user_edges_df),
        "max_edge_weight": int(edge_weights.max()) if not edge_weights.empty else 0,
        "mean_edge_weight": round_metric(edge_weights.mean()) if not edge_weights.empty else 0.0,
        "median_edge_weight": round_metric(edge_weights.median()) if not edge_weights.empty else 0.0,
    }

    rows: list[dict[str, Any]] = []
    for metric in OVERALL_METRIC_ORDER:
        rows.append({"section": "overall", "threshold": "", "metric": metric, "value": overall_metrics[metric]})

    for _, row in threshold_diagnostics_df.iterrows():
        for metric in THRESHOLD_METRIC_ORDER:
            rows.append(
                {
                    "section": "threshold_diagnostic",
                    "threshold": int(row["threshold"]),
                    "metric": metric,
                    "value": row[metric],
                }
            )

    return pd.DataFrame(rows, columns=["section", "threshold", "metric", "value"])


def create_user_participation_figure(user_nodes_df: pd.DataFrame) -> plt.Figure:
    """Create a clean bar chart for repeat user participation across videos."""
    participation_counts = user_nodes_df["videos_commented"] if not user_nodes_df.empty else pd.Series(dtype="int64")
    buckets = {
        "1 video": int((participation_counts == 1).sum()),
        "2 videos": int((participation_counts == 2).sum()),
        "3 videos": int((participation_counts == 3).sum()),
        "4+ videos": int((participation_counts >= 4).sum()),
    }

    colors = ["#496A81", "#5E8C61", "#D08C60", "#B84A62"]
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(list(buckets.keys()), list(buckets.values()), color=colors, width=0.65)
    axis.set_title("User Participation Across Videos", fontsize=14, pad=12)
    axis.set_xlabel("Distinct videos commented on", fontsize=11)
    axis.set_ylabel("Users", fontsize=11)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    axis.set_axisbelow(True)

    for index, value in enumerate(buckets.values()):
        axis.text(index, value, f"{value:,}", ha="center", va="bottom", fontsize=10)

    figure.tight_layout()
    return figure


def create_threshold_diagnostics_figure(threshold_diagnostics_df: pd.DataFrame) -> plt.Figure:
    """Create a clean three-panel diagnostic figure for threshold comparison."""
    thresholds = threshold_diagnostics_df["threshold"].astype(int).tolist()
    x_positions = list(range(len(thresholds)))

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    metrics = [
        ("nodes", "Nodes retained", "#496A81"),
        ("edges", "Edges retained", "#D08C60"),
        ("largest_component_share", "Largest component share", "#5E8C61"),
    ]

    for axis, (metric, title, color) in zip(axes, metrics):
        values = threshold_diagnostics_df[metric].tolist()
        axis.bar(x_positions, values, color=color, width=0.6)
        axis.set_title(title, fontsize=12, pad=10)
        axis.set_xticks(x_positions, [f">={threshold}" for threshold in thresholds])
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
        axis.set_axisbelow(True)

        if metric == "largest_component_share":
            axis.set_ylim(0, 1.05)
            for index, value in enumerate(values):
                axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
        else:
            for index, value in enumerate(values):
                axis.text(index, value, f"{int(value):,}", ha="center", va="bottom", fontsize=10)

    figure.suptitle("Network Threshold Diagnostics", fontsize=14, y=1.03)
    figure.tight_layout()
    return figure


def validate_thresholds(thresholds: list[int]) -> list[int]:
    """Ensure threshold values are unique positive integers."""
    clean_thresholds = sorted(set(thresholds))
    if not clean_thresholds:
        raise ValueError("At least one threshold is required.")
    invalid = [threshold for threshold in clean_thresholds if threshold < 1]
    if invalid:
        raise ValueError(f"Thresholds must be positive integers: {invalid}")
    return clean_thresholds


def main() -> None:
    """Run the network construction CLI."""
    args = parse_args()
    thresholds = validate_thresholds(args.thresholds)

    comments_path = resolve_path(args.comments)
    user_video_output = resolve_path(args.user_video_output)
    edges_output = resolve_path(args.edges_output)
    nodes_output = resolve_path(args.nodes_output)
    summary_output = resolve_path(args.summary_output)
    figures_dir = resolve_path(args.figures_dir)

    if not comments_path.exists():
        raise FileNotFoundError(f"Missing required file: {display_path(comments_path)}")

    comments_df = load_comments(comments_path)
    print_status(
        f"Loaded {len(comments_df)} processed comments from {comments_df['author_id'].nunique()} users across {comments_df['video_id'].nunique()} videos."
    )

    user_video_edges_df = build_user_video_edges(comments_df)
    user_nodes_df = build_user_nodes(comments_df)
    user_user_edges_df = build_user_user_edges(user_video_edges_df)
    threshold_diagnostics_df = compute_threshold_diagnostics(
        edges_df=user_user_edges_df,
        all_user_ids=set(user_nodes_df["author_id"]),
        thresholds=thresholds,
    )
    summary_df = build_summary_table(
        comments_df=comments_df,
        user_video_edges_df=user_video_edges_df,
        user_nodes_df=user_nodes_df,
        user_user_edges_df=user_user_edges_df,
        threshold_diagnostics_df=threshold_diagnostics_df,
    )

    user_participation_figure = create_user_participation_figure(user_nodes_df)
    threshold_diagnostics_figure = create_threshold_diagnostics_figure(threshold_diagnostics_df)

    write_csv_atomic(user_video_edges_df, user_video_output, overwrite=args.overwrite)
    write_csv_atomic(user_user_edges_df, edges_output, overwrite=args.overwrite)
    write_csv_atomic(user_nodes_df, nodes_output, overwrite=args.overwrite)
    write_csv_atomic(summary_df, summary_output, overwrite=args.overwrite)

    save_figure_atomic(
        user_participation_figure,
        figures_dir / "user_participation_distribution.png",
        overwrite=args.overwrite,
    )
    save_figure_atomic(
        threshold_diagnostics_figure,
        figures_dir / "network_threshold_diagnostics.png",
        overwrite=args.overwrite,
    )

    print_status(
        f"Built {len(user_video_edges_df)} user-video edges, {len(user_user_edges_df)} user-user edges, and {len(user_nodes_df)} user nodes."
    )
    for _, row in threshold_diagnostics_df.iterrows():
        print_status(
            "Threshold >= "
            f"{int(row['threshold'])}: "
            f"nodes={int(row['nodes'])}, "
            f"edges={int(row['edges'])}, "
            f"largest_component_share={float(row['largest_component_share']):.4f}"
        )


if __name__ == "__main__":
    main()