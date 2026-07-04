"""Detect participation communities and create simple report-ready figures."""

from __future__ import annotations

import argparse
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import pandas as pd

from config import FIGURES_DIR, PROCESSED_DATA_DIR, ROOT_DIR, TABLES_DIR

try:
    from networkx.algorithms.community import louvain_communities

    HAS_NETWORKX_LOUVAIN = True
except ImportError:  # pragma: no cover - fallback for older NetworkX versions
    HAS_NETWORKX_LOUVAIN = False
    louvain_communities = None

try:  # pragma: no cover - fallback path depends on environment
    import community as community_louvain

    HAS_PYTHON_LOUVAIN = True
except ImportError:  # pragma: no cover - fallback path depends on environment
    HAS_PYTHON_LOUVAIN = False
    community_louvain = None


DEFAULT_EDGES_PATH = PROCESSED_DATA_DIR / "user_network_edges.csv"
DEFAULT_NODES_PATH = PROCESSED_DATA_DIR / "user_network_nodes.csv"
DEFAULT_USER_VIDEO_PATH = PROCESSED_DATA_DIR / "user_video_edges.csv"
DEFAULT_COMMENTS_PATH = PROCESSED_DATA_DIR / "comments_processed.csv"
DEFAULT_USERS_OUTPUT = PROCESSED_DATA_DIR / "user_communities.csv"
DEFAULT_SUMMARY_OUTPUT = PROCESSED_DATA_DIR / "community_summary.csv"
DEFAULT_VIDEO_PROFILES_OUTPUT = PROCESSED_DATA_DIR / "community_video_profiles.csv"
DEFAULT_SUMMARY_TABLE_OUTPUT = TABLES_DIR / "community_detection_summary.csv"
DEFAULT_THRESHOLD = 2
DEFAULT_MIN_COMMUNITY_SIZE = 20
DEFAULT_TOP_HIGHLIGHT_COMMUNITIES = 5
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

OVERALL_METRIC_ORDER = [
    "edge_threshold_used",
    "users_in_main_graph",
    "edges_in_main_graph",
    "communities_detected",
    "modularity",
    "largest_community_users",
    "median_community_users",
    "small_communities_under_10_users",
    "communities_with_at_least_20_users",
    "share_users_in_communities_at_least_20",
    "notes_about_threshold_choice",
]

COLOR_SEQUENCE = [
    "#496A81",
    "#5E8C61",
    "#D08C60",
    "#B84A62",
    "#6D597A",
    "#3E7C73",
    "#D9A441",
    "#8C5E58",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Detect weighted co-commenting communities and create simple PNG figures.")
    parser.add_argument("--edges", default=str(DEFAULT_EDGES_PATH.relative_to(ROOT_DIR)), help="User-user edge CSV path.")
    parser.add_argument("--nodes", default=str(DEFAULT_NODES_PATH.relative_to(ROOT_DIR)), help="User-node CSV path.")
    parser.add_argument("--user-video", default=str(DEFAULT_USER_VIDEO_PATH.relative_to(ROOT_DIR)), help="User-video edge CSV path.")
    parser.add_argument(
        "--comments",
        default=str(DEFAULT_COMMENTS_PATH.relative_to(ROOT_DIR)),
        help="Processed comments CSV path. Accepted for CLI consistency; current profiling uses user-video and user-node tables.",
    )
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD, help="Minimum edge weight for the main graph.")
    parser.add_argument(
        "--min-community-size",
        type=int,
        default=DEFAULT_MIN_COMMUNITY_SIZE,
        help="Minimum community size shown in profile figures.",
    )
    parser.add_argument(
        "--top-highlight-communities",
        type=int,
        default=DEFAULT_TOP_HIGHLIGHT_COMMUNITIES,
        help="Number of largest communities to export as separate highlight network figures.",
    )
    parser.add_argument("--output-users", default=str(DEFAULT_USERS_OUTPUT.relative_to(ROOT_DIR)), help="User-community CSV path.")
    parser.add_argument("--output-summary", default=str(DEFAULT_SUMMARY_OUTPUT.relative_to(ROOT_DIR)), help="Community summary CSV path.")
    parser.add_argument(
        "--output-video-profiles",
        default=str(DEFAULT_VIDEO_PROFILES_OUTPUT.relative_to(ROOT_DIR)),
        help="Community-video profile CSV path.",
    )
    parser.add_argument(
        "--summary-table",
        default=str(DEFAULT_SUMMARY_TABLE_OUTPUT.relative_to(ROOT_DIR)),
        help="Compact community detection summary CSV path.",
    )
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR.relative_to(ROOT_DIR)), help="Directory for PNG figures.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for Louvain and spring layouts.")
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
    """Render a path relative to the repository root when possible."""
    if path.is_relative_to(ROOT_DIR):
        return str(path.relative_to(ROOT_DIR))
    return str(path)


def print_status(message: str) -> None:
    """Print a terminal status message."""
    print(f"[detect_communities] {message}")


def require_columns(dataframe: pd.DataFrame, required_columns: list[str], file_label: str) -> None:
    """Validate that required columns are present."""
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Missing required columns in {file_label}: {', '.join(missing)}")


def round_metric(value: float) -> float:
    """Round float metrics for compact report-ready tables."""
    return round(float(value), 4)


def normalize_label(value: Any) -> str:
    """Convert missing values into a stable label."""
    if value is None or pd.isna(value):
        return "unknown"
    label = str(value).strip()
    return label if label else "unknown"


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
    """Write a PNG figure atomically, overwriting by default."""
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


def cleanup_stale_visualization_outputs(figures_dir: Path) -> None:
    """Remove stale interactive or superseded community visualization files."""
    stale_names = [
        "community_size_distribution.html",
        "community_context_profile.html",
        "community_category_profile.html",
        "community_backbone_network.html",
        "community_backbone_network.png",
        "user_network_interactive.html",
        "test_plotly_browser.html",
        "test_plotly_browser.png",
        "test_plotly_browser2.png",
    ]
    for name in stale_names:
        path = figures_dir / name
        if path.exists():
            path.unlink()
            print_status(f"Removed stale file: {display_path(path)}")


def load_edges(path: Path, threshold: int) -> pd.DataFrame:
    """Load and filter user-user edges for the main community graph."""
    edges_df = pd.read_csv(path)
    require_columns(edges_df, EDGE_REQUIRED_COLUMNS, display_path(path))
    edges_df["weight"] = pd.to_numeric(edges_df["weight"], errors="coerce")
    edges_df = edges_df.dropna(subset=["source", "target", "weight"]).copy()
    edges_df["source"] = edges_df["source"].astype(str)
    edges_df["target"] = edges_df["target"].astype(str)
    edges_df["weight"] = edges_df["weight"].astype(int)
    edges_df = edges_df[edges_df["weight"] >= threshold].copy()
    if edges_df.empty:
        raise ValueError(f"No user-user edges remain after filtering to weight >= {threshold}.")
    return edges_df.sort_values(["weight", "source", "target"], ascending=[False, True, True], kind="stable").reset_index(drop=True)


def load_nodes(path: Path) -> pd.DataFrame:
    """Load user-level node metadata used for profiling communities."""
    nodes_df = pd.read_csv(path)
    require_columns(nodes_df, NODE_REQUIRED_COLUMNS, display_path(path))
    nodes_df["author_id"] = nodes_df["author_id"].astype(str)
    nodes_df["dominant_primary_category"] = nodes_df["dominant_primary_category"].apply(normalize_label)
    nodes_df["dominant_video_context_type"] = nodes_df["dominant_video_context_type"].apply(normalize_label)
    return nodes_df


def load_user_video(path: Path) -> pd.DataFrame:
    """Load user-video participation edges used to profile communities."""
    user_video_df = pd.read_csv(path)
    require_columns(user_video_df, USER_VIDEO_REQUIRED_COLUMNS, display_path(path))
    user_video_df["author_id"] = user_video_df["author_id"].astype(str)
    user_video_df["video_id"] = user_video_df["video_id"].astype(str)
    user_video_df["comment_count"] = pd.to_numeric(user_video_df["comment_count"], errors="coerce").fillna(0).astype(int)
    user_video_df["title"] = user_video_df["title"].apply(normalize_label)
    user_video_df["channel_title"] = user_video_df["channel_title"].apply(normalize_label)
    user_video_df["primary_category"] = user_video_df["primary_category"].apply(normalize_label)
    user_video_df["video_context_type"] = user_video_df["video_context_type"].apply(normalize_label)
    return user_video_df


def build_main_graph(edges_df: pd.DataFrame) -> nx.Graph:
    """Build the weighted user-user co-commenting graph.

    The main graph uses edge weight >= 2 because threshold >= 1 was dominated
    by one-off overlaps, while threshold >= 3 retained too few users. A
    threshold of >= 2 keeps repeated co-participation while preserving enough
    network structure for later profiling.
    """
    graph = nx.Graph()
    graph.add_weighted_edges_from(edges_df[["source", "target", "weight"]].itertuples(index=False, name=None))
    return graph


def detect_louvain_partitions(graph: nx.Graph, seed: int) -> list[set[str]]:
    """Detect weighted participation communities with Louvain.

    A detected community is a group of anonymized users who repeatedly
    participate in similar World Cup video discussion spaces. It does not mean
    friendship, direct replies, agreement, or support for the same team.
    """
    if HAS_NETWORKX_LOUVAIN:
        communities = louvain_communities(graph, weight="weight", seed=seed)
        return [set(community) for community in communities]

    if HAS_PYTHON_LOUVAIN:  # pragma: no cover - fallback path depends on environment
        partition = community_louvain.best_partition(graph, weight="weight", random_state=seed)
        grouped: defaultdict[int, set[str]] = defaultdict(set)
        for author_id, community_id in partition.items():
            grouped[int(community_id)].add(author_id)
        return list(grouped.values())

    raise RuntimeError("No Louvain implementation is available. Install a newer NetworkX or python-louvain.")


def order_communities(communities: list[set[str]]) -> list[set[str]]:
    """Order communities by size so display labels stay intuitive."""
    return sorted(communities, key=lambda community: (-len(community), sorted(community)[0]))


def community_label(community_id: int) -> str:
    """Render a stable community display label."""
    return f"C{community_id}"


def build_user_communities(
    graph: nx.Graph,
    ordered_communities: list[set[str]],
    nodes_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[int, set[str]]]:
    """Assign users to Louvain communities and join user metadata."""
    community_to_authors: dict[int, set[str]] = {}
    author_to_community: dict[str, tuple[int, str]] = {}

    for index, authors in enumerate(ordered_communities, start=1):
        label = community_label(index)
        for author_id in authors:
            author_to_community[author_id] = (index, label)
        community_to_authors[index] = set(authors)

    user_rows = pd.DataFrame({"author_id": sorted(graph.nodes())})
    user_rows["community_id"] = user_rows["author_id"].map(lambda author_id: author_to_community[author_id][0])
    user_rows["community_label"] = user_rows["author_id"].map(lambda author_id: author_to_community[author_id][1])
    user_rows["degree"] = user_rows["author_id"].map(dict(graph.degree()))
    user_rows["weighted_degree"] = user_rows["author_id"].map(dict(graph.degree(weight="weight")))

    user_communities_df = user_rows.merge(nodes_df, on="author_id", how="left", validate="one_to_one")
    ordered_columns = [
        "author_id",
        "community_id",
        "community_label",
        "total_comments",
        "videos_commented",
        "degree",
        "weighted_degree",
        "dominant_primary_category",
        "dominant_video_context_type",
    ]
    user_communities_df = user_communities_df[ordered_columns].sort_values(
        ["community_id", "author_id"], kind="stable"
    ).reset_index(drop=True)
    return user_communities_df, community_to_authors


def build_community_video_profiles(
    user_communities_df: pd.DataFrame,
    user_video_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate community participation by video for later inspection."""
    merged = user_communities_df[["author_id", "community_id", "community_label"]].merge(
        user_video_df,
        on="author_id",
        how="inner",
        validate="one_to_many",
    )

    profiles_df = (
        merged.groupby(["community_id", "community_label", "video_id"], as_index=False)
        .agg(
            community_users_on_video=("author_id", "nunique"),
            community_comments_on_video=("comment_count", "sum"),
            title=("title", "first"),
            channel_title=("channel_title", "first"),
            primary_category=("primary_category", "first"),
            video_context_type=("video_context_type", "first"),
        )
        .sort_values(
            ["community_id", "community_users_on_video", "community_comments_on_video", "video_id"],
            ascending=[True, False, False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return profiles_df, merged


def select_dominant_share(series: pd.Series, weights: pd.Series) -> tuple[str, float]:
    """Select the dominant label and its weighted share."""
    counts: defaultdict[str, int] = defaultdict(int)
    total_weight = int(weights.sum())
    for label, weight in zip(series, weights):
        counts[normalize_label(label)] += int(weight)

    if total_weight == 0 or not counts:
        return "unknown", 0.0

    label, label_weight = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    return label, round_metric(label_weight / total_weight)


def join_top_values(values: list[str], limit: int = 5) -> str:
    """Join top values into a compact delimiter-separated string."""
    clean_values = [normalize_label(value) for value in values if normalize_label(value) != "unknown"]
    return " | ".join(clean_values[:limit])


def build_community_summary(
    graph: nx.Graph,
    user_communities_df: pd.DataFrame,
    community_user_video_df: pd.DataFrame,
    community_video_profiles_df: pd.DataFrame,
    community_to_authors: dict[int, set[str]],
) -> pd.DataFrame:
    """Build one summary row per detected community."""
    members_df = user_communities_df.groupby(["community_id", "community_label"], as_index=False).agg(
        users=("author_id", "size"),
        total_comments=("total_comments", "sum"),
        avg_degree=("degree", "mean"),
        avg_weighted_degree=("weighted_degree", "mean"),
    )
    users_in_main_graph = len(user_communities_df)

    internal_edge_lookup: dict[int, tuple[int, int]] = {}
    for community_id, authors in community_to_authors.items():
        subgraph = graph.subgraph(authors)
        internal_edge_lookup[community_id] = (
            subgraph.number_of_edges(),
            int(subgraph.size(weight="weight")),
        )

    rows: list[dict[str, Any]] = []
    for _, community_row in members_df.iterrows():
        community_id = int(community_row["community_id"])
        label = str(community_row["community_label"])
        profile_df = community_video_profiles_df[community_video_profiles_df["community_id"] == community_id].copy()
        user_video_rows_df = community_user_video_df[community_user_video_df["community_id"] == community_id].copy()
        internal_edges, internal_edge_weight = internal_edge_lookup[community_id]

        dominant_category, dominant_category_share = select_dominant_share(
            profile_df["primary_category"],
            profile_df["community_comments_on_video"],
        )
        dominant_context, dominant_context_share = select_dominant_share(
            profile_df["video_context_type"],
            profile_df["community_comments_on_video"],
        )

        top_videos_df = profile_df.sort_values(
            ["community_users_on_video", "community_comments_on_video", "video_id"],
            ascending=[False, False, True],
            kind="stable",
        ).head(5)
        top_channels_df = (
            profile_df.groupby("channel_title", as_index=False)
            .agg(channel_users=("community_users_on_video", "sum"), channel_comments=("community_comments_on_video", "sum"))
            .sort_values(["channel_users", "channel_comments", "channel_title"], ascending=[False, False, True], kind="stable")
        )

        rows.append(
            {
                "community_id": community_id,
                "community_label": label,
                "users": int(community_row["users"]),
                "share_of_network_users": round_metric(int(community_row["users"]) / users_in_main_graph if users_in_main_graph else 0.0),
                "total_comments": int(community_row["total_comments"]),
                "videos_touched": int(profile_df["video_id"].nunique()),
                "user_video_edges": int(user_video_rows_df.shape[0]),
                "internal_edges": internal_edges,
                "internal_edge_weight": internal_edge_weight,
                "avg_degree": round_metric(community_row["avg_degree"]),
                "avg_weighted_degree": round_metric(community_row["avg_weighted_degree"]),
                "dominant_primary_category": dominant_category,
                "dominant_primary_category_share": dominant_category_share,
                "dominant_video_context_type": dominant_context,
                "dominant_video_context_type_share": dominant_context_share,
                "top_video_ids": join_top_values(top_videos_df["video_id"].tolist()),
                "top_video_titles": join_top_values(top_videos_df["title"].tolist()),
                "top_channels": join_top_values(top_channels_df["channel_title"].tolist()),
            }
        )

    return pd.DataFrame(rows).sort_values(["users", "community_id"], ascending=[False, True], kind="stable").reset_index(drop=True)


def build_summary_table(
    threshold: int,
    graph: nx.Graph,
    community_summary_df: pd.DataFrame,
    modularity_score: float,
) -> pd.DataFrame:
    """Build a compact report-ready summary table."""
    community_sizes = community_summary_df["users"] if not community_summary_df.empty else pd.Series(dtype="int64")
    communities_at_least_20 = community_summary_df[community_summary_df["users"] >= 20]

    metrics = {
        "edge_threshold_used": threshold,
        "users_in_main_graph": graph.number_of_nodes(),
        "edges_in_main_graph": graph.number_of_edges(),
        "communities_detected": len(community_summary_df),
        "modularity": round_metric(modularity_score),
        "largest_community_users": int(community_sizes.max()) if not community_sizes.empty else 0,
        "median_community_users": round_metric(community_sizes.median()) if not community_sizes.empty else 0.0,
        "small_communities_under_10_users": int((community_sizes < 10).sum()) if not community_sizes.empty else 0,
        "communities_with_at_least_20_users": int((community_sizes >= 20).sum()) if not community_sizes.empty else 0,
        "share_users_in_communities_at_least_20": round_metric(
            communities_at_least_20["users"].sum() / graph.number_of_nodes() if graph.number_of_nodes() else 0.0
        ),
        "notes_about_threshold_choice": "Main graph uses weight >= 2 because threshold 1 was dominated by one-off overlaps, while threshold 3 retained too few users.",
    }

    rows = [{"section": "overall", "metric": metric, "value": metrics[metric]} for metric in OVERALL_METRIC_ORDER]
    for _, row in community_summary_df.head(10).iterrows():
        rows.append({"section": "community_size", "metric": f"{row['community_label']}_users", "value": int(row["users"])})
    return pd.DataFrame(rows, columns=["section", "metric", "value"])


def apply_matplotlib_style() -> None:
    """Apply a simple reusable style for report-ready figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


def add_title_and_subtitle(axis: plt.Axes, title: str, subtitle: str) -> None:
    """Add a clear title and subtitle to a figure axis."""
    axis.set_title(title, loc="left", pad=30)
    axis.text(0.0, 1.04, subtitle, transform=axis.transAxes, ha="left", va="bottom", fontsize=10, color="#4B5563")


def clean_axis(axis: plt.Axes, show_x_grid: bool = False, show_y_grid: bool = True) -> None:
    """Remove chart clutter while keeping guides readable."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#E5E7EB", linewidth=0.8, alpha=0.6) if show_x_grid else axis.grid(False, axis="x")
    axis.grid(axis="y", color="#E5E7EB", linewidth=0.8, alpha=0.8) if show_y_grid else axis.grid(False, axis="y")


def pretty_label(label: str) -> str:
    """Convert underscore labels into readable title-style text."""
    return label.replace("_", " ").title()


def wrap_label(label: str, width: int = 16) -> str:
    """Wrap long legend labels for readability."""
    return "\n".join(textwrap.wrap(pretty_label(label), width=width)) or pretty_label(label)


def build_community_palette(community_labels: list[str]) -> dict[str, str]:
    """Assign consistent colors to the highlighted communities."""
    return {label: COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)] for index, label in enumerate(community_labels)}


def create_community_size_figure(community_summary_df: pd.DataFrame) -> plt.Figure:
    """Create a horizontal bar chart for the top 10 communities plus the long tail."""
    top_df = community_summary_df.head(10).copy()
    if len(community_summary_df) > 10:
        other_users = int(community_summary_df.iloc[10:]["users"].sum())
        top_df = pd.concat(
            [
                top_df[["community_label", "users"]],
                pd.DataFrame([{"community_label": "Other small communities", "users": other_users}]),
            ],
            ignore_index=True,
        )
    else:
        top_df = top_df[["community_label", "users"]].copy()

    top_df = top_df.iloc[::-1].reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(12, 7))
    axis.barh(top_df["community_label"], top_df["users"], color="#496A81")
    add_title_and_subtitle(
        axis,
        title="Community Size Distribution",
        subtitle="Top 10 Louvain communities by size, with the remaining long tail grouped together",
    )
    axis.set_xlabel("Users")
    axis.set_ylabel("Community")
    clean_axis(axis, show_x_grid=True, show_y_grid=False)

    max_users = int(top_df["users"].max()) if not top_df.empty else 0
    for index, value in enumerate(top_df["users"]):
        axis.text(value + max(1, max_users * 0.01), index, f"{int(value):,}", va="center", ha="left", fontsize=10)

    axis.set_xlim(0, max_users * 1.12 if max_users > 0 else 1)
    figure.tight_layout()
    return figure


def create_community_size_long_tail_figure(community_summary_df: pd.DataFrame) -> plt.Figure:
    """Create a full long-tail community size diagnostic across all communities."""
    chart_df = community_summary_df[["community_label", "users"]].copy()
    x_positions = np.arange(len(chart_df))

    figure, axis = plt.subplots(figsize=(12, 7))
    axis.bar(x_positions, chart_df["users"], color="#496A81", width=0.75)
    add_title_and_subtitle(
        axis,
        title="Long-tail distribution of detected communities",
        subtitle="A few large communities contain most users, while many communities are small peripheral groups",
    )
    axis.set_xlabel("Communities sorted by size")
    axis.set_ylabel("Users")
    clean_axis(axis, show_x_grid=False, show_y_grid=True)

    tick_labels_to_show = {"C1", "C2", "C3", "C5", "C10", "C20", "C40", "C60", "C71"}
    tick_positions = [index for index, label in enumerate(chart_df["community_label"]) if label in tick_labels_to_show]
    tick_labels = [chart_df.iloc[index]["community_label"] for index in tick_positions]
    axis.set_xticks(tick_positions, tick_labels)
    axis.set_xlim(-0.75, len(chart_df) - 0.25)

    figure.tight_layout()
    return figure


def build_percentage_profile(
    community_video_profiles_df: pd.DataFrame,
    community_summary_df: pd.DataFrame,
    value_column: str,
    label_column: str,
    min_community_size: int,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a tidy percentage table for the main community profile figures."""
    main_communities = community_summary_df[community_summary_df["users"] >= min_community_size][["community_id", "community_label"]].copy()
    if main_communities.empty:
        main_communities = community_summary_df.head(min(6, len(community_summary_df)))[["community_id", "community_label"]].copy()

    chart_df = community_video_profiles_df.merge(main_communities, on=["community_id", "community_label"], how="inner", validate="many_to_one")
    grouped = (
        chart_df.groupby(["community_label", label_column], as_index=False)[value_column]
        .sum()
        .rename(columns={value_column: "value"})
    )
    totals = grouped.groupby("community_label", as_index=False)["value"].sum().rename(columns={"value": "group_total"})
    grouped = grouped.merge(totals, on="community_label", how="left", validate="many_to_one")
    grouped["percentage"] = grouped["value"] / grouped["group_total"]

    community_order = main_communities["community_label"].tolist()
    label_order = (
        grouped.groupby(label_column, as_index=False)["value"]
        .sum()
        .sort_values(["value", label_column], ascending=[False, True], kind="stable")[label_column]
        .tolist()
    )
    return grouped, community_order, label_order


def create_stacked_percentage_figure(
    percentage_df: pd.DataFrame,
    community_order: list[str],
    label_order: list[str],
    label_column: str,
    title: str,
    subtitle: str,
) -> plt.Figure:
    """Create a 100% stacked bar chart for the main communities."""
    figure, axis = plt.subplots(figsize=(12, 7))
    palette = build_community_palette(label_order)
    bottom = pd.Series(0.0, index=community_order, dtype="float64")

    for label in label_order:
        subset = percentage_df[percentage_df[label_column] == label].set_index("community_label")["percentage"]
        values = [float(subset.get(community_label_text, 0.0)) for community_label_text in community_order]
        axis.bar(
            community_order,
            values,
            bottom=bottom[community_order].tolist(),
            color=palette[label],
            width=0.7,
            label=wrap_label(label),
            edgecolor="white",
            linewidth=0.5,
        )
        bottom = bottom.add(pd.Series(values, index=community_order), fill_value=0.0)

    add_title_and_subtitle(axis, title=title, subtitle=subtitle)
    axis.set_xlabel("Community")
    axis.set_ylabel("Share of community participation")
    axis.set_ylim(0, 1.0)
    axis.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0], labels=["0%", "25%", "50%", "75%", "100%"])
    clean_axis(axis, show_x_grid=False, show_y_grid=True)
    axis.legend(title=pretty_label(label_column), bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    figure.tight_layout()
    return figure


def largest_component_subgraph(graph: nx.Graph) -> nx.Graph:
    """Return the largest connected component for readable network figures."""
    largest_nodes = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_nodes).copy()


def compute_layout(graph: nx.Graph, seed: int) -> dict[str, Any]:
    """Compute a reusable spring layout for the largest connected component.

    The resulting positions are visual aids only. They communicate structure,
    not exact metric distance, and should not be overinterpreted.
    """
    return nx.spring_layout(graph, seed=seed, weight="weight", k=0.28, iterations=150)


def create_network_overview_figure(
    graph: nx.Graph,
    layout: dict[str, Any],
    user_communities_df: pd.DataFrame,
    community_summary_df: pd.DataFrame,
    min_community_size: int,
) -> plt.Figure:
    """Create a simple static overview of the largest connected component."""
    figure, axis = plt.subplots(figsize=(13, 7))
    main_communities = community_summary_df[community_summary_df["users"] >= min_community_size]["community_label"].tolist()
    if not main_communities:
        main_communities = community_summary_df.head(min(5, len(community_summary_df)))["community_label"].tolist()
    palette = build_community_palette(main_communities)

    node_df = user_communities_df[user_communities_df["author_id"].isin(graph.nodes())][["author_id", "community_label"]].copy()
    node_to_label = dict(node_df.itertuples(index=False, name=None))

    nx.draw_networkx_edges(graph, pos=layout, ax=axis, edge_color="#CFCFCF", width=0.35, alpha=0.18)

    small_nodes = [node for node in graph.nodes() if node_to_label.get(node) not in main_communities]
    if small_nodes:
        nx.draw_networkx_nodes(
            graph,
            pos=layout,
            nodelist=small_nodes,
            node_color="#D9D9D9",
            node_size=10,
            linewidths=0,
            alpha=0.85,
            ax=axis,
        )

    for label in main_communities:
        community_nodes = [node for node in graph.nodes() if node_to_label.get(node) == label]
        nx.draw_networkx_nodes(
            graph,
            pos=layout,
            nodelist=community_nodes,
            node_color=palette[label],
            node_size=16,
            linewidths=0,
            alpha=0.95,
            ax=axis,
        )

    add_title_and_subtitle(
        axis,
        title="Community Network Overview",
        subtitle="Largest connected component of the weight >= 2 co-commenting graph; edges indicate repeated co-commenting in at least two videos, not direct interaction",
    )
    axis.set_axis_off()
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", label=label, markerfacecolor=palette[label], markersize=8)
        for label in main_communities
    ]
    legend_handles.append(Line2D([0], [0], marker="o", color="w", label="Other small communities", markerfacecolor="#D9D9D9", markersize=8))
    axis.legend(handles=legend_handles, loc="upper right", frameon=False, title="Community")
    figure.tight_layout()
    return figure


def create_community_highlight_figure(
    graph: nx.Graph,
    layout: dict[str, Any],
    user_communities_df: pd.DataFrame,
    community_summary_row: pd.Series,
    color: str,
) -> plt.Figure:
    """Create a highlight figure for one large community."""
    label = str(community_summary_row["community_label"])
    users = int(community_summary_row["users"])
    dominant_context = pretty_label(str(community_summary_row["dominant_video_context_type"]))
    dominant_category = pretty_label(str(community_summary_row["dominant_primary_category"]))

    node_df = user_communities_df[user_communities_df["author_id"].isin(graph.nodes())][["author_id", "community_label"]].copy()
    node_to_label = dict(node_df.itertuples(index=False, name=None))
    highlight_nodes = [node for node in graph.nodes() if node_to_label.get(node) == label]
    other_nodes = [node for node in graph.nodes() if node_to_label.get(node) != label]

    touched_edges = [(source, target) for source, target in graph.edges() if source in highlight_nodes or target in highlight_nodes]
    other_edges = [(source, target) for source, target in graph.edges() if (source, target) not in touched_edges]

    figure, axis = plt.subplots(figsize=(13, 7))
    nx.draw_networkx_edges(graph, pos=layout, edgelist=other_edges, ax=axis, edge_color="#E4E4E4", width=0.3, alpha=0.12)
    nx.draw_networkx_edges(graph, pos=layout, edgelist=touched_edges, ax=axis, edge_color="#B8B8B8", width=0.5, alpha=0.28)
    nx.draw_networkx_nodes(graph, pos=layout, nodelist=other_nodes, node_color="#D9D9D9", node_size=10, linewidths=0, alpha=0.55, ax=axis)
    nx.draw_networkx_nodes(graph, pos=layout, nodelist=highlight_nodes, node_color=color, node_size=26, linewidths=0, alpha=0.96, ax=axis)

    add_title_and_subtitle(
        axis,
        title=f"Community {label} In The Co-Commenting Network",
        subtitle=f"{users} users; dominant context: {dominant_context}; dominant category: {dominant_category}; edges indicate repeated co-commenting, not direct interaction",
    )
    axis.set_axis_off()
    figure.tight_layout()
    return figure


def main() -> None:
    """Run the community detection CLI."""
    args = parse_args()
    if args.threshold < 1:
        raise ValueError("Threshold must be a positive integer.")
    if args.min_community_size < 1:
        raise ValueError("Minimum community size must be a positive integer.")
    if args.top_highlight_communities < 1:
        raise ValueError("Top highlight communities must be a positive integer.")

    apply_matplotlib_style()

    edges_path = resolve_path(args.edges)
    nodes_path = resolve_path(args.nodes)
    user_video_path = resolve_path(args.user_video)
    comments_path = resolve_path(args.comments)
    output_users_path = resolve_path(args.output_users)
    output_summary_path = resolve_path(args.output_summary)
    output_video_profiles_path = resolve_path(args.output_video_profiles)
    summary_table_path = resolve_path(args.summary_table)
    figures_dir = resolve_path(args.figures_dir)

    for required_path in [edges_path, nodes_path, user_video_path]:
        if not required_path.exists():
            raise FileNotFoundError(f"Missing required file: {display_path(required_path)}")
    if not comments_path.exists():
        print_status(f"Optional comments file not found, continuing without it: {display_path(comments_path)}")

    edges_df = load_edges(edges_path, threshold=args.threshold)
    nodes_df = load_nodes(nodes_path)
    user_video_df = load_user_video(user_video_path)
    graph = build_main_graph(edges_df)
    print_status(
        f"Built main graph at weight >= {args.threshold} with {graph.number_of_nodes()} users and {graph.number_of_edges()} edges."
    )

    communities = order_communities(detect_louvain_partitions(graph, seed=args.seed))
    modularity_score = nx.algorithms.community.quality.modularity(graph, communities, weight="weight")

    user_communities_df, community_to_authors = build_user_communities(
        graph=graph,
        ordered_communities=communities,
        nodes_df=nodes_df,
    )
    community_video_profiles_df, community_user_video_df = build_community_video_profiles(
        user_communities_df=user_communities_df,
        user_video_df=user_video_df,
    )
    community_summary_df = build_community_summary(
        graph=graph,
        user_communities_df=user_communities_df,
        community_user_video_df=community_user_video_df,
        community_video_profiles_df=community_video_profiles_df,
        community_to_authors=community_to_authors,
    )
    summary_table_df = build_summary_table(
        threshold=args.threshold,
        graph=graph,
        community_summary_df=community_summary_df,
        modularity_score=modularity_score,
    )

    context_percentage_df, context_community_order, context_label_order = build_percentage_profile(
        community_video_profiles_df=community_video_profiles_df,
        community_summary_df=community_summary_df,
        value_column="community_comments_on_video",
        label_column="video_context_type",
        min_community_size=args.min_community_size,
    )
    category_percentage_df, category_community_order, category_label_order = build_percentage_profile(
        community_video_profiles_df=community_video_profiles_df,
        community_summary_df=community_summary_df,
        value_column="community_comments_on_video",
        label_column="primary_category",
        min_community_size=args.min_community_size,
    )

    largest_component_graph = largest_component_subgraph(graph)
    layout = compute_layout(largest_component_graph, seed=args.seed)
    top_highlight_df = community_summary_df.head(min(args.top_highlight_communities, len(community_summary_df))).copy()
    highlight_palette = build_community_palette(top_highlight_df["community_label"].tolist())

    write_csv_atomic(user_communities_df, output_users_path, overwrite=args.overwrite)
    write_csv_atomic(community_summary_df, output_summary_path, overwrite=args.overwrite)
    write_csv_atomic(community_video_profiles_df, output_video_profiles_path, overwrite=args.overwrite)
    write_csv_atomic(summary_table_df, summary_table_path, overwrite=args.overwrite)

    cleanup_stale_visualization_outputs(figures_dir)

    save_figure_atomic(create_community_size_figure(community_summary_df), figures_dir / "community_size_distribution.png", overwrite=args.overwrite)
    save_figure_atomic(
        create_community_size_long_tail_figure(community_summary_df),
        figures_dir / "community_size_long_tail.png",
        overwrite=args.overwrite,
    )
    save_figure_atomic(
        create_stacked_percentage_figure(
            percentage_df=context_percentage_df,
            community_order=context_community_order,
            label_order=context_label_order,
            label_column="video_context_type",
            title="Community Context Profile",
            subtitle="Video context composition for communities with at least 20 users, weighted by community comment participation",
        ),
        figures_dir / "community_context_profile.png",
        overwrite=args.overwrite,
    )
    save_figure_atomic(
        create_stacked_percentage_figure(
            percentage_df=category_percentage_df,
            community_order=category_community_order,
            label_order=category_label_order,
            label_column="primary_category",
            title="Community Category Profile",
            subtitle="Primary sampling category composition for communities with at least 20 users, weighted by community comment participation",
        ),
        figures_dir / "community_category_profile.png",
        overwrite=args.overwrite,
    )
    save_figure_atomic(
        create_network_overview_figure(
            graph=largest_component_graph,
            layout=layout,
            user_communities_df=user_communities_df,
            community_summary_df=community_summary_df,
            min_community_size=args.min_community_size,
        ),
        figures_dir / "community_network_overview.png",
        overwrite=args.overwrite,
    )

    for _, row in top_highlight_df.iterrows():
        label = str(row["community_label"])
        save_figure_atomic(
            create_community_highlight_figure(
                graph=largest_component_graph,
                layout=layout,
                user_communities_df=user_communities_df,
                community_summary_row=row,
                color=highlight_palette[label],
            ),
            figures_dir / f"community_highlight_{label}.png",
            overwrite=args.overwrite,
        )

    print_status(
        f"Detected {len(community_summary_df)} communities with modularity {modularity_score:.4f}. Largest community has {int(community_summary_df['users'].max()) if not community_summary_df.empty else 0} users."
    )


if __name__ == "__main__":
    main()
