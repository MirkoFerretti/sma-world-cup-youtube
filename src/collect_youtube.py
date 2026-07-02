"""Helpers for loading selected videos and collecting YouTube comments."""

from pathlib import Path


def load_selected_videos(metadata_path: Path):
    """Load the selected video list from metadata."""
    # TODO: Read the selected videos template and validate required columns.
    pass


def build_youtube_client(api_key: str):
    """Create an authenticated YouTube API client."""
    # TODO: Initialize the YouTube Data API client.
    pass


def collect_video_comments(video_id: str, api_key: str):
    """Collect comments for a single YouTube video."""
    # TODO: Request paginated comments for one video and store raw fields.
    pass


def collect_comments_for_videos(selected_videos_path: Path, output_path: Path, api_key: str):
    """Collect comments for all selected videos."""
    # TODO: Loop through selected videos and save raw comment data.
    pass
