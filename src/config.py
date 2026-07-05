"""Project configuration and shared paths."""

from pathlib import Path
import os

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FINAL_DATA_DIR = DATA_DIR / "final"
METADATA_DIR = ROOT_DIR / "metadata"
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
NETWORKS_DIR = OUTPUTS_DIR / "networks"

load_dotenv(ROOT_DIR / ".env")

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")