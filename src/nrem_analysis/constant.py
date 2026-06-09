import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = Path(os.environ.get("DATA_PATH"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTERIM_DIR = DATA_DIR / "interim"

MOUSE_IDS_DUAL = ["99b", "100b", "102b", "103c", "106b", "107b", "110b", "111b"]