import os
import pandas as pd
import requests
import zipfile
from pathlib import Path
from pydantic import BaseModel
from typing import Optional

class MovieDataProcessor:
    """Class to handle the CMU Movie Corpus dataset."""
    
    DATA_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    DOWNLOAD_DIR = Path("downloads/")
    FILE_NAME = DOWNLOAD_DIR / "MovieSummaries.tar.gz"

    def __init__(self):
        """Initialize class by downloading, extracting, and loading the dataset."""
        self._ensure_download_dir()
        self._download_data()
        self._extract_data()
        self._load_data()

    def _ensure_download_dir(self):
        """Ensure the download directory exists."""
        self.DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    def _download_data(self):
        """Download the dataset if it does not already exist."""
        if not self.FILE_NAME.exists():
            print("Downloading dataset...")
            response = requests.get(self.DATA_URL, stream=True)
            with open(self.FILE_NAME, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print("Download complete.")

    def _extract_data(self):
        """Extract dataset if not already extracted."""
        extracted_path = self.DOWNLOAD_DIR / "MovieSummaries"
        if not extracted_path.exists():
            print("Extracting dataset...")
            with zipfile.ZipFile(self.FILE_NAME, "r") as zip_ref:
                zip_ref.extractall(self.DOWNLOAD_DIR)
            print("Extraction complete.")

    def _load_data(self):
        """Load datasets into pandas DataFrames."""
        movie_file = self.DOWNLOAD_DIR / "MovieSummaries/MovieSummaries.txt"
        if movie_file.exists():
            self.movies_df = pd.read_csv(movie_file, delimiter="\t", header=None)
            print("Data loaded successfully.")
        else:
            print("Dataset file not found.")
