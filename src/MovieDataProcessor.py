import os
import pandas as pd
import requests
import tarfile
from pathlib import Path


class MovieDataProcessor:
    """Class to handle the CMU Movie Corpus dataset."""
    
    DATA_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    DOWNLOAD_DIR = Path("downloads/")
    FILE_NAME = DOWNLOAD_DIR / "MovieSummaries.tar.gz"
    EXTRACTED_DIR = DOWNLOAD_DIR / "MovieSummaries"


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
            try:
                with tarfile.open(self.FILE_NAME, "r:gz") as tar:
                    tar.extractall(self.DOWNLOAD_DIR)
                print("Extraction complete.")
            except tarfile.TarError:
                print("Error: File is not a valid tar.gz archive.")

    def _load_data(self):
        """Load datasets into pandas DataFrames."""
        print("Loading datasets into pandas DataFrames...")
        try:
            self.character_metadata = pd.read_csv(self.EXTRACTED_DIR / "character.metadata.tsv", sep='\t', header=None)
            self.movie_metadata = pd.read_csv(self.EXTRACTED_DIR / "movie.metadata.tsv", sep='\t', header=None)
            self.name_clusters = pd.read_csv(self.EXTRACTED_DIR / "name.clusters.txt", sep='\t', header=None)
            self.plot_summaries = pd.read_csv(self.EXTRACTED_DIR / "plot_summaries.txt", sep='\t', header=None)
            self.tv_tropes_clusters = pd.read_csv(self.EXTRACTED_DIR / "tvtropes.clusters.txt", sep='\t', header=None)

            print("Datasets successfully loaded!")
        except Exception as e:
            print(f"Error loading datasets: {e}")

