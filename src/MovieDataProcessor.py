import os
import pandas as pd
import requests
import tarfile
import ast
import matplotlib.pyplot as plt
from pathlib import Path



class MovieDataProcessor:
    """Class to handle the CMU Movie Corpus dataset."""
    
    DATA_URL = "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"
    DOWNLOAD_DIR = Path("downloads/")
    FILE_NAME = DOWNLOAD_DIR / "MovieSummaries.tar.gz"
    EXTRACTED_DIR = DOWNLOAD_DIR / "MovieSummaries"


    def __init__(self):

        if self.FILE_NAME.exists() and self.EXTRACTED_DIR.exists():
            print("Dataset already downloaded and extracted. Skipping...")
        else:
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

    def __movie_type__(self, N=10):
        """
        Returns a DataFrame with the N most common movie types and their counts.
        :param N: int, the number of top movie types to return.
        :return: pandas DataFrame with columns ['Movie_Type', 'Count']
        """
        if not isinstance(N, int) or N<1:
            raise ValueError("N must be a postitive integer")
        
        # Ensure the DataFrame has column names
        self.movie_metadata.columns = [str(i) for i in range(len(self.movie_metadata.columns))]
        
        # Extract the movie types from column 8
        all_genres = []
        for row in self.movie_metadata['8'].dropna():
            try:
                genre_dict = ast.literal_eval(row) if isinstance(row, str) else row
                all_genres.extend(genre_dict.values())
            except (SyntaxError, ValueError):
                continue
        
        # Count occurrences of each movie type
        type_counts = pd.Series(all_genres).value_counts().reset_index()
        type_counts.columns = ['Movie_Type', 'Count']
        if N>len(type_counts):
            av_movies=len(type_counts)
            raise KeyError(f"N is larger than the available movie types of {len(type_counts)}")
        # Return the top N movie types
        return type_counts.head(N)

        #- [ ] Develop a second method called __actor_count__. This method calculates a pandas dataframe with a histogram of "number of actors" vs "movie counts".

    def __actor_count__(self):
        """
        Computes a histogram of the number of actors per movie.
        Returns a DataFrame with columns ['Number_of_Actors', 'Movie_Count'].
        """
        # 1) Group character_metadata by the movie ID (in column 0) and count rows
        actor_counts = self.character_metadata.groupby(0).size()  
        #   => This maps each movie ID to its number of actors
        
        # 2) Convert that mapping into a histogram:
        #    - .value_counts() tells how many movies have X actors
        hist = actor_counts.value_counts().reset_index()
        hist.columns = ['Number_of_Actors', 'Movie_Count']
        
        # 3) Sort by the number of actors (ascending)
        hist = hist.sort_values('Number_of_Actors').reset_index(drop=True)
        
        return hist

    def __actor_distributions__(
            self,
            gender: str,
            max_height: float,
            min_height: float,
            plot: bool = False
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of height distributions filtered by gender and height range.
        Optionally plots the distribution.

        :param gender: str
            Accepts "All" or one of the distinct non-missing gender values in the dataset.
        :param max_height: float
            The maximum height (in meters) to filter actors.
        :param min_height: float
            The minimum height (in meters) to filter actors.
        :param plot: bool
            If True, plots the height distribution histogram.
        :return: pd.DataFrame
            A DataFrame with columns ['Height', 'Count'] representing height distribution.
        :raises ValueError:
            If gender is not a string or if height values are not numerical.
        """

        # Ensure column names are correctly set
        self.character_metadata.columns = [str(i) for i in range(len(self.character_metadata.columns))]

        df_actors = self.character_metadata.copy()

        # Define correct column mappings
        HEIGHT_COLUMN = "6"  # Actor height in meters
        GENDER_COLUMN = "5"  # Actor gender

        # Drop rows where gender or height is missing
        df_actors = df_actors[df_actors[GENDER_COLUMN].notna() & df_actors[HEIGHT_COLUMN].notna()]


        available_genders=df_actors[GENDER_COLUMN].unique()

        # Ensure input types are correct
        if not isinstance(gender, str):
            raise ValueError(f"Gender must be a string.")
        if not gender in available_genders:
            raise ValueError(f"available genders are {available_genders}")
        if not isinstance(max_height, (int, float)) or not isinstance(min_height, (int, float)):
            raise ValueError("Max and min heights must be numerical values.")



        # Filter by gender (except when "All" is selected)
        if gender != "All":
            df_actors = df_actors[df_actors[GENDER_COLUMN] == gender]

        # Convert height column to numeric (it should be in meters)
        try:
            df_actors[HEIGHT_COLUMN] = df_actors[HEIGHT_COLUMN].astype(float)
        except ValueError:
            raise ValueError("Some height values are non-numeric and cannot be converted to float.")

        # Filter by height range
        df_actors = df_actors[(df_actors[HEIGHT_COLUMN] >= min_height) & (df_actors[HEIGHT_COLUMN] <= max_height)]

        # Return a height histogram
        height_counts = df_actors[HEIGHT_COLUMN].value_counts().sort_index().reset_index()
        height_counts.columns = ["Height", "Count"]

        # Optional: plot histogram
        if plot:
            plt.figure(figsize=(7, 5))
            plt.hist(df_actors[HEIGHT_COLUMN], bins=20, edgecolor="black", alpha=0.7)
            plt.xlabel("Actor Height (meters)")
            plt.ylabel("Frequency")
            plt.title(f"Height Distribution (Gender: {gender}, Range: {min_height}-{max_height})")
            plt.show()

        return height_counts

    def debug_column_4(self):
        """Prints column 4 of the character metadata dataset for debugging."""
        if hasattr(self, "character_metadata"):
            print("First 10 values of column 4:")
            print(self.character_metadata[4].head(10))  # Show first 10 rows

            print("\nUnique values (first 20):")
            print(self.character_metadata[4].unique()[:20])  # Show first 20 unique values

            print("\nData type of column 4:")
            print(self.character_metadata[4].dtype)
        else:
            print("Error: character_metadata is not loaded.")







