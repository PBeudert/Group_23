import os
import pandas as pd
import requests
import tarfile
import ast
import matplotlib.pyplot as plt
from pathlib import Path
import ollama
from ollama import chat, ChatResponse

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
        """Load datasets into pandas DataFrames, converting columns to string labels."""
        print("Loading datasets into pandas DataFrames...")
        try:
            self.character_metadata = pd.read_csv(self.EXTRACTED_DIR / "character.metadata.tsv", 
                                                 sep='\t', header=None)
            self.movie_metadata = pd.read_csv(self.EXTRACTED_DIR / "movie.metadata.tsv", 
                                              sep='\t', header=None)
            self.name_clusters = pd.read_csv(self.EXTRACTED_DIR / "name.clusters.txt", 
                                             sep='\t', header=None)
            self.plot_summaries = pd.read_csv(self.EXTRACTED_DIR / "plot_summaries.txt", 
                                              sep='\t', header=None)
            self.tv_tropes_clusters = pd.read_csv(self.EXTRACTED_DIR / "tvtropes.clusters.txt", 
                                                 sep='\t', header=None)

            # Assign column names to movie_metadata
            self.movie_metadata.columns = [
                "Movie_ID", "Freebase_ID", "Movie_Title", "Release_Date", "Revenue",
                "Runtime", "Languages", "Countries", "Genres"
            ]

            # Assign column names to character_metadata
            self.character_metadata.columns = [
                "Movie_ID", "Freebase_ID", "Release_Date", "Character_Name", "Actor_Birthdate",
                "Actor_Gender", "Actor_Height", "Actor_Ethnicity", "Actor_Name", "Actor_Age",
                "Freebase_Char_ID_1", "Freebase_Char_ID_2", "Freebase_Char_ID_3"
            ]


            # Assign column names to name_clusters
            self.name_clusters.columns = [
                "Character_Name", "Freebase_ID"
            ]

            # Assign column names to plot_summaries
            self.plot_summaries.columns = [
                "Movie_ID", "Plot_Summary"
            ]

            # Assign column names to tv_tropes_clusters
            self.tv_tropes_clusters.columns = [
                "Trope", "Character_Movie_Details"
            ]

        except Exception as e:
            print(f"Error loading datasets: {e}")

    def movie_type(self, N=10):
        """
        Returns a DataFrame with the N most common movie types and their counts.
        :param N: int, the number of top movie types to return.
        :return: pandas DataFrame with columns ['Movie_Type', 'Count']
        """
        if not isinstance(N, int) or N < 1:
            raise ValueError("N must be a positive integer")

        # Ensure we have the correct genre column
        if "Genres" not in self.movie_metadata.columns:
            raise KeyError("Column 'Genres' not found in movie_metadata. Check dataset format.")

        # Extract the movie types from the Genres column
        all_genres = []
        for row in self.movie_metadata["Genres"].dropna():
            try:
                # row could be a string or already a dict
                genre_dict = ast.literal_eval(row) if isinstance(row, str) else row
                all_genres.extend(genre_dict.values())
            except (SyntaxError, ValueError):
                continue

        # Count occurrences of each movie type
        type_counts = pd.Series(all_genres).value_counts().reset_index()
        type_counts.columns = ['Movie_Type', 'Count']

        if N > len(type_counts):
            raise KeyError("N is larger than the available movie types")

        return type_counts.head(N)

    def actor_count(self):
        """
        Computes a histogram of the number of actors per movie.
        Returns a DataFrame with columns ['Number_of_Actors', 'Movie_Count'].
        """
        # Group by "Movie_ID" instead of column "0"
        actor_counts = self.character_metadata.groupby("Movie_ID").size()
        
        # Compute histogram of number of actors per movie
        hist = actor_counts.value_counts().reset_index()
        hist.columns = ['Number_of_Actors', 'Movie_Count']
        hist = hist.sort_values('Number_of_Actors').reset_index(drop=True)
        
        return hist
        
    def actor_distributions(
            self,
            gender: str,
            max_height: float,
            min_height: float,
            plot: bool = False
    ) -> pd.DataFrame:
        """
        Returns a DataFrame of height distributions filtered by gender and height range.
        Optionally plots the distribution.
        """

        # Ensure columns exist
        if "Actor_Gender" not in self.character_metadata.columns or "Actor_Height" not in self.character_metadata.columns:
            raise KeyError("Required columns 'Actor_Gender' or 'Actor_Height' not found in character_metadata.")

        df_actors = self.character_metadata.copy()

        # ✅ Keep only numeric height values (remove invalid ones)
        df_actors = df_actors[df_actors["Actor_Height"].astype(str).str.match(r'^\d+(\.\d+)?$', na=False)]

        # ✅ Convert height to float after removing invalid values
        df_actors["Actor_Height"] = df_actors["Actor_Height"].astype(float)

        # Validate input types
        if not all(isinstance(val, (int, float)) for val in [min_height, max_height]):
            raise ValueError("Max and min heights must be numerical values.")

        if min_height >= max_height:
            raise ValueError("min_height must be less than max_height.")

        # Filter by gender
        available_genders = df_actors["Actor_Gender"].dropna().unique()
        if gender != "All" and gender not in available_genders:
            raise ValueError(f"Invalid gender selection. Available options: {available_genders}")
        if gender != "All":
            df_actors = df_actors[df_actors["Actor_Gender"] == gender]

        # Filter height range
        df_actors = df_actors[(df_actors["Actor_Height"] >= min_height) & (df_actors["Actor_Height"] <= max_height)]

        # Check if data remains after filtering
        if df_actors.empty:
            print("Warning: No actors found in the given height range.")
            return pd.DataFrame(columns=["Height", "Count"])

        # Build the histogram
        height_counts = df_actors["Actor_Height"].value_counts().sort_index().reset_index()
        height_counts.columns = ["Height", "Count"]

        # Optional plot
        if plot:
            plt.figure(figsize=(7, 5))
            plt.hist(df_actors["Actor_Height"], bins=20, edgecolor="black", alpha=0.7)
            plt.xlabel("Actor Height in Meters")
            plt.ylabel("Frequency")
            plt.title(f"Height Distribution For {gender} Actors ({min_height}m - {max_height}m)")
            plt.show()

        return height_counts
    def releases(self, genre=None):
        """
        Returns a DataFrame showing the number of movie releases per year.
        If a genre is specified, it filters only movies of that genre.

        :param genre: str or None, the genre to filter movies by (default: None, includes all movies)
        :return: pandas DataFrame with columns ['Year', 'Movie_Count']
        """

        # Ensure required columns exist
        if "Release_Date" not in self.movie_metadata.columns or "Genres" not in self.movie_metadata.columns:
            raise KeyError("Required columns 'Release_Date' or 'Genres' not found in movie_metadata.")

        # Extract relevant columns
        df_movies = self.movie_metadata[["Release_Date", "Genres"]].copy()

        # Drop missing years
        df_movies = df_movies.dropna(subset=["Release_Date"])

        # Extract the year from dates (if applicable)
        df_movies["Year"] = df_movies["Release_Date"].astype(str).str.extract(r"(\d{4})")  # Extract four-digit year

        # Convert to integer after extraction
        df_movies = df_movies.dropna(subset=["Year"])
        df_movies["Year"] = df_movies["Year"].astype(int)

        # If a genre is specified, filter movies that contain that genre
        if genre:
            def genre_filter(genre_dict):
                try:
                    parsed_genres = ast.literal_eval(genre_dict)  # Convert string to dictionary
                    return genre in parsed_genres.values()
                except (SyntaxError, ValueError):
                    return False

            df_movies = df_movies[df_movies["Genres"].apply(genre_filter)]
            print(f"Data shape after filtering by genre '{genre}':", df_movies.shape)

        # If no valid data remains after filtering, return empty DataFrame
        if df_movies.empty:
            print(f"No movies found for genre '{genre}'.")
            return pd.DataFrame(columns=["Year", "Movie_Count"])

        # Count movies per year
        releases_per_year = df_movies.groupby("Year").size().reset_index(name="Movie_Count")

        return releases_per_year

    def ages(self, group_by="Y"):
        """
        Computes a DataFrame counting actor births per year ('Y') or per month ('M').
        
        :param group_by: str, 'Y' for year (default) or 'M' for month.
        :return: pandas DataFrame with columns ['Year' or 'Month', 'Birth_Count']
        """
        # Ensure the required column exists
        if "Actor_Birthdate" not in self.character_metadata.columns:
            raise KeyError("Required column 'Actor_Birthdate' not found in character_metadata.")

        # Drop missing birthdates
        df_births = self.character_metadata.dropna(subset=["Actor_Birthdate"]).copy()

        # Extract Year and Month from birthdate
        df_births["Year"] = df_births["Actor_Birthdate"].astype(str).str.extract(r"(\d{4})")
        df_births["Month"] = df_births["Actor_Birthdate"].astype(str).str.extract(r"-(\d{2})-")

        # Convert to numeric values
        df_births["Year"] = pd.to_numeric(df_births["Year"], errors="coerce")
        df_births["Month"] = pd.to_numeric(df_births["Month"], errors="coerce")

        # Default to yearly count if an invalid option is passed
        if group_by not in ["Y", "M"]:
            group_by = "Y"

        # Count occurrences
        if group_by == "Y":
            birth_counts = df_births.groupby("Year").size().reset_index(name="Birth_Count")
        else:  # group_by == "M"
            birth_counts = df_births.groupby("Month").size().reset_index(name="Birth_Count")

        return birth_counts
    

    def generate_movie_summary(self, movie_title):
        """
        Uses Ollama to generate a summary for a given movie title.

        :param movie_title: str, the title of the movie
        :return: str, the generated summary
        """
        prompt = f"Provide a short and engaging summary for the movie '{movie_title}'."
        
        try:
            response = ollama.chat("mistral", messages=[{"role": "user", "content": prompt}])
            return response["message"]
        except Exception as e:
            return f"Error generating summary: {e}"
