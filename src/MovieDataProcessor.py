import os
import pandas as pd
import requests
import tarfile
import ast
import matplotlib.pyplot as plt
from pathlib import Path
import ollama
from openai import OpenAI  
import random
from typing import Optional
from pydantic import BaseModel, field_validator
from pydantic.functional_validators import field_validator

class MovieInfo(BaseModel):
    title: str
    summary: str
    genres: str

    @field_validator("title")
    def title_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Title must be a non-empty string.")
        return v

    @field_validator("summary")
    def summary_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Summary must be a non-empty string.")
        return v
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

            ### THis is new
            self.merged_df = pd.merge(
            self.movie_metadata, 
            self.plot_summaries,
            on="Movie_ID",
            how="inner"  # only keep rows that exist in both
            )       
            ####   

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
        



        
    def parse_genre_dictionary(self, genre_dict_str: str) -> str:
            """
            Some rows in 'Genres' might be stored like:
            {"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction"}
            This helper function extracts just the values (e.g. "Thriller, Science Fiction").

            :param genre_dict_str: A string representation of a dict
            :return: Comma-separated string of genre values
            """
            # In many datasets, "Genres" might already be a Python dict. 
            # If it's a string that *looks* like a dict, we can safely eval it 
            # (but be aware of security concerns; for a hackathon it’s probably fine).
            if not genre_dict_str:
                return ""

            try:
                genre_dict = eval(genre_dict_str)
                if isinstance(genre_dict, dict):
                    return ", ".join(genre_dict.values())
                else:
                    # If it’s not a dict, just return the original or handle accordingly
                    return str(genre_dict_str)
            except:
                # If parsing fails, just return the raw string
                return str(genre_dict_str)

    def get_random_movie(self) -> MovieInfo:
        """
        Picks a random row from the merged DataFrame, parses the genres,
        and returns a MovieInfo pydantic model.
        """
        if self.merged_df.empty:
            raise ValueError("No movie data available.")

        random_idx = random.randint(0, len(self.merged_df) - 1)
        row = self.merged_df.iloc[random_idx]

        title = row["Movie_Title"]
        summary = row["Plot_Summary"]
        parsed_genres = self.parse_genre_dictionary(row["Genres"])

        # Build the pydantic model
        movie_info = MovieInfo(
            title=title,
            summary=summary,
            genres=parsed_genres
        )
        return movie_info

    def classify_genres_with_llm(self, summary: str) -> str:
        """
        Calls your local LLM (e.g. Ollama) to classify the movie's summary into genres.
        Prompt-engineer it so it tries to ONLY output the genre(s).
        """
        # Example prompt:
        prompt = f"""
You are a helpful assistant that classifies a movie plot into appropriate genres.
Read the following summary and output ONLY the genres that best describe it.
No explanations, no extra text. Just the genre(s).

Summary:
{summary}
        """

        try:
            response = ollama.chat(
                "mistral",  # or whichever model name
                messages=[{"role": "user", "content": prompt}]
            )
            # Print or inspect the structure of the response
            print(response)  # For debugging, check the actual structure

            if hasattr(response, "message") and hasattr(response.message, "content"):
                llm_output = response.message.content.strip()  # Accessing 'content' safely
            else:
                return "Error: Unexpected response format from LLM."

            if not llm_output:
                return "No genres found in response."

            return llm_output
        except Exception as e:
            return f"Error during classification: {e}"
        
    def evaluate_llm_classification(self, db_genres: str, llm_genres: str) -> str:
        """
        Asks the LLM to compare the genres from the database with its own classified genres
        and determine if the classification was correct.
        """

        prompt = f"""
    You are a movie classification assistant.

    Your task:
    1. You will receive two sets of genres: 
    - 'Database Genres': The actual genres from a movie database.
    - 'LLM Genres': The genres you classified based on the movie summary.
    
    2. Compare these two lists:
    - Are they similar?
    - Did you add any incorrect genres?
    - Did you miss any important genres?
    - Give a final judgment: Did you classify this movie correctly? Answer with "Yes" or "No".

    **Database Genres:** {db_genres}  
    **LLM Genres:** {llm_genres}  

    Respond in a structured format:
    - Correct Classification: (Yes/No)
    - Extra Genres Added: (List if any)
    - Missing Genres: (List if any)
    - Final Thoughts: (Brief explanation)
    """

        try:
            response = ollama.chat(
                "mistral",  # or whichever model you are using
                messages=[{"role": "user", "content": prompt}]
            )

            # Extract response safely
            if hasattr(response, "message") and hasattr(response.message, "content"):
                evaluation = response.message.content.strip()
            else:
                return "Error: Unexpected response format from LLM."

            return evaluation
        except Exception as e:
            return f"Error during evaluation: {e}"

    def personalize_movie_plot(self, name: str, api_key: str) -> str:
        """
        Creates a personalized movie plot based on a random movie but with the given name as the main character.
        
        This function:
        1. Gets a random movie from the dataset
        2. Uses OpenAI to rewrite the plot with the provided name as the main character
        3. Returns the personalized plot
        
        :param name: str, the name of the person to make the main character
        :param api_key: str, the OpenAI API key to use for the request
        :return: str, the personalized movie plot
        
        :raises ValueError: If name is empty or invalid or if the API key is empty
        :raises RuntimeError: If there's an issue with OpenAI API processing
        """
        # Validate input
        if not name or not isinstance(name, str) or not name.strip():
            raise ValueError("Name must be a non-empty string.")
        
        if not api_key or not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("API key must be a non-empty string.")
        
        name = name.strip()
        api_key = api_key.strip()
        
        try:
            # Get a random movie
            movie_info = self.get_random_movie()
            
            # Create prompt for OpenAI
            prompt = f"""
    Rewrite the following movie plot to make a person named "{name}" the main character.
    Make appropriate adjustments to the storyline to naturally incorporate "{name}" as the protagonist,
    while keeping the core plot elements, setting, and theme intact.
    
    Original Movie: "{movie_info.title}"
    Original Plot: {movie_info.summary}
    
    Please provide ONLY the rewritten plot, no explanations or other text.
    """
            
            try:
                # Create a temporary OpenAI client with the provided API key
                client = OpenAI(api_key=api_key)
                
                # Call OpenAI API
                response = client.chat.completions.create(
                    model="gpt-4o",  
                    messages=[
                        {"role": "system", "content": "You are a creative assistant that rewrites movie plots."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1000
                )
                
                # Extract response safely from OpenAI format
                if hasattr(response, "choices") and len(response.choices) > 0:
                    personalized_plot = response.choices[0].message.content.strip()
                else:
                    return f"Failed to personalize plot for '{movie_info.title}'. Unexpected response format."
                
                if not personalized_plot:
                    return f"Failed to generate personalized plot for '{movie_info.title}'."
                
                # Add information about the original movie
                result = f"Personalized plot for '{name}' based on '{movie_info.title}':\n\n{personalized_plot}"
                return result
                
            except Exception as e:
                return f"Error while personalizing plot with OpenAI: {str(e)}"
        
        except ValueError as ve:
            return f"Error getting random movie: {str(ve)}"