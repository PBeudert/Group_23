import pandas as pd
from movie_data_processor import MovieDataProcessor

# Instantiate the processor
processor = MovieDataProcessor()

# Test releases method without genre filter
print("Testing releases() without genre filter...")
releases_df = processor.releases()
print(releases_df.head())

# Test releases method with a specific genre
test_genre = "Comedy"
print(f"Testing releases() with genre '{test_genre}'...")
genre_releases_df = processor.releases(test_genre)
print(genre_releases_df.head())
