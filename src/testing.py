import pandas as pd
from MovieDataProcessor import MovieDataProcessor

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

# Additional tests to validate functionality
print("\nChecking the total number of years in dataset:")
print(releases_df['Year'].nunique())

print("\nChecking if years are properly sorted:")
print(releases_df.sort_values('Year').head())

print("\nChecking if the genre filter works correctly:")
if not genre_releases_df.empty:
    print("Genre filter applied successfully.")
else:
    print("No movies found for the specified genre.")
