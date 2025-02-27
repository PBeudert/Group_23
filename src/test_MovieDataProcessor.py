import pytest
import pandas as pd
from unittest.mock import patch
from MovieDataProcessor import MovieDataProcessor
import ast

@pytest.fixture
def mock_processor():
    """
    Creates a mock instance of MovieDataProcessor with download and extraction disabled.
    """
    with patch.object(MovieDataProcessor, "_download_data"), \
         patch.object(MovieDataProcessor, "_extract_data"), \
         patch.object(MovieDataProcessor, "_load_data"):
        processor = MovieDataProcessor()
    return processor


def test_movie_type_valid(mock_processor):
    """
    Test the __movie_type__ method returns correct top-N results with mock data.
    """
    mock_data = pd.DataFrame({
        0: [1, 2, 3, 4],  # Ensure numeric column names match actual dataset
        8: [
            {"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction"},  # Dict, not a string
            {"/m/02kdv5l": "Action", "/m/03k9fj": "Adventure"},
            {"/m/07s9rl0": "Drama"},
            {"/m/03bxz7": "Biographical film", "/m/07s9rl0": "Drama"}
        ]
    })
    
    mock_processor.movie_metadata = mock_data
    
    top_2 = mock_processor.__movie_type__(N=2)
    
    # Expected genres and counts:
    # "Drama" appears in two movies
    # "Thriller", "Science Fiction", "Action", "Adventure", "Biographical film" appear once each
    genre_list = top_2["Movie_Type"].tolist()
    
    assert "Drama" in genre_list  # "Drama" should be the most common genre
    assert len(top_2) == 2  # Only the top 2 should be returned


def test_movie_type_valid(mock_processor):
    """
    Test the __movie_type__ method returns correct top-N results with mock data.
    """
    mock_data = pd.DataFrame({
        0: [1, 2, 3, 4],  # Column 0 is Movie ID
        8: [
            {"/m/01jfsb": "Thriller", "/m/06n90": "Science Fiction"},  
            {"/m/02kdv5l": "Action", "/m/03k9fj": "Adventure"},
            {"/m/07s9rl0": "Drama"},
            {"/m/03bxz7": "Biographical film", "/m/07s9rl0": "Drama"}
        ]
    })
    
    # Manually assign column names to match real dataset indexing behavior
    mock_data.columns = range(len(mock_data.columns))  # Force integer-based column indexing

    # Assign to mock processor
    mock_processor.movie_metadata = mock_data
    
    # Ensure column 8 exists before proceeding
    assert 8 in mock_processor.movie_metadata.columns, "Column 8 is missing in the dataset!"

    # Call method and check output
    top_2 = mock_processor.__movie_type__(N=2)
    
    # Expected genres and counts:
    # "Drama" appears twice
    # "Thriller", "Science Fiction", "Action", "Adventure", "Biographical film" appear once each
    genre_list = top_2["Movie_Type"].tolist()
    
    assert "Drama" in genre_list  # "Drama" should be the most common genre
    assert len(top_2) == 2  # Only the top 2 should be returned


def test_movie_type_key_error(mock_processor):
    """
    Test __movie_type__ raises KeyError if N exceeds available genre types.
    """
    mock_data = pd.DataFrame({8: [{"/m/02kdv5l": "Action"}]})  # Only 1 genre exists
    mock_processor.movie_metadata = mock_data
    
    with pytest.raises(KeyError, match="N is larger than the available movie types"):
        mock_processor.__movie_type__(N=2)
