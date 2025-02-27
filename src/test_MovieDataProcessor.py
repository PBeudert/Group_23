import pytest
from unittest.mock import patch
import pandas as pd
from MovieDataProcessor import MovieDataProcessor

@pytest.fixture
def mock_processor():
    """Mocked MovieDataProcessor without real downloading or extraction."""
    with patch.object(MovieDataProcessor, "_download_data"), \
         patch.object(MovieDataProcessor, "_extract_data"), \
         patch.object(MovieDataProcessor, "_load_data"):
        processor = MovieDataProcessor()
    return processor

def test_movie_type_valid(mock_processor):
    """Test __movie_type__ returns correct top-N results."""
    mock_data = pd.DataFrame({
        "0": [1, 2, 3, 4],  # Movie ID column
        "8": [
            "{'/m/01jfsb': 'Thriller', '/m/06n90': 'Science Fiction'}",
            "{'/m/02kdv5l': 'Action', '/m/03k9fj': 'Adventure'}",
            "{'/m/07s9rl0': 'Drama'}",
            "{'/m/03bxz7': 'Biographical film', '/m/07s9rl0': 'Drama'}"
        ]
    })
    mock_processor.movie_metadata = mock_data

    top_2 = mock_processor.__movie_type__(N=2)
    assert len(top_2) == 2
    assert "Drama" in top_2["Movie_Type"].values

def test_movie_type_too_large(mock_processor):
    """Test __movie_type__ raises KeyError if N > available genres."""
    mock_data = pd.DataFrame({
        "8": ["{'/m/01': 'Action'}"]  # Column 8 must be string-labeled
    })
    mock_processor.movie_metadata = mock_data

    with pytest.raises(KeyError, match="N is larger than the available movie types"):
        mock_processor.__movie_type__(N=2)  # Only 1 genre, but we request 2

def test_actor_count(mock_processor):
    """Test __actor_count__ for actor histogram."""
    mock_data = pd.DataFrame({
        "0": [101, 101, 102, 103, 103, 103],  # Movie ID column
        "1": ["actor1", "actor2", "actor3", "actor4", "actor5", "actor6"]
    })
    mock_processor.character_metadata = mock_data

    hist_df = mock_processor.__actor_count__()
    assert len(hist_df) == 3  # Movies have 1, 2, and 3 actors
    assert list(hist_df["Number_of_Actors"]) == [1, 2, 3]
    assert list(hist_df["Movie_Count"]) == [1, 1, 1]
