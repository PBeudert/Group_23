import streamlit as st
import pandas as pd
import numpy as np
from MovieDataProcessor import MovieDataProcessor

st.title("Movie Data Processor")

# Cache the processor so it doesn't reload on every UI interaction
@st.cache_data
def load_processor():
    return MovieDataProcessor()

processor = load_processor()

# User input for number of movie types
N = st.number_input("What number of movie types would you like to see", min_value=1, max_value=50, value=10, step=1)

# Display movie types
st.write("### Top Movie Types")
st.dataframe(processor.__movie_type__(N))  # Use user-selected value


