import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
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
df= processor.__movie_type__(N)  # Use user-selected value

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df["Movie_Type"], df["Count"], color="skyblue")
ax.set_xlabel("Movie Type")
ax.set_ylabel("Count")
ax.set_title("Histogram of Movie Types")
plt.xticks(rotation=45, ha="right")

# Show the plot in Streamlit
st.pyplot(fig)
