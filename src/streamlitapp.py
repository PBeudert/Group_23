import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from MovieDataProcessor import MovieDataProcessor

st.title("Movie Data Processor")

# Cache the processor so it doesn't reload on every UI interaction
@st.cache_data
def load_processor():
    return MovieDataProcessor()

processor = load_processor()

# Navigation Sidebar
st.sidebar.title("Navigation")
page = st.radio("Go to", ["Main Page", "Chronological Info"], horizontal=True)

if page == "Main Page":
    st.header("Top Movie Types")
    # User input for number of movie types
    N = st.number_input("What number of movie types would you like to see", min_value=1, max_value=50, value=10, step=1)
    
    # Display movie types
    df_movie_type = processor.movie_type(N)  # Use user-selected value
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_movie_type["Movie_Type"], df_movie_type["Count"], color="skyblue")
    ax.set_xlabel("Movie Type")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Movie Types")
    plt.xticks(rotation=45, ha="right")
    
    # Show the plot in Streamlit
    st.pyplot(fig)
    
    st.header("Number of Movies versus Number of Actors")
    df_actor_count = processor.actor_count()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_actor_count["Number_of_Actors"], df_actor_count["Movie_Count"], color="green")
    ax.set_xlabel("Number of Actors")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Histogram of Number of Actors")
    plt.xticks(rotation=45, ha="right")
    
    st.pyplot(fig)
    
    st.header("Actor Height Distribution")
    # Dropdown for gender
    gender = st.selectbox("Select Gender", ["All", "Male", "Female"])
    if gender == "Male":
        gender = "M"
    elif gender == "Female":
        gender = "F"
    
    # Input fields for height range
    min_height = st.number_input("Enter Minimum Height (m)", min_value=1.0, max_value=2.1, value=1.5, step=0.1)
    max_height = st.number_input("Enter Maximum Height (m)", min_value=1.1, max_value=2.3, value=2.0, step=0.1)
    
    result_df = processor.actor_distributions(gender, max_height, min_height, plot=True)
    
    # Ensure Streamlit properly renders the plot
    st.pyplot(plt)

elif page == "Chronological Info":
    st.title("Chronological Movie Releases")
    
    # Dropdown for selecting genre
    available_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Fantasy", "Thriller", "Documentary", "Animation"]
    selected_genre = st.selectbox("Select a genre", [None] + available_genres)
    
    # Retrieve the data
    releases_df = processor.releases(selected_genre)
    
    if not releases_df.empty:
        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(releases_df["Year"], releases_df["Movie_Count"], color="blue")
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Movies Released")
        ax.set_xlim(left=1900) 
        ax.set_xlim(right=2030) 
        ax.set_title(f"Movie Releases Over Time ({'All Genres' if not selected_genre else selected_genre})")
        plt.xticks(rotation=45)
        
        # Show the plot
        st.pyplot(fig)
    else:
        st.write("No data available for the selected genre.")