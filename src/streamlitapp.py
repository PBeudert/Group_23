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
page = st.radio("Go to", ["Main Page", "Chronological Info", "Movie Summarizer"], horizontal=True)


if page == "Main Page":
    st.header("Top Movie Types")
    
    # 🎨 Add color picker for movie types plot
    movie_color = st.color_picker("Click and Pick your favorite color!", "#87CEEB")  # skyblue default
    
    N = st.number_input("What number of movie types would you like to see", min_value=1, max_value=50, value=10, step=1)
    df_movie_type = processor.movie_type(N)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_movie_type["Movie_Type"], df_movie_type["Count"], color=movie_color)
    ax.set_xlabel("Movie Type")
    ax.set_ylabel("Count")
    ax.set_title("Histogram of Movie Types")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    st.header("Number of Movies versus Number of Actors")

    # 🎨 Add color picker for actor count plot
    actor_color = st.color_picker("Click and Pick your favorite color!", "#008000")  # green default

    df_actor_count = processor.actor_count()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(df_actor_count["Number_of_Actors"], df_actor_count["Movie_Count"], color=actor_color)
    ax.set_xlabel("Number of Actors")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Histogram of Number of Actors")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

    st.header("Actor Height Distribution")

    gender = st.selectbox("Select Gender", ["All", "Male", "Female"])
    if gender == "Male":
        gender = "M"
    elif gender == "Female":
        gender = "F"

    min_height = st.number_input("Enter Minimum Height (m)", min_value=1.0, max_value=2.1, value=1.5, step=0.1)
    max_height = st.number_input("Enter Maximum Height (m)", min_value=1.1, max_value=2.3, value=2.0, step=0.1)

    # 🎨 Add color picker for height distribution plot
    height_color = st.color_picker("Click and Pick your favorite color!", "#1E90FF")  # dodgerblue default

    # Get DataFrame and plot
    result_df = processor.actor_distributions(gender, max_height, min_height, plot=False)

    # Plot manually with selected color
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(result_df["Height"] * 100, bins=20, alpha=0.7, color=height_color, edgecolor="black")
    ax.set_xlabel("Height (cm)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Height Distribution ({gender})")
    st.pyplot(fig)


elif page == "Chronological Info":
    st.title("Chronological Movie Releases")
    
    # Dropdown for selecting genre
    available_genres = ["Action", "Comedy", "Drama", "Horror", "Romance", "Science Fiction", "Fantasy", "Thriller", "Documentary", "Animation"]
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
    
        # Add Birth Year/Month Plot Below the Previous Plot
    st.header("Actor Births Over Time")

    # Dropdown for Year vs. Month Selection
    time_selection = st.selectbox("Group Births By", ["Year", "Month"])

    # Convert selection to format used in `ages` method
    group_by = "Y" if time_selection == "Year" else "M"

    # Get the birth count data
    births_df = processor.ages(group_by)

    # Ensure all 12 months are represented (if using months)
    if group_by == "M":
        all_months = pd.DataFrame({"Month": range(1, 13)})
        births_df = all_months.merge(births_df, on="Month", how="left").fillna(0)

    # Plot the data
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(births_df.iloc[:, 0], births_df["Birth_Count"], color="purple", width=0.9)

    # Set axis labels
    ax.set_xlabel(time_selection)
    ax.set_ylabel("Number of Births")
    ax.set_title(f"Actor Births Per {time_selection}")

    # Adjust x-axis for Yearly and Monthly views
    if group_by == "Y":
        ax.set_xlim(left=1900, right=2010)  # Ensure years are properly displayed

    elif group_by == "M":
        ax.set_xticks(births_df["Month"])  # Ensure ticks align with bars
        ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", 
                            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
        ax.set_xlim(0.5, 12.5)  # Extend x-axis slightly to prevent cutoff

    # Show plot
    st.pyplot(fig)

elif page == "Movie Summarizer":
    st.title("Shuffle: Random Movie Classification")

    if st.button("Shuffle"):
        # 1) Grab a random movie
        movie_info = processor.get_random_movie()

        # 2) Display the random movie's title & summary in the first text box
        st.subheader("Random Movie Title & Summary")
        st.text_area(
            label="Title & Summary",
            value=f"Title: {movie_info.title}\n\nPlot Summary:\n{movie_info.summary}",
            height=200
        )

        # 3) Display the genres from the database in the second text box
        st.subheader("Genres (From Database)")
        st.text_area(
            label="Genres in DB",
            value=movie_info.genres,
            height=70
        )

        # 4) Classify the summary via your LLM
        llm_genres = processor.classify_genres_with_llm(movie_info.summary)

        # 5) Show the classification result in the third text box
        st.subheader("LLM Classification")
        st.text_area(
            label="LLM-decided Genres",
            value=llm_genres,
            height=70
        )

        # Store the genres for evaluation
        st.session_state["db_genres"] = movie_info.genres
        st.session_state["llm_genres"] = llm_genres

    # Evaluation Section
    if "db_genres" in st.session_state and "llm_genres" in st.session_state:
        if st.button("Evaluate Classification"):
            evaluation_result = processor.evaluate_llm_classification(
                st.session_state["db_genres"], st.session_state["llm_genres"]
            )

            st.subheader("Evaluation Result")
            st.text_area(
                label="LLM's Self-Evaluation",
                value=evaluation_result,
                height=150
            )
    else:
        st.write("Click the **Shuffle** button to pick a random movie and see its genres!")
    