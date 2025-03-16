import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from MovieDataProcessor import MovieDataProcessor

# Configure page settings - must be first Streamlit command
st.set_page_config(
    page_title="Movie Data Processor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize theme in session state if it doesn't exist
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Store height values in session state to maintain consistency
if "min_height" not in st.session_state:
    st.session_state.min_height = 1.5
if "max_height" not in st.session_state:
    st.session_state.max_height = 2.0

# Theme toggle in sidebar
st.sidebar.title("Settings")
with st.sidebar.expander("Theme Settings"):
    # Create radio buttons for theme selection
    selected_theme = st.radio(
        "Choose Theme",
        options=["Light", "Dark"],
        index=0 if st.session_state.theme == "light" else 1,
        key="theme_selector"
    )
    
    # Apply theme based on the selection
    if selected_theme.lower() == "dark":
        st.session_state.theme = "dark"
    else:
        st.session_state.theme = "light"

# Apply theme CSS based on current session state
if st.session_state.theme == "dark":
    # Apply dark theme
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }
    .stRadio label {
        color: #FFFFFF !important;
        font-weight: 500 !important;
    }
    /* Fix other elements in dark mode */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #FFFFFF !important;
    }
    /* Fix info boxes in dark mode */
    .stAlert {
        color: #FAFAFA !important;
    }
    .stAlert a {
        color: #4DADFF !important;
    }
    /* Fix all text elements */
    p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #FAFAFA !important;
    }
    </style>
    """, unsafe_allow_html=True)
else:
    # Apply light theme
    st.markdown("""
    <style>
    .stApp {
        background-color: white !important;
        color: #262730 !important;
    }
    .stRadio label {
        color: #262730 !important;
        font-weight: 500 !important;
    }
    /* Fix other elements in light mode */
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #262730 !important;
    }
    /* Fix info boxes in light mode */
    .stAlert {
        color: #262730 !important;
    }
    /* Fix all text elements */
    p, span, div, h1, h2, h3, h4, h5, h6 {
        color: #262730 !important;
    }
    /* Make sure info box text is visible */
    .stAlert p {
        color: #262730 !important;
    }
    .element-container div.stMarkdown p {
        color: #262730 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Always apply these styles regardless of theme
st.markdown("""
<style>
/* Ensure navigation options are always visible with good contrast */
.stRadio label {
    font-size: 1rem !important;
}
.stButton button {
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("Movie Data Processor")

# Cache the processor so it doesn't reload on every UI interaction
@st.cache_data
def load_processor():
    return MovieDataProcessor()

processor = load_processor()

# Navigation Sidebar
page = st.radio("Go to", ["Main Page", "Chronological Info", "Movie Summarizer", "Become the main character"], horizontal=True)


if page == "Main Page":
    st.header("Top Movie Types")
    
    # 🎨 Add color picker for movie types plot
    movie_color = st.color_picker("Click and Pick your favorite color!", "#C0C0C0")  # skyblue default
    
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
    actor_color = st.color_picker("Click and Pick your favorite color!", "#C0C0C1")  # green default

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
 # Height range selection with validation logic
    col1, col2 = st.columns(2)
    
    with col1:
        # Min height input - limit to max_height
        new_min_height = st.number_input(
            "Enter Minimum Height (m)", 
            min_value=1.0, 
            max_value=st.session_state.max_height,
            value=st.session_state.min_height, 
            step=0.1,
            key="min_height_input"
        )
        
        # Update session state if changed
        if new_min_height != st.session_state.min_height:
            st.session_state.min_height = new_min_height
    
    with col2:
        # Max height input - must be >= min_height
        new_max_height = st.number_input(
            "Enter Maximum Height (m)", 
            min_value=st.session_state.min_height, 
            max_value=2.3,
            value=max(st.session_state.max_height, st.session_state.min_height), 
            step=0.1,
            key="max_height_input"
        )
        
        # Update session state if changed
        if new_max_height != st.session_state.max_height:
            st.session_state.max_height = new_max_height

# 🎨 Add color picker for height distribution plot
    height_color = st.color_picker("Click and Pick your favorite color!", "#C0C0C2")  # dodgerblue default

    # Get DataFrame and plot
    result_df = processor.actor_distributions(gender, st.session_state.max_height, st.session_state.min_height, plot=False)

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
    movie_color = st.color_picker("Click and Pick your favorite color!", "#C0C0C0")  # skyblue default
    
    # Retrieve the data
    releases_df = processor.releases(selected_genre)
    
    if not releases_df.empty:
        # Plot the data
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(releases_df["Year"], releases_df["Movie_Count"], color=movie_color)
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
    movie_color = st.color_picker("Click and Pick your favorite color!", "#C0C0C1")  # skyblue default
    
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
    ax.bar(births_df.iloc[:, 0], births_df["Birth_Count"], color=movie_color, width=0.9)

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

elif page == "Become the main character":
    st.title("Become the Main Character")
    st.write("Enter your name and an OpenAI API key to generate a personalized movie plot where you're the star!")
    
    # Input field for API key (password field for security)
    api_key = st.text_input("Enter your OpenAI API key", type="password", help="Your API key will not be stored")
    
    # Input field for name
    name = st.text_input("Enter your name", help="This name will be used as the main character in a random movie plot")
    
    # Button to generate personalized plot
    if st.button("Generate My Movie"):
        if not name or not api_key:
            st.error("Please enter both your name and an API key.")
        else:
            # Show loading spinner while generating
            with st.spinner("Creating your personalized movie plot..."):
                try:
                    # Call the personalize_movie_plot function
                    personalized_plot = processor.personalize_movie_plot(name, api_key)
                    
                    # Display the result
                    st.subheader("Your Personalized Movie Plot")
                    st.text_area(
                        label="",
                        value=personalized_plot,
                        height=400
                    )
                except ValueError as ve:
                    st.error(f"Error: {str(ve)}")
                except Exception as e:
                    st.error(f"Something went wrong: {str(e)}")
    
    # Add some helpful information and disclaimers
    st.info("""
    **Note**: 
    - This feature requires a valid OpenAI API key
    - Your API key is used only for this request and is not stored
    - You may be charged by OpenAI for the API usage
    """)  