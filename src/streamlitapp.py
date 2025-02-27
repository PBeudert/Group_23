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

# User input for number of movie types
N = st.number_input("What number of movie types would you like to see", min_value=1, max_value=50, value=10, step=1)

# Display movie types
st.write("### Top Movie Types")
df_movie_type= processor.__movie_type__(N)  # Use user-selected value

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_movie_type["Movie_Type"], df_movie_type["Count"], color="skyblue")
ax.set_xlabel("Movie Type")
ax.set_ylabel("Count")
ax.set_title("Histogram of Movie Types")
plt.xticks(rotation=45, ha="right")

# Show the plot in Streamlit
st.pyplot(fig)

df_actor_count=processor.__actor_count__()
fag, ax = plt.subplots(figsize=(8, 5))
ax.bar(df_actor_count["Number_of_Actors"], df_actor_count["Movie_Count"], color="green")
ax.set_xlabel("Number of Actors")
ax.set_ylabel("Number of Movies")
ax.set_title("Histogram of Number_of_Actors")
plt.xticks(rotation=45, ha="right")

st.pyplot(fag)


min_h = st.number_input("What is the plot minimum height", min_value=1.0, max_value=2.1, value=1.2, step=0.1)
max_h = st.number_input("What is the plot maximum height", min_value=1.1, max_value=2.3, value=2.1, step=0.1)

gender = st.selectbox("What gender are you interested in?;)",["All","Male","Female"])
if gender == "Male":
    gender="M"
elif gender == "Female":
    gender = "F"


df_actor_distr=processor.__actor_distributions__(gender,max_h,min_h,False)

hist_values, bin_edges = np.histogram(df_actor_distr["Height"], bins=20,alpha=0.7)

# Plot histogram
fug, ax = plt.subplots(figsize=(10, 6))

ax.bar(bin_edges[:-1], hist_values, width=np.diff(bin_edges), color="darkblue", edgecolor="black", align="edge")

# Labels
ax.set_xlabel("Heights")
ax.set_ylabel("Actor Height Occurrences")
ax.set_title("Histogram of Actor Height Distributions")

# Adjust Y-axis to prevent cutoff
ax.set_ylim(0, hist_values.max() * 1.1)  # Adds 10% space above highest bar

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Show plot
st.pyplot(fug)