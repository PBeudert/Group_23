import streamlit as st
import pandas as pd
import numpy as np
from MovieDataProcessor import MovieDataProcessor
processor = MovieDataProcessor()
N=st.number_input("What number of movie types would you like to see",value=10)
st.write(processor.__movie_type__(10))
st.title('Test1')

