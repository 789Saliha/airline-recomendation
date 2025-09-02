import streamlit as st
import pandas as pd
from hybrid_recommender import hybrid_recommend

st.title("Airline Recommendation System")

# Load dataset
df = pd.read_csv("/content/airlines_reviews.csv")

traveller_type = st.selectbox("Select Traveller Type", df["Type of Traveller"].unique())

top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

st.write(hybrid_recommend(traveller_type, top_n=top_n))
