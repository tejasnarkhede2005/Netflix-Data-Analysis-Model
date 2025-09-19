import streamlit as st
import pickle
import numpy as np

# -------------------------
# Load the trained model
# -------------------------
@st.cache_resource
def load_model():
    with open("Netflix Data.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🎬 Netflix Movie Duration Predictor")
st.write("Predict the duration (in minutes) of a movie/show using the trained model.")

# Input fields
release_year = st.number_input("Release Year", min_value=1900, max_value=2025, value=2020, step=1)

category = st.selectbox("Category", ["Movie", "TV Show"])
rating = st.selectbox("Rating", ["G", "PG", "PG-13", "R", "TV-Y", "TV-Y7", "TV-G", "TV-PG", "TV-14", "TV-MA"])
country = st.text_input("Country", "United States")

# -------------------------
# Preprocess inputs
# -------------------------
# NOTE: The preprocessing here must match how you trained the model.
# If you used OneHotEncoding or LabelEncoding, you should apply the same here.
# For demonstration, let's use simple numeric encoding placeholders.

category_map = {"Movie": 0, "TV Show": 1}
rating_map = {"G": 0, "PG": 1, "PG-13": 2, "R": 3, "TV-Y": 4, "TV-Y7": 5, "TV-G": 6, "TV-PG": 7, "TV-14": 8, "TV-MA": 9}

category_num = category_map.get(category, 0)
rating_num = rating_map.get(rating, 0)

# For country, just take length of name (dummy encoding for now)
country_num = len(country)

# Final input array
input_data = np.array([[release_year, category_num, rating_num, country_num]])

# -------------------------
# Predict button
# -------------------------
if st.button("Predict Duration"):
    prediction = model.predict(input_data)
    st.success(f"📽️ Predicted Duration: {prediction[0]:.2f} minutes")
