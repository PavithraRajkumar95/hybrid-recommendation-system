import streamlit as st
import requests

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 Movie Recommendation System")

user_id = st.number_input("Enter User ID", min_value=1, step=1)

BACKEND_URL = "http://backend:8000"

if st.button("Get Recommendations"):

    try:
        response = requests.get(f"{BACKEND_URL}/recommend/{user_id}")

        if response.status_code == 200:
            data = response.json()

            st.subheader("Recommendations")

            recs = data.get("recommendations", [])

            if not recs:
                st.warning("No recommendations found 😢")
            else:
                for movie in recs:
                    st.write("👉", movie)

        else:
            st.error("Backend error")

    except Exception as e:
        st.error(f"Connection error: {e}")