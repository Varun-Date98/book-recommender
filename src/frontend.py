import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv


load_dotenv()
home = os.environ["BASE_URL"]
st.set_page_config(
    page_title="📚 Book Recommender",
    page_icon="📚",
    layout="wide"
)

st.markdown("""
<style>
.small { color:#94a3b8; font-size:0.9rem; }
.card {
  padding: 1rem; border: 1px solid #1f2937; border-radius: 12px;
  background: #0b1220; color: #e5e7eb;
}
.grid { display:grid; grid-gap: 12px; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
.book { background:#0f172a; border:1px solid #1f2937; border-radius:10px; padding:12px; }
.cover { width:100%; height:340px; object-fit:cover; border-radius:8px; background:#0b1220; }
.title { font-weight:700; margin-top:8px; }
.author { color:#9ca3af; margin-top:4px; }
.score { color:#34d399; margin-top:6px; }
.sugg-item { padding:4px 8px; border-radius:6px; }
.sugg-item:hover { background:#111827; cursor:pointer; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_titles():
    resp = requests.get("/".join([home, "get_titles"]))

    if resp.status_code == 200:
        return resp.json().get("books", [])

    return []


titles_list = load_titles()

st.title("📚 Book Recommender")
st.caption("Type a book you like; get similar reads.")

title_col, author_col, mood_col, button_col = st.columns(4)

with title_col:
    title = st.selectbox("Title", options=titles_list, index=None, placeholder="Start typing a book title")

with author_col:
    author = st.text_input("Author", placeholder="Book Author")

with mood_col:
    mood = st.text_input("Mood", placeholder="Happy, Adventurous etc.")

with button_col:
    recommend = st.button("Recommend", type="primary", use_container_width=True)

if recommend:
    payload = {
        "title": title if title else None,
        "author": author if author else None,
        "mood": mood if mood else None
    }

    if not any(payload[x] for x in payload):
        st.warning("Please enter any of book title or author or mood to get suggestions.")
    else:
        try:
            with st.spinner("Fetching book recommendations"):
                resp = requests.post("/".join([home, "recommend"]), json=payload)

                if resp.status_code == 200:
                    recs_df = pd.DataFrame(resp.json())

                    st.dataframe(
                        recs_df[["covers", "titles", "authors", "reasons"]],
                        use_container_width= True, hide_index=True,
                        row_height=100,
                        column_config={
                            "covers": st.column_config.ImageColumn("Cover", help="From Open Library", width=60),
                            "titles": st.column_config.TextColumn("Book Title", width="small"),
                            "authors": st.column_config.TextColumn("Book Author", width="small"),
                            "reasons": st.column_config.TextColumn("Reason", width="large")
                        }
                    )
        except Exception as e:
            print(f"Error occurred, {e}")
