import os
import math
import logging
import numpy as np
import pandas as pd
from datetime import date
from dotenv import load_dotenv

from typing import Optional, List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager

from src.LanguageModel import LanguageModel


# Setting up app logging
load_dotenv()
os.makedirs("logs", exist_ok=True)
log_file = f"logs/{date.today()}app.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Book Recommendation Logger")


# Required boilerplate code
models = {}
llm = LanguageModel()


class RecommendRequest(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    mood: Optional[str] = None


def recommend_books(title: str, k: int = 5):
    books_df = models["books"]
    item_factors_df = models["item_factors"]

    # Get book_id for the title
    book_id = books_df.loc[books_df["Title"].str.lower() == title.lower()]

    if book_id.empty:
        logger.info(f"Book with title {title} not found in metadata")
        return []

    book_id = book_id["book_id"].values[0]

    # Get the associated book vector
    target_book = item_factors_df.loc[item_factors_df["id"] == book_id]
    book_vector = target_book["features"].values[0]
    book_magnitude = target_book["magnitude"].values[0]

    def calculate_cosine_similarity(row):
        item_vector = row["features"]
        item_magnitude = row["magnitude"]
        return 1.0 * np.dot(item_vector, book_vector) / (item_magnitude * book_magnitude)

    # Get similarities to each of the books
    sims = item_factors_df.copy()
    sims["similarity"] = item_factors_df.apply(calculate_cosine_similarity, axis=1)

    # Filtering out title book itself
    sims = sims.loc[sims["id"] != book_id]

    # Get tok k similar books
    sims = sims.sort_values(by="similarity", ascending=False).iloc[:k]
    recs = pd.merge(books_df, sims, how="inner", left_on="book_id", right_on="id")

    return recs["Title"].values

def get_cover_images(titles: List[str], authors: List[str]):
    ids = []
    covers = []
    BASE_URL = os.environ["OPEN_LIB_BASE"]
    cover_img_url = os.environ["OPEN_LIB_COVER_IMG"]
    cover_id_url = BASE_URL + "title={}&author={}&fields=cover_i&limit=1"

    retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)

    try:
        with requests.Session() as s:
            s.mount("https://", adapter)
            s.headers.update({"User-Agent": "book-recommender/1.0"})

            for title, author in zip(titles, authors):
                title_ = "+".join(title.split(" "))
                author_ = "+".join(author.split(" "))
                query = cover_id_url.format(title_, author_)

                logger.info(f"Retrieving cover from url: {query}")

                resp = s.get(query)

                if resp.status_code == 200:
                    d = resp.json().get("docs", [None])[0]

                    if d:
                        ids.append(d.get("cover_i", None))
                    else:
                        ids.append(None)
                else:
                    ids.append(None)
                    logger.error(f"Failed to get response from open library.")
    except Exception as e:
        logger.error(f"Error occurred while trying to fetch book covers, {e}")
    finally:
        s.close()
        logger.info("Session closed successfully")

    for i in ids:
        covers.append(cover_img_url.format(i))

    logger.info(f"URLs for the book covers are: {covers}")
    return covers

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading models
    models["books"] = pd.read_parquet("models/book_metadata")
    models["item_factors"] = pd.read_parquet("models/item_factors")

    # Calculating magnitudes for each item factor
    models["item_factors"]["magnitude"] = models["item_factors"]["features"].apply(
        lambda x: math.sqrt(sum(y * y for y in x))
    )

    yield

app = FastAPI(title="Book Recommendation", version="0.0.0", lifespan=lifespan)

@app.get("/heart_beat")
def heart_beat():
    return {"status": "Ok"}

@app.get("/get_titles")
def get_book_titles():
    books = models["books"]["Title"].astype(str).tolist()
    return {"books": books}

@app.post("/recommend")
def make_recommendation(request: RecommendRequest = None):
    als_recs = None
    mood = request.mood
    title = request.title
    author = request.author

    if title:
        als_recs = recommend_books(title)

    try:
        recs = llm.refine_recommendations(title, author, mood, als_recs)
        titles = recs.get("books", [])
        authors = recs.get("authors", [])
        reasons = recs.get("reasons", [])
        covers = get_cover_images(titles, authors)
        return dict(
            titles=titles,
            authors=authors,
            reasons=reasons,
            covers=covers
        )
    except Exception as e:
        logger.error(f"Error occurred while fetching recommendations from LLM: {e}")
