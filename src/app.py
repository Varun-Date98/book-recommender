import os
import math
import logging
import numpy as np
import pandas as pd
from datetime import date
from typing import Optional
from pydantic import BaseModel
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.LanguageModel import LanguageModel


# Setting up app logging
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

@app.post("/recommend")
def make_recommendation(request: RecommendRequest = None):
    als_recs = None
    mood = request.mood
    title = request.title
    author = request.author

    if title:
        als_recs = recommend_books(title)

    return llm.refine_recommendations(title, author, mood, als_recs)
