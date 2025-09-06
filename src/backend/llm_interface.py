import os
import json
import logging
from typing import Dict
from typing import List
from openai import OpenAI
from datetime import date
from dotenv import load_dotenv


os.makedirs("logs", exist_ok=True)
log_file = f"logs/{date.today()}_llm.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8")
    ]
)

logger = logging.getLogger("LLM Logger")


class LanguageModel:
    def __init__(self, model: str = "gpt-5-nano", temperature: float = 0.2):
        load_dotenv()
        self.model = model
        self.temp = temperature
        self.client = OpenAI(api_key=os.environ["OPEN_AI_KEY"])

    def refine_recommendations(self, title: str, author: str,
                               mood: str, als_recs: List[str]) -> Dict[str, str]:
        query = f"""
        You are a book recommendation system.

        Rules:
        1. The user may provide a book title, an author, a mood, or any combination of these.
        2. You must return exactly five books as recommendations.
        3. When a book title is given, you will also receive five recommendations from an ALS model. 
           - Keep ALS recommendations that are relevant.
           - Remove irrelevant ones.
           - If fewer than five relevant remain, fill in with your own.
        4. For each of the recommended book, provide a short (less than 300 characters) reason as to why the user might like it.
        5. Output **only valid JSON** in this schema:
        {{
          "books": ["Book Title 1", "Book Title 2", "Book Title 3", "Book Title 4", "Book Title 5"],
          "authors": ["Author 1", "Author 2", "Author 3", "Author 4", "Author 5"],
          "reasons": ["Reason 1", "Reason 2", "Reason 3", "Reason 4", "Reason 5"]
        }}
        
        User input:
        - book title: {title}
        - author: {author}
        - mood: {mood}
        - ALS recommendations: {als_recs}
        """

        response = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": query}],
            response_format={"type": "json_object"}
        )

        result = response.choices[0].message.content
        logger.info(f"Recommendations for input (book title: {title}, author: {author}, mood: {mood},"
                    f"recs: {als_recs}) is: {result}")
        return json.loads(result)
