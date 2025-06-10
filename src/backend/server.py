import os
import dspy
import uvicorn
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Instagram Comments Sentiment Analyzer API",
    description="An API serving a DSPy program that analyzes comments received on marketing campaign run by Samsung for its phones on Instagram based on its content.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SQLite setup ---
def init_db():
    conn = sqlite3.connect("interactions.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            comment TEXT,
            sentiment TEXT,
            score REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_interaction(comment: str, sentiment: str, score: float):
    conn = sqlite3.connect("interactions.db")
    c = conn.cursor()
    c.execute(
        "INSERT INTO interactions (comment, sentiment, score) VALUES (?, ?, ?)",
        (comment, sentiment, score)
    )
    conn.commit()
    conn.close()

init_db()
# --- End SQLite setup ---

lm = dspy.LM(
    model='openai/gpt-4o-mini',
    temperature=0,
    api_key=os.environ['OPENAI_API_KEY']
)

dspy.settings.configure(lm=lm, async_max_workers=8)

optimized_sentiment_analyzer_path = 'output/optimized-sentiment-analyzer'

optimized_sentiment_analyzer = dspy.load(optimized_sentiment_analyzer_path)
optimized_sentiment_analyzer_async = dspy.asyncify(optimized_sentiment_analyzer)

class Comment(BaseModel):
    text: str

@app.post("/predict")
async def predict(input: Comment):
    try:
        result = await optimized_sentiment_analyzer_async(comment=input.text)
        result_dict = result.toDict()
        # Extract sentiment and score from result_dict
        sentiment = result_dict.get("sentiment_label", "")
        score = result_dict.get("sentiment_score", 0)
        # Log the interaction
        log_interaction(input.text, sentiment, score)
        return {
            "status": "success",
            "data": result_dict
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=4)