import os
import dspy
import uvicorn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Instagram Comments Sentiment Analyzer API",
    description="An API serving a DSPy program that analyzes comments received on marketing campaign run by Samsung for its phones on Instagram based on its content.",
    version="1.0.0"
)

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
        return {
            "status": "success",
            "data": result.toDict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, workers=4)