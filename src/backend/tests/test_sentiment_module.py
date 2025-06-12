import os
import dspy
import pytest
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(
    model='openai/gpt-4o-mini',
    temperature=0,
    api_key=os.environ['OPENAI_API_KEY']
)

dspy.settings.configure(lm=lm, async_max_workers=8)

optimized_sentiment_analyzer_path = 'output/optimized-sentiment-analyzer'
optimized_sentiment_analyzer = dspy.load(optimized_sentiment_analyzer_path)

@pytest.mark.parametrize("text,expected_label", [
    ("I love this phone", "positive"),
    ("This is a crappy phone", "negative"),
    ("", ""),  # Adjust expected_label as per your model's behavior for empty input
])
def test_sentiment_analysis(text, expected_label):
    result = optimized_sentiment_analyzer(comment=text)
    result_dict = result.toDict()
    assert "sentiment_label" in result_dict
    assert "sentiment_score" in result_dict
    # Optionally check label if your model is deterministic
    if expected_label:
        assert result_dict["sentiment_label"].lower() == expected_label