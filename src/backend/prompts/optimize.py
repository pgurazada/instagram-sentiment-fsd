import os
import dspy
import mlflow

import pandas as pd

from typing import Literal
from dotenv import load_dotenv

load_dotenv()

lm = dspy.LM(
    model='openai/gpt-4o-mini',
    temperature=0,
    api_key=os.environ['OPENAI_API_KEY']
)

prompt_gen_lm = dspy.LM(
    model='openai/gpt-4o',
    temperature=0.4,
    api_key=os.environ['OPENAI_API_KEY']
)

dspy.configure(lm=lm, async_max_workers=4)
mlflow.dspy.autolog(log_traces=True, log_evals=True, log_traces_from_eval=True, silent=True)
mlflow.set_experiment('instagram-sentiment-analyzer')

# Access annotated data
splits = {'train': 'train.csv', 'test': 'test.csv'}
train_df = pd.read_csv("hf://datasets/pgurazada1/instagram-comments-sentiment/" + splits["train"])
test_df = pd.read_csv("hf://datasets/pgurazada1/instagram-comments-sentiment/" + splits["test"])

def validate_sentiment(example, prediction, trace=None):
    return prediction.sentiment_label == example.sentiment_label

class SentimentAssignmentInstructions(dspy.Signature):
    """
    Assign sentiment to a comment received on a marketing campaign run by Samsung for its phones on Instagram based on its content.
    Use a scale of -1 to 1, where -1 is negative, 0 is neutral, and 1 is positive.
    """

    comment: str = dspy.InputField()
    sentiment_label: Literal['positive', 'negative', 'neutral'] = dspy.OutputField()
    sentiment_score: float = dspy.OutputField(desc='Assign sentiment on a scale of -1 (negative) to 1 (positive)')

trainset = [
    dspy.Example(comment=row.text, sentiment_label=row.sentiment_label_rater1).with_inputs("comment")
    for row in train_df.itertuples()
]

testset = [
    dspy.Example(comment=row.text, sentiment_label=row.sentiment_label_rater1).with_inputs("comment")
    for row in test_df.itertuples()
]

sentiment_analyzer = dspy.ChainOfThought(SentimentAssignmentInstructions)

# Baseline on the gold examples (before prompt is optimized)

evaluator = dspy.Evaluate(devset=testset, num_threads=4, display_progress=True)

baseline_accuracy = evaluator(sentiment_analyzer, metric=validate_sentiment)

print(f"Baseline Accuracy: {baseline_accuracy}%")

# Optimize the baseline prompt

optimizer = dspy.MIPROv2(
    metric=validate_sentiment,
    auto='light',
    prompt_model=prompt_gen_lm,
    task_model=lm,
    max_labeled_demos=8,
    max_bootstrapped_demos=8,
    num_threads=4
)

optimized_sentiment_analyzer = optimizer.compile(
    sentiment_analyzer,
    trainset=trainset,
    requires_permission_to_run=False
)

new_accuracy = evaluator(optimized_sentiment_analyzer, metric=validate_sentiment)

print(f"New Accuracy after optimization: {new_accuracy}%")

# save the optimized sentiment analyzer

optimized_sentiment_analyzer.save('output/optimized-sentiment-analyzer', save_program=True)