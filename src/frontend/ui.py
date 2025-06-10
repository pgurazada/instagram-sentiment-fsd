import requests
import streamlit as st

st.title('Instagram Comments Sentiment Analyzer')
st.markdown("""
An API serving a DSPy program that analyzes comments received on marketing campaign run by Samsung for its phones on Instagram based on its content.            
""")

url = "http://127.0.0.1:8000/predict"

def get_prediction(text):
    response = requests.post(url, json={"text": text})
    return response.json()['data']['sentiment_label'], response.json()['data']['sentiment_score']


text_input = st.text_input('Enter a comment', 'I hate this phone')

if st.button('Submit'):
    sentiment, sentiment_score = get_prediction(text_input)
    st.write(f'The sentiment of the comment is {sentiment} with a score of {sentiment_score}')
