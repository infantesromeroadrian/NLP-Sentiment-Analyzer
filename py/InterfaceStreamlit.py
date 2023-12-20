import streamlit as st
from TwitterSentimentAnalyzer import TwitterSentimentAnalyzer

# Page configuration
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")

# Title
st.title("Twitter Sentiment Analyzer")

# Creating the analyzer
analyzer = TwitterSentimentAnalyzer()

# Input field for the tweet
tweet_input = st.text_area("Enter the Tweet to analyze:", height=150)

# Button to perform the analysis
analyze_button = st.button("Analyze Sentiment")

# When clicking the analysis button
if analyze_button and tweet_input:
    with st.spinner('Analyzing...'):
        result = analyzer.analyze_tweet(tweet_input)
        sentiment, confidence = analyzer.interpret_result(result)
        st.success(f"Sentiment: {sentiment}\nConfidence: {confidence:.2f}")

# Additional information or credits
st.sidebar.header("About")
st.sidebar.info("This is a Twitter sentiment analysis project using natural language processing models.")

