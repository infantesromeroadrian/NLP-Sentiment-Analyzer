import streamlit as st
from TwitterSentimentAnalyzer import TwitterSentimentAnalyzer
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Twitter Sentiment Analyzer", layout="wide")

# Title and description
st.title("🔍 Twitter Sentiment Analyzer")
st.markdown("### Analyze the sentiment of tweets interactively and visually.")

# Creating the sentiment analyzer
analyzer = TwitterSentimentAnalyzer()

# Input field for the tweet
tweet_input = st.text_area("Enter the Tweet to analyze:", height=150)

# Button to perform the analysis
analyze_button = st.button("Analyze Sentiment")


# Function to create a simple bar chart
def plot_sentiment_bar(sentiment, confidence):
    fig, ax = plt.subplots()
    colors = {'Positive': 'green', 'Negative': 'red', 'Neutral': 'gray'}
    ax.bar(sentiment, confidence, color=colors.get(sentiment, 'blue'))
    plt.xlabel('Sentiment')
    plt.ylabel('Confidence')
    plt.ylim(0, 1)
    return fig


# When clicking the analysis button
if analyze_button and tweet_input:
    with st.spinner('Analyzing...'):
        result = analyzer.analyze_tweet(tweet_input)
        sentiment, confidence = analyzer.interpret_result(result)
        emoji = "😊" if sentiment == "Positive" else "😟" if sentiment == "Negative" else "😐"
        st.success(f"Sentiment: {sentiment} {emoji}")

        # Display the result chart
        st.pyplot(plot_sentiment_bar(sentiment, confidence))

# Additional information or credits in the sidebar
st.sidebar.header("About")
st.sidebar.info("This is a Twitter sentiment analysis project using natural language processing models.")