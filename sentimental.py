from flask import Flask, request, jsonify
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download VADER lexicon
nltk.download('vader_lexicon')

# Flask application
app = Flask(__name__)

# VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define route for sentiment analysis
@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    texttt = request.json.get('text')
    sentimentalScore = sia.polarity_scores(texttt)
    
    # Determine the sentiment label based on the compound score
    if sentimentalScore['compound'] >= 0.05:
        sentimentalLabel = 'Positive'
    elif sentimentalScore['compound'] <= -0.05:
        sentimentalLabel = 'Negative'
    else:
        sentimentalLabel = 'Neutral'
    
    # Prepare response
    response = {
        'sentimentalLabel': sentimentalLabel,
        'sentimentalScore': sentimentalScore
    }
    
    return jsonify(response)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
