from flask import Flask, request, jsonify
import json
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from flask_cors import CORS  # Allow cross-origin requests

app = Flask(__name__)
CORS(app)  # Enable CORS for API access from mobile apps

# Load intents from JSON file
file_path = os.path.abspath("./intents.json")
with open(file_path, "r") as file:
    intents = json.load(file)

# Initialize vectorizer and classifier
vectorizer = TfidfVectorizer(ngram_range=(1, 4))
clf = LogisticRegression(random_state=0, max_iter=10000)

# Prepare training data
tags = []
patterns = []
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)

# Train the model
x = vectorizer.fit_transform(patterns)
y = tags
clf.fit(x, y)

# Chatbot function
def chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = clf.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Sorry, I don't understand that."

# API Endpoint for Chatbot
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"response": "Please provide a message"}), 400

    response = chatbot(user_message)
    return jsonify({"response": response})

# API Endpoint for Health Check
@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "TechCertify Chatbot API is running!"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Use port from Render, default to 10000
    app.run(host="0.0.0.0", port=port)
