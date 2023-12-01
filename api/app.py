from flask import Flask, render_template, request
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load your preprocessed JSON data
with open("your_data.json", "r") as json_file:
    data = json.load(json_file)

# Define the weights for each aspect (key)
weights = {
    "Client Feedback": 0.2,
    "Jurisdiction": 0.3,
    "charges": 0.2,
    "Client Demographics": 0.5
    # Add more aspects and adjust their weights as needed
}

# Define a route to handle the form input
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get user input from the form
        user_input = {
            "Client Feedback": request.form.get("feedback"),
            "Jurisdiction": request.form.get("jurisdiction"),
            "charges": request.form.get("charges"),
            "Client Demographics": request.form.get("demographics")
        }

        # Calculate similarity for each key based on user input
        similarities = []

        for item in data:
            total_similarity = 0
            for key in user_input:
                target_value = user_input[key]
                item_value = item.get(key, '')

                try:
                    if key == "Client Feedback":
                        target_value = float(target_value)
                        item_value = float(item_value)
                        aspect_similarity = 1 - np.abs(item_value - target_value) / 5
                    else:
                        if target_value and item_value:
                            # Calculate text similarity using cosine similarity
                            tfidf_vectorizer = TfidfVectorizer()
                            tfidf_matrix = tfidf_vectorizer.fit_transform([target_value, item_value])
                            aspect_similarity = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0, 0]
                        else:
                            aspect_similarity = 0

                    total_similarity += weights[key] * aspect_similarity
                except (ValueError, KeyError):
                    pass

            similarities.append((item, total_similarity))

        # Sort the results by total similarity in descending order
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Extract the top 5 most similar items
        top_n = 5
        top_results = similarities[:top_n]

        return render_template("result.html", results=top_results)

    return render_template("index.html")

@app.route("/index", methods=["GET", "POST"])
def home():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
