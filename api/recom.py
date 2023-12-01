import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your preprocessed JSON data
with open("your_data.json", "r") as json_file:
    data = json.load(json_file)

# Define the weights for each aspect (key), excluding "Client Demographics"
weights = {
    "Client Feedback": 0.2,
    "Jurisdiction": 0.3,
    "charges": 0.2,
    "Client Demographics" :0.3
    # Add more aspects and adjust their weights as needed
}

# Initialize a list to store user inputs for each key
user_input = {}
keys = list(data[0].keys())  # Get the keys from the first data item and convert to a list
keys.remove("name")  # Remove "name" key from the list of keys
keys.remove("practice location")  # Remove "practice location" key from the list of keys
keys.remove("practices at")  # Remove "practices at" key from the list of keys

# Prompt the user for input for each key (excluding "name", "practice location", "practices at", and "Client Demographics")
for key in keys:
    while True:
        value = input(f"Enter a value for '{key}': ")
        if value:
            user_input[key] = value
            break
        else:
            print(f"Value for '{key}' cannot be empty.")

# Calculate similarity for each key (excluding "name", "practice location", "practices at", and "Client Demographics") based on user input
similarities = []

for item in data:
    total_similarity = 0
    for key in keys:
        target_value = user_input[key]
        item_value = item.get(key, '')  # Use get() to provide a default value ('') if the key is missing

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
                    aspect_similarity = 0  # Set a default similarity if either text is empty

            total_similarity += weights[key] * aspect_similarity
        except ValueError:
            # Handle non-numeric or invalid values gracefully
            pass

    similarities.append((item, total_similarity))

# Sort the results by total similarity in descending order
similarities.sort(key=lambda x: x[1], reverse=True)

# Display the top 5 most similar items based on the combined similarity
top_n = 5
print(f"Top {top_n} most similar items based on the combined similarity:")
for i, (item, similarity) in enumerate(similarities[:top_n]):
    print(f"Rank {i + 1}:")
    for key, value in item.items():
        print(f"{key}: {value}")
    print(f"Combined Similarity: {similarity:.4f}")