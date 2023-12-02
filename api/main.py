

import json
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Load the data from the XLSX file
file_path = 'data.xlsx'  # Replace with the path to your XLSX file
df = pd.read_excel(file_path)

# Initialize NLTK stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Initialize a list to store the processed data
processed_data = []

# Define the fields to extract from the data
fields_to_extract = [
    "Name",
    "years of experience",
    "in what field",
    "Client Feedback",
    "Jurisdiction",
    "charges",
    "average days of disposal",
    "Language speaks",
    "practices at",
    "practice location",  # Corrected the missing comma
    "service to community",
    "Client Demographics",
]

# Process each data line and extract information
for index, row in df.iterrows():
    lawyer_data = {}
    line = row['Information']  # Replace 'Your_Column_Name' with the actual column name

    parts = line.split(". ")
    lawyer_data["name"] = parts[0].split(' has ')[0]

    for part in parts[1:]:
        for field in fields_to_extract:
            if field in part:
                value = part.split(field)[1].strip(".")
                # Remove the 'is' keyword from the key
                field = field.replace(" is ", " ")
                # Tokenize the value and remove stopwords
                tokens = nltk.word_tokenize(value)
                tokens = [word for word in tokens if word.lower() not in stop_words]
                value = " ".join(tokens)
                if "charges" in field:
                    value = value.split("USD")[0].strip()
                if "Client Feedback" in field:
                    value = value.split(" ")[0].strip()
                if "practices at" in field:
                    # Check if "based in" is present, and if so, extract the practice location
                    if ", based" in value:
                        practice_location = value.split(", based")[1].strip()
                        lawyer_data["practice location"] = practice_location
                    lawyer_data["practices at"] = value.split(",")[0].strip()
                else:
                    lawyer_data[field] = value

    processed_data.append(lawyer_data)

# Convert processed data to JSON format
json_data = json.dumps(processed_data, indent=2)

# Specify the path to the JSON file where you want to store the data
json_file_path = 'your_data.json'

# Save the data to the JSON file
with open(json_file_path, 'w') as json_file:
    json_file.write(json_data)

print(f"Data has been stored in {json_file_path}")


