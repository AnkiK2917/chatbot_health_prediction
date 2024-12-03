import json
import pandas as pd

# Load intents.json
with open('intents.json', 'r') as file:
    intents = json.load(file)

# Extract messages and labels
data = []
for intent in intents['intents']:
    tag = intent['tag']
    for pattern in intent['patterns']:
        data.append({"Message": pattern, "Label": tag})

# Convert to DataFrame
df = pd.DataFrame(data)

# Save as CSV for training or further use
df.to_csv('health_prediction_dataset.csv', index=False)

label_mapping = {
    "greeting": "neutral",
    "mental_health_anxiety": "anxiety",
    "mental_health_depression": "depression",
    "mental_health_stress": "stress",
    "physical_health_pain": "pain",
    "physical_health_fatigue": "fatigue",
    "exercise_advice": "physical stress",
    "diet_nutrition": "nutrition",
    "goodbye": "neutral",
    "thank_you": "neutral"
}

# Apply mapping
df['Label'] = df['Label'].map(label_mapping)

# Save updated dataset
df.to_csv('health_data.csv', index=False)