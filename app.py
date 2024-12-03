from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from tensorflow.keras.models import load_model
import numpy as np
import nltk
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import os
#import sklearn
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Use a secret key for session management

CHAT_HISTORY_FILE = 'chat_log.json'
CHAT_LOGS_DIR = "chat_logs"
if not os.path.exists(CHAT_LOGS_DIR):
    os.makedirs(CHAT_LOGS_DIR)
with open('health_model.pkl', 'rb') as f:
    health_model = pickle.load(f)
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Function to save chat history for each user
def save_chat_history(user_id, user_message, bot_response):
    # Get the current timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create a new entry for the chat message
    new_entry = {
        'timestamp': timestamp,
        'user_message': user_message,
        'bot_response': bot_response
    }

    chat_file = os.path.join(CHAT_LOGS_DIR, f"{user_id}_chat.json")
    
    try:
        # Load existing chat data if available
        if os.path.exists(chat_file):
            with open(chat_file, 'r') as file:
                chat_history = json.load(file)
        else:
            chat_history = []

        chat_history.append(new_entry)

        # Save updated chat history to the JSON file
        with open(chat_file, 'w') as file:
            json.dump(chat_history, file, indent=4)
        print(f"Chat saved for user {user_id}.")
    except Exception as e:
        print(f"Error saving chat history: {e}")

# Helper Functions (clean_up_sentence, bow, predict_class, get_response)
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    print("Model Prediction:", res)  # Debugging line
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    print("Predicted Intent:", tag)  # Debugging line
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Routes for User Interaction
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        session['username'] = username  # Store username in session
        return redirect(url_for('login'))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form['username']
        password = request.form['password']
        session['username'] = username  # Store username in session
        return redirect(url_for('profile'))
    return render_template("login.html")

@app.route("/profile")
def profile():
    if 'username' in session:
        return render_template("profile.html", username=session['username'])
    return redirect(url_for('login'))

@app.route("/chat")
def chat():
    if 'username' in session:
        return render_template("chat.html")
    return redirect(url_for('login'))

@app.route("/get_response", methods=["POST"])
def chatbot_response():
    if 'username' not in session:
        return redirect(url_for('login'))

    user_message = request.form['msg']
    ints = predict_class(user_message, model)
    bot_response = get_response(ints, intents)

    # Save the chat history (user message and bot response)
    save_chat_history(session['username'], user_message, bot_response)

    return jsonify({"response": bot_response})

@app.route("/health_prediction", methods=["GET", "POST"])
def health_prediction():
    """Health prediction based on chat data"""
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":
        user_id = session.get('username')
        chat_file = os.path.join(CHAT_LOGS_DIR, f"{user_id}_chat.json")

        try:
            # Check if the chat file exists
            if os.path.exists(chat_file):
                with open(chat_file, 'r') as file:
                    chat_history = json.load(file)
                    # Combine all user messages into a single string
                    messages = " ".join([chat["user_message"] for chat in chat_history if "user_message" in chat])

                # Handle empty messages
                if not messages.strip():
                    return jsonify({"error": "No chat data available for health prediction."})

                try:
                    # Convert messages to a vector and ensure dense format
                    message_vector = vectorizer.transform([messages]).toarray()  # Convert to dense array
                    prediction = health_model.predict(message_vector)

                    # Extract prediction and convert to a standard Python int
                    if isinstance(prediction, np.ndarray):
                        prediction = int(prediction[0])
                    elif isinstance(prediction, (int, float)):
                        prediction = int(prediction)

                    return jsonify({"prediction": prediction})

                except Exception as e:
                    return jsonify({"error": f"Error during health prediction: {str(e)}"})
            else:
                return jsonify({"error": "Chat file not found. Please ensure you have chat data."})

        except json.JSONDecodeError:
            return jsonify({"error": "Error reading or decoding chat history. The file might be corrupted."})
        except Exception as e:
            return jsonify({"error": f"Unexpected error: {str(e)}"})

    return render_template("health_prediction.html")



@app.route("/logout")
def logout():
    session.pop('username', None)
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(debug=True)
