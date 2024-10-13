# import datetime
# import json
# import pickle
# import random

# import nltk
# import numpy as np
# import pyttsx3
# import speech_recognition as sr
# from keras.models import load_model
# from nltk.stem import WordNetLemmatizer

# nltk.download('punkt')
# nltk.download('wordnet')

# # Initialize the lemmatizer and load the model and intents
# lemmatizer = WordNetLemmatizer()
# intents = json.loads(open('intents.json').read())
# words = pickle.load(open('words.pkl', 'rb'))
# classes = pickle.load(open('classes.pkl', 'rb'))
# model = load_model('voice_model.h5')

# # Initialize pyttsx3 engine
# engine = pyttsx3.init()

# # List available voices
# voices = engine.getProperty('voices')
# for voice in voices:
#     print(f"Voice: {voice.name}, ID: {voice.id}, Language: {voice.languages}")

# # Set properties for pyttsx3
# engine.setProperty('voice', voices[1].id)  # Change the index to select a different voice
# engine.setProperty('rate', 175)  # Speed of speech
# engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

# def clean_up_sentence(sentence):
#     sentence_words = nltk.word_tokenize(sentence)
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words

# def respond_with_speech(response_text):
#     # Use pyttsx3 to convert text to speech
#     engine.say(response_text)
#     engine.runAndWait()

# def bag_of_words(sentence):
#     sentence_words = clean_up_sentence(sentence)
#     bag = [0] * len(words)
#     for w in sentence_words:
#         for i, word in enumerate(words):
#             if word == w:
#                 bag[i] = 1
#     return np.array(bag)

# def predict_class(sentence):
#     bow = bag_of_words(sentence)
#     res = model.predict(np.array([bow]))[0]
#     ERROR_THRESHOLD = 0.25
#     results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
#     results.sort(key=lambda x: x[1], reverse=True)
#     return_list = []
#     for r in results:
#         return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
#     return return_list

# def get_response(intents_list, intents_json):
#     if not intents_list:
#         return "Sorry, I don't understand that."

#     tag = intents_list[0]['intent']
#     list_of_intents = intents_json['intents']

#     # Handle date and time intents
#     if tag == 'datetime':
#         current_date = datetime.date.today().strftime('%Y-%m-%d')
#         response = random.choice([
#             "Today's date is {current_date}.",
#             "It's {current_date} today.",
#             "The current date is {current_date}."
#         ]).format(current_date=current_date)
#         return response

#     elif tag == 'time':
#         current_time = datetime.datetime.now().strftime('%H:%M:%S')
#         response = random.choice([
#             "The current time is {current_time}.",
#             "It's {current_time} right now.",
#             "The time is {current_time}."
#         ]).format(current_time=current_time)
#         return response

#     # Handle other intents
#     for intent in list_of_intents:
#         if intent['tag'] == tag:
#             return random.choice(intent['responses'])

#     return "Sorry, I don't understand that."

# def recognize_speech():
#     recognizer = sr.Recognizer()
#     with sr.Microphone() as source:
#         print("Listening...")
#         recognizer.adjust_for_ambient_noise(source)
#         try:
#             audio = recognizer.listen(source)
#             text = recognizer.recognize_google(audio)
#         except:
#             return None
#         return text

# while True:
#     msg = recognize_speech()
#     if msg is not None:
#         intents_list = predict_class(msg)
#         response = get_response(intents_list, intents)
#         respond_with_speech(response)
#     else:
#         respond_with_speech("Couldn't understand your speech.")

import datetime
import json
import pickle
import random

import nltk
import numpy as np
import pyttsx3
import speech_recognition as sr
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
from transformers import pipeline

# Initialize NLTK and load your models
nltk.download("punkt")
nltk.download("wordnet")
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model("voice_model.h5")

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("voice", engine.getProperty("voices")[1].id)
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)

# Initialize Hugging Face model for text generation
llm = pipeline("text-generation", model="gpt2")  # Change the model name as needed

def clean_up_sentence(sentence):
    """ 
    Cleans up the input sentence by tokenizing it and lemmatizing 
    each word to its base form. This helps in standardizing the 
    input for further processing.
    
    Args:
        sentence (str): The input sentence to be cleaned.
    
    Returns:
        list: A list of lemmatized words in lowercase.
    """
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the sentence
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lemmatize and lowercase

def respond_with_speech(response_text):
    """ 
    Uses the text-to-speech engine to vocalize the response text.
    
    Args:
        response_text (str): The text to be converted to speech.
    """
    engine.say(response_text)  # Speak the response text
    engine.runAndWait()  # Wait until the speech is finished

def bag_of_words(sentence):
    """ 
    Converts the input sentence into a bag-of-words representation,
    which is useful for training machine learning models.
    
    Args:
        sentence (str): The input sentence.
    
    Returns:
        np.array: A binary array indicating the presence of words.
    """
    sentence_words = clean_up_sentence(sentence)  # Clean the sentence
    bag = [0] * len(words)  # Initialize the bag with zeros
    for w in sentence_words:  # Iterate over the cleaned words
        for i, word in enumerate(words):  # Check against known words
            if word == w:
                bag[i] = 1  # Set the presence of the word to 1
    return np.array(bag)  # Return the bag as a numpy array

def predict_class(sentence):
    """ 
    Predicts the intent of the input sentence using the trained model.
    
    Args:
        sentence (str): The input sentence.
    
    Returns:
        list: A list of predicted intents with their probabilities.
    """
    bow = bag_of_words(sentence)  # Convert the sentence to bag-of-words
    res = model.predict(np.array([bow]))[0]  # Get model predictions
    ERROR_THRESHOLD = 0.25  # Define a threshold for predictions
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filter results
    results.sort(key=lambda x: x[1], reverse=True)  # Sort by probability
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]  # Return intents

def get_response(intents_list, intents_json):
    """ 
    Generates a response based on the predicted intents.
    
    Args:
        intents_list (list): The list of predicted intents.
        intents_json (dict): The JSON structure containing intent responses.
    
    Returns:
        str: A response string based on the intent.
    """
    if not intents_list:  # Check if there are any predicted intents
        return "Sorry, I don't understand that."  # Default response

    tag = intents_list[0]["intent"]  # Get the top predicted intent

    # Provide specific responses for datetime and time intents
    if tag == "datetime":
        return f"Today's date is {datetime.date.today().strftime('%Y-%m-%d')}."
    elif tag == "time":
        return f"The current time is {datetime.datetime.now().strftime('%H:%M:%S')}."

    # Look for the response in the intents JSON
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])  # Randomly select a response

    return "Sorry, I don't understand that."  # Fallback response

def recognize_speech():
    """ 
    Recognizes speech from the microphone and returns it as text.
    
    Returns:
        str or None: The recognized text or None if recognition fails.
    """
    recognizer = sr.Recognizer()  # Initialize recognizer
    with sr.Microphone() as source:  # Use the microphone as the audio source
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        try:
            audio = recognizer.listen(source)  # Listen for audio input
            return recognizer.recognize_google(audio)  # Convert audio to text
        except:
            return None  # Return None if recognition fails

while True:
    msg = recognize_speech()  # Recognize speech input
    if msg is not None:  # Check if a valid message was received
        intents_list = predict_class(msg)  # Predict the intent of the message
        response = get_response(intents_list, intents)  # Get the response based on the intent

        # Use Hugging Face model to generate enhanced response
        enhanced_response = llm(response, max_length=50, num_return_sequences=1)[0]['generated_text']
        respond_with_speech(enhanced_response)  # Vocalize the enhanced response
    else:
        respond_with_speech("Couldn't understand your speech.")  # Fallback for unrecognized speech

