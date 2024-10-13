# import json
# import pickle
# import random

# import nltk
# import numpy as np
# from nltk.stem import WordNetLemmatizer
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.optimizers import SGD

# # Ensure NLTK data is downloaded
# nltk.download('punkt')
# nltk.download('wordnet')

# #initialize the lemmatizer for finding the lemmas
# lemmatizer = WordNetLemmatizer()

# # Load intents from JSON file
# intents = json.loads(open('intents.json').read())

# words = []
# classes = []
# documents = []
# ignore_letters = ['?', '!', '.', ',']

# # Iterate through intents and collect patterns and tags
# for intent in intents['intents']:
#     for pattern in intent['patterns']:
#         word_list = nltk.word_tokenize(pattern)
#         words.extend(word_list)
#         documents.append((word_list, intent['tag']))
#         if intent['tag'] not in classes:
#             classes.append(intent['tag'])

# # Lemmatize words and remove ignored characters
# words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
# words = sorted(set(words))  # Remove duplicates and sort

# # Sort classes
# classes = sorted(set(classes))

# # Save words and classes using pickle
# pickle.dump(words, open('words.pkl', 'wb'))
# pickle.dump(classes, open('classes.pkl', 'wb'))

# # Prepare training data
# training = []
# output_empty = [0] * len(classes)

# # Create bag of words and prepare output rows
# for doc in documents:
#     bag = []
#     word_patterns = doc[0]
#     word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
#     for word in words:
#         bag.append(1) if word in word_patterns else bag.append(0)

#     output_row = list(output_empty)
#     output_row[classes.index(doc[1])] = 1
#     training.append([bag, output_row])

# # Shuffle training data
# random.shuffle(training)

# # Separate into input features (train_x) and output labels (train_y)
# train_x = np.array([sample[0] for sample in training])
# train_y = np.array([sample[1] for sample in training])

# # Print shapes for debugging
# print(f"train_x shape: {train_x.shape}")
# print(f"train_y shape: {train_y.shape}")

# # Define the model architecture
# model = Sequential()
# model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(classes), activation='softmax'))

# # Configure optimizer with learning rate
# sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# # Compile model with optimizer and loss function
# model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# # Train the model
# history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# # Save the trained model
# model.save('voice_model.h5')



import json
import os
import pickle
import random

import nltk
import numpy as np
import speech_recognition as sr
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load data
intents_path = 'intents.json'
with open(intents_path, 'r') as file:
    intents = json.load(file)  # Load intents from a JSON file

# Load pre-processed words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Load the trained model
model = load_model('voice_model.h5')

# Function to clean up the input sentence
def clean_up_sentence(sentence):
    """ 
    Tokenizes and lemmatizes the input sentence for consistent processing.
    
    Args:
        sentence (str): The input sentence to clean.
    
    Returns:
        list: A list of cleaned and lemmatized words.
    """
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the sentence
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Lemmatize and lowercase
    return sentence_words

# Function to create a bag of words representation
def bag_of_words(sentence):
    """ 
    Creates a bag-of-words representation of the input sentence.
    
    Args:
        sentence (str): The input sentence.
    
    Returns:
        np.array: A binary array representing the presence of words.
    """
    sentence_words = clean_up_sentence(sentence)  # Clean the sentence
    bag = [0] * len(words)  # Initialize the bag with zeros
    for w in sentence_words:  # Check each word in the cleaned sentence
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1  # Set presence of the word to 1
    return np.array(bag)  # Return the bag as a numpy array

# Function to predict the intent class of the input sentence
def predict_class(sentence):
    """ 
    Predicts the class (intent) of the input sentence using the trained model.
    
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
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Create a list of intents and probabilities
    return return_list

# Function to get a response based on predicted intents
def get_response(intents_list, intents_json):
    """ 
    Generates a response based on the predicted intents.
    
    Args:
        intents_list (list): The list of predicted intents.
        intents_json (dict): The JSON structure containing intent responses.
    
    Returns:
        str or None: A response string based on the intent or None if not found.
    """
    if intents_list:  # Check if there are any predicted intents
        tag = intents_list[0]['intent']  # Get the top predicted intent
        list_of_intents = intents_json['intents']  # Get the list of intents from JSON
        for intent in list_of_intents:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])  # Randomly select a response
                return result
    return None  # Return None if no response is found

# Function to update intents and retrain the model with new input
def update_intents(new_intent, response):
    """ 
    Updates the intents JSON file with a new intent and retrains the model.
    
    Args:
        new_intent (str): The new intent phrase.
        response (str): The corresponding response for the new intent.
    """
    with open(intents_path, 'r') as file:
        intents = json.load(file)  # Load current intents
        
    # Check if the new intent already exists
    for intent in intents['intents']:
        if new_intent in intent['patterns']:
            intent['responses'].append(response)  # Add the new response
            break
    else:
        intents['intents'].append({
            'tag': 'new_intent',
            'patterns': [new_intent],
            'responses': [response]
        })  # Add a completely new intent

    # Save the updated intents to the JSON file
    with open(intents_path, 'w') as file:
        json.dump(intents, file, indent=4)

    # Retrain the model with the updated intents
    words = []
    classes = []
    documents = []
    ignore_letters = ['?', '!', '.', ',']  # Define characters to ignore

    # Prepare the training data
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = nltk.word_tokenize(pattern)  # Tokenize each pattern
            words.extend(word_list)  # Add words to the words list
            documents.append((word_list, intent['tag']))  # Append (pattern, intent) pair
            if intent['tag'] not in classes:
                classes.append(intent['tag'])  # Add new classes

    words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]  # Clean words
    words = sorted(set(words))  # Remove duplicates and sort
    classes = sorted(set(classes))  # Sort classes

    # Save the updated words and classes
    pickle.dump(words, open('words.pkl', 'wb'))
    pickle.dump(classes, open('classes.pkl', 'wb'))

    # Prepare training data
    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        word_patterns = doc[0]
        word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]  # Clean up patterns
        for word in words:
            bag.append(1) if word in word_patterns else bag.append(0)  # Create bag-of-words

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1  # Set the output row for the corresponding class
        training.append([bag, output_row])  # Append training sample

    random.shuffle(training)  # Shuffle the training data

    # Split training data into inputs (X) and outputs (y)
    train_x = np.array([sample[0] for sample in training])
    train_y = np.array([sample[1] for sample in training])

    # Create a new model
    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))  # Hidden layer
    model.add(Dropout(0.5))  # Dropout layer for regularization
    model.add(Dense(64, activation='relu'))  # Second hidden layer
    model.add(Dropout(0.5))  # Dropout layer
    model.add(Dense(len(classes), activation='softmax'))  # Output layer with softmax activation

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)  # Stochastic Gradient Descent optimizer
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Compile the model
    model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)  # Train the model

    model.save('voice_model.h5')  # Save the trained model

# Function to recognize speech and handle new input
def recognize_speech():
    """ 
    Recognizes speech from the microphone and returns it as text.
    
    Returns:
        str: The recognized text or an error message if recognition fails.
    """
    recognizer = sr.Recognizer()  # Initialize recognizer

    with sr.Microphone() as source:
        print("Listening...")  # Prompt to indicate listening
        recognizer.adjust_for_ambient_noise(source)  # Adjust for ambient noise
        audio = recognizer.listen(source)  # Listen for audio input

    try:
        text = recognizer.recognize_google(audio)  # Convert audio to text
        return text
    except sr.UnknownValueError:
        return "Sorry, I could not understand what you said."  # Handle unrecognized speech
    except sr.RequestError:
        return "Sorry, my speech service is down."  # Handle service errors

# Main loop to handle interaction and learning
while True:
    msg = recognize_speech()  # Recognize speech input
    if not msg.startswith("Sorry"):  # Check for valid input
        intents_list = predict_class(msg)  # Predict the intent of the message
        response = get_response(intents_list, intents)  # Get the response based on the intent
        if response is None:  # If no response is found
            print("Bot: Sorry, I didn't understand that. Can you provide the appropriate response?")
            new_response = input("You: ")  # Ask user for a response
            update_intents(msg, new_response)  # Update intents with the new response
            intents_list = predict_class(msg)  # Re-predict the intent
            response = get_response(intents_list, intents)  # Get the updated response
        print("Bot:", response)  # Print the bot's response
