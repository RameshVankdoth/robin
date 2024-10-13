import json
import pickle
import random

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.layers import Activation, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD

# Ensure NLTK data is downloaded
nltk.download('punkt')
nltk.download('wordnet')

# initialize the lemmatizer for finding lemma 
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents = json.loads(open('intents.json').read())

words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Iterate through intents and collect patterns and tags
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignored characters
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_letters]
words = sorted(set(words))  # Remove duplicates and sort

# Sort classes
classes = sorted(set(classes))

# Save words and classes using pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

# Create bag of words and prepare output rows
for doc in documents:
    bag = []
    word_patterns = doc[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

# Shuffle training data
random.shuffle(training)

# Separate into input features (train_x) and output labels (train_y)
train_x = np.array([sample[0] for sample in training])
train_y = np.array([sample[1] for sample in training])

# Print shapes for debugging
print(f"train_x shape: {train_x.shape}")
print(f"train_y shape: {train_y.shape}")

# Define the model architecture
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

# Configure optimizer with learning rate
sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)

# Compile model with optimizer and loss function
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save('voice_model.h5')




# import json
# import pickle
# import random

# import nltk
# import numpy as np
# from keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from keras.layers import Dense, Dropout
# from keras.models import Sequential
# from keras.optimizers import Adam
# from nltk.stem import WordNetLemmatizer

# # Ensure NLTK data is downloaded
# nltk.download('punkt')
# nltk.download('wordnet')

# lemmatizer = WordNetLemmatizer()

# # Load intents from JSON file
# intents = json.loads(open('Voice_Model/intents.json').read())

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
# model.add(Dense(256, input_shape=(len(train_x[0]),), activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(len(classes), activation='softmax'))

# # Configure optimizer with learning rate
# adam = Adam(learning_rate=0.001)

# # Compile model with optimizer and loss function
# model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# # Define callbacks
# early_stopping = EarlyStopping(monitor='loss', patience=10, verbose=1)
# reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

# # Train the model
# history = model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1, callbacks=[early_stopping, reduce_lr])

# # Save the trained model
# model.save('voice_model.h5')
