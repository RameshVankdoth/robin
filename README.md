# Overview
This project is a conversational AI chatbot that can understand and respond to user input using a custom dataset. 
The chatbot has been trained using Keras layers and a neural network model, allowing it to engage in natural conversations similar to human interaction.

## Features
Custom Dataset: Utilizes an intents.json file to define conversation patterns and responses.
Natural Language Processing: Uses NLTK for text processing, including tokenization and lemmatization.
Speech Recognition: Integrates speech recognition capabilities to allow users to interact via voice.
Text-to-Speech: Converts responses into speech for a more interactive experience.
Dynamic Learning: Updates its knowledge base by allowing users to add new intents and responses on-the-fly.

## Technologies Used
```Python
Keras (TensorFlow backend)
NLTK (Natural Language Toolkit)
SpeechRecognition
Pyttsx3 (Text-to-Speech)
NumPy
Pickle
JSON
```
## Installation
Clone the repository:
```bash
git clone https://github.com/RameshVankdoth/robin.git
```
### Install the required packages:
```bash
nltk==3.6.3
numpy==1.21.2
tensorflow==2.8.0
speechrecognition==3.8.1
pyttsx3==2.7
keras==2.8.0
```
### Install the requirements.txt
```bash
pip install -r requirements.txt
```
### Ensure you have the necessary NLTK resources:

```python

import nltk
nltk.download('punkt')
nltk.download('wordnet')
```
#### Prepare your intents.json, words.pkl, and classes.pkl files in the project directory.

## Usage
### Run the chatbot script:

```bash
python chatbot.py
```
### Speak or type your message. The chatbot will respond accordingly. If it doesn't understand your input, it will prompt you to provide an appropriate response, which it will then learn.

### Contributing
If you want to contribute to this project, feel free to open issues or submit pull requests. Any enhancements or new features are welcome!

### Acknowledgements
Keras for the deep learning framework.
NLTK for natural language processing.
SpeechRecognition for recognizing speech.
Pyttsx3 for text-to-speech conversion.
