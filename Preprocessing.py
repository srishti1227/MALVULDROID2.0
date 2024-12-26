# importing the necesssary libraries
import re

import pandas as pd
import torch
import re
import nltk
from nltk.stem import WordNetLemmatizer

#Download NLTK resources for lemmatization
nltk.download('punkt_tab')
nltk.download('wordnet')

#initiliase the lemmatizer
lemmatizer = WordNetLemmatizer()

# make a function that processes text
def preprocess_text(text):
    #convert to lowercase
    text = text.lower()
    print("Converted to lower case")
    #Remove sppecial characters and digits
    text = re.sub(r'[^a-z\s]', '', text)
    print("Special characters removed")
    #tokenize the text
    tokens = nltk.word_tokenize(text)
    print("tokenized the Text")
    #lemmatize the tokens
    lemmatized_text = [lemmatizer.lemmatize(token) for token in tokens]
    print("Lemmatized the Text")
    return lemmatized_text


#load your datasets
malware_df = pd.read_csv('malware_family_dataset.csv')
vulnerability_df = pd.read_csv('vulnerability_dataset.csv')

#preprocess the description in both the cases
malware_df['cleaned_description'] = malware_df['description'].apply(preprocess_text)
vulnerability_df['cleaned_description'] = vulnerability_df['description'].apply(preprocess_text)
print("This is the preprocessed list of tokens of Malware  description")
print(malware_df['cleaned_description'])

print("This is the preprocessed list of tokens of Vulnerability description")
print(vulnerability_df['cleaned_description'])

