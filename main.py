# ---------- READ FILES USING PYTHON LIBRARIES ----------

"""
import os

#Read the file using standard python libraries

with open(os.getcwd()+ "/Spark-Course-Description.txt", 'r') as fh:
    filedata = fh.read()

#print first 200 chars in the file
print("Data read from file : ", filedata[0:200])
"""

"""
# ---------- READ FILES USING NLTK CorpusReader ----------
import os
import nltk
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
nltk.download('punkt')

# Read the file into a corpus. The same command can read an entire directory
corpus = PlaintextCorpusReader(os.getcwd(), "Spark-Course-Description.txt")

print(corpus.raw())

# Extract the file ID from the corpus
print("Files in this corpus : ", corpus.fileids())

# Extract paragraphs from the corpus
paragraphs = corpus.paras()
print("\n Total paragraphs in this corpus : ", len(paragraphs))

# Extract sentences from the corpus
sentences = corpus.sents()
print("\n Total sentences in this corpus : ", len(sentences))
print("\n The first sentence : ", sentences[0])
# Extract words from the corpus
words = corpus.words()
print("\n Total words in this corpus : ", len(words))

# Analyze the corpus ----------

# Find the frequency distribution of words in the corpus
course_freq_dist = nltk.FreqDist(corpus.words())

# Print most commonly used words
print("Top 10 words in the corpus: ", course_freq_dist.most_common(10))

# find the distribution for a specific word
print("\n Distribution for \"Spark\" :", course_freq_dist.get("Spark"))
"""

"""

# ---------- Tokenization ----------

import nltk
import os

# Read the base file into a raw text variable
base_file = open(os.getcwd() + "/Spark-Course-Description.txt", 'rt')
raw_text = base_file.read()
base_file.close()

# Extract tokens
token_list = nltk.word_tokenize(raw_text)
print("Token List : ", token_list[:20])
print("\n Total Tokens : ", len(token_list))

# ---------- Cleansing ----------

# Remove punctuation
# Use the Punkt library to extract tokens
token_list2 = list(filter(lambda token: nltk.tokenize.punkt.PunktToken(token).is_non_punct, token_list))
print("Token List after removing punctuation : ", token_list2[:20])
print("\nTotal tokens after removing punctuation : ", len(token_list2))

# Convert to lower case
token_list3 = [word.lower() for word in token_list2]
print("Token list after converting to lower case : ", token_list3[:20])
print("\nTotal tokens after converting to lower case : ", len(token_list3))

# ---------- Stop-Word Removal ----------

# Download the standard stopword list
nltk.download('stopwords')
from nltk.corpus import stopwords

# Remove stopwords
token_list4 = list(filter(lambda token: token not in stopwords.words('english'), token_list3))
print("Token list after removing stop words : ", token_list4[:20])
print("\nTotal tokens after removing stop words : ", len(token_list4))

# ---------- Stemming ----------

# Use the PorterStemmer library for stemming.
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

# Stem data
token_list5 = [stemmer.stem(word) for word in token_list4]
print("Token list after stemming : ", token_list5[:20])
print("\nTotal tokens after Stemming : ", len(token_list5))

# ---------- Lemmatization ----------
# Use the wordnet library to map words to their lemmatized form
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
token_list6 = [lemmatizer.lemmatize(word) for word in token_list4]
print("Token list after Lemmatization : ", token_list6[:20])
print("\nTotal tokens after Lemmatization : ", len(token_list6))

# Check for token technlogies
print("Raw : ", token_list4[20], " , Stemmed : ", token_list5[20], " , Lemmatized : ", token_list6[20])

"""

# ---------- ADVANCED TEXT PROCESSING ----------

# Prepare data for use in this exercise

import nltk
import os
# Download punkt package, used part of the other commands
nltk.download('punkt')

# Read the base file into a token list
base_file = open(os.getcwd()+ "/Spark-Course-Description.txt", 'rt')
raw_text = base_file.read()
base_file.close()

# Execute the same pre-processing done in module 3
token_list = nltk.word_tokenize(raw_text)

token_list2 = list(filter(lambda token: nltk.tokenize.punkt.PunktToken(token).is_non_punct, token_list))

token_list3=[word.lower() for word in token_list2 ]

nltk.download('stopwords')
from nltk.corpus import stopwords
token_list4 = list(filter(lambda token: token not in stopwords.words('english'), token_list3))

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
token_list6 = [lemmatizer.lemmatize(word) for word in token_list4 ]

print("\n Total Tokens : ", len(token_list6))

from nltk.util import ngrams
from collections import Counter

# Find bigrams and print the most common 5
bigrams = ngrams(token_list6, 2)
print("Most common bigrams : ")
print(Counter(bigrams).most_common(5))

# Find trigrams and print the most common 5
trigrams = ngrams(token_list6, 3)
print(" \n Most common trigrams : ")
print(Counter(trigrams).most_common(5))

# 04_02 Parts-of-Speech Tagging

# Some examples of Parts-of-Speech abbreviations:
# NN : noun
# NNS : noun plural
# VBP : Verb singular present.

# download the tagger package
nltk.download('averaged_perceptron_tagger')
# Tag and print the first 10 tokens
tags = nltk.pos_tag(token_list4)[:10]
print(tags)

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

#Use a small corpus for each visualization
vector_corpus = [
    'NBA is a Basketball league',
    'Basketball is popular in America.',
    'TV in America telecast BasketBall.',
]

#Create a vectorizer for english language
vectorizer = TfidfVectorizer(stop_words='english')

#Create the vector
tfidf=vectorizer.fit_transform(vector_corpus)

print("Tokens used as features are : ")
print(vectorizer.get_feature_names())

print("\n Size of array. Each row represents a document. Each column represents a feature/token")
print(tfidf.shape)

print("\n Actual TF-IDF array")
tfidf.toarray()
