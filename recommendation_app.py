"""
Author: Mackenzie Ross
Date: 6 December 2022
Description: Import data about the top romance books on Goodreads, 
             clean the data, and recommend books based on book synopses
"""

# Import necessary libraries
import pandas as pd
import nltk
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, render_template, request

# Load the dataset
book_data = 'books.csv'
book_df = pd.read_csv(book_data)

# Preprocess text data
en_stopwords = nltk.corpus.stopwords.words('english')

# funtion modified from Sarkar
def normalize_document(doc):
    # lower case and remove special characters/whitespace
    doc = re.sub(r'[a-zA-Z0-9\s]', '', doc, re.I|re.A)
    doc = doc.lower()
    doc = doc.strip()
    # tokenize document
    tokens = nltk.word_tokenize(doc)
    # filter stopword out of document
    filtered_tokens = [t for t in tokens if t not in en_stopwords] # recreate document from filtered tokens
    doc = ' '.join(filtered_tokens)
    return doc

normalize_corpus = np.vectorize(normalize_document)
norm_corpus = normalize_corpus(book_df['synopsis'])

# parentheticals from book titles
def normalize_titles(doc):
    doc = re.sub('[\(\[].*?[\)\]]', '', doc)
    doc.strip()
    return doc

for i in range(len(book_df)):
    title = book_df.loc[i, 'title']
    norm_title = normalize_titles(title)
    book_df.loc[i, 'title'] = norm_title

# feature extraction
tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
tfidf_matrix = tf.fit_transform(norm_corpus)

# main functionality
doc_similarity = cosine_similarity(tfidf_matrix)
doc_similarity_df = pd.DataFrame(doc_similarity)
book_list = book_df.loc[:,'title'].values
author_list = book_df.loc[:,'author'].values
# lower case book titles and author names for easier handling, also strip trailing spaces
book_list = [b.lower().strip() for b in book_list]
author_list = [a.lower().strip() for a in author_list]
# convert lists to numpy arrays
np_book_list = np.array(book_list)
np_author_list = np.array(author_list)
# adapted from week 8 coding practice, returns top 3 similar books for the selected book
def book_recs(book_title, books=np_book_list, authors=np_author_list, doc_sims=doc_similarity_df):
    if isinstance(book_title, str):
        book_index = np.where(books == book_title)[0][0]
        book_similarities = doc_sims.iloc[book_index].values
        similar_book_indices = np.argsort(-book_similarities)[1:4]
        similar_books = books[similar_book_indices]
        similar_authors = authors[similar_book_indices]

        books_with_authors = []
        for i in range(len(similar_books)):
            b_title = similar_books[i]
            b_author = similar_authors[i]
            books_with_authors.append(str(b_title + ' by ' + b_author))
        return books_with_authors
    else:
        return "book title was not a string"

# print(book_recs(book_title='beach read'))

# create and run app
app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/index.html")
def return_home():
    return render_template('index.html')

@app.route("/recommender.html")
def recommendations():
    return render_template('recommender.html')

@app.route('/suggestions.html')
def suggestions():
    return render_template('suggestions.html')

@app.route("/results", methods=['GET', 'POST'])
def recommend_book():
    choice = request.form.get('book-choice')
    cleaned = choice.lower().strip()
    books = book_recs(book_title=cleaned)
    return render_template('results.html', book=books)

if __name__ == "__main__":
    app.run(debug=True)