import os
import math
from collections import defaultdict
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

# Make sure to download necessary NLTK resources
import nltk

nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


class VSM:
    def __init__(self, corpus_path):
        self.corpus_path = corpus_path
        self.dictionary = defaultdict(list)
        self.doc_lengths = {}  # For document normalization
        self.doc_id_to_name = {}  # Maps document ID to file name
        self.N = 0  # Number of documents
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))

    def preprocess(self, text):
        # Convert to lowercase
        text = text.lower()

        # Tokenization
        tokens = word_tokenize(text)

        # Remove punctuation and non-alphabetical tokens
        tokens = [word for word in tokens if word.isalpha()]

        # Stop word removal
        tokens = [word for word in tokens if word not in self.stop_words]

        # Lemmatization
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        return tokens

    def build_index(self):
        doc_id = 0
        for filename in os.listdir(self.corpus_path):
            doc_id += 1
            self.doc_id_to_name[doc_id] = filename  # Map doc ID to file name
            with open(
                os.path.join(self.corpus_path, filename), "r", encoding="utf-8"
            ) as file:
                try:
                    text = file.read()
                except UnicodeDecodeError as e:
                    print(f"Error decoding {filename}: {e}")
                    continue  # Skip the file if it's unreadable

                terms = self.preprocess(text)
                term_freq = defaultdict(int)
                for term in terms:
                    term_freq[term] += 1

                # Build the postings list
                for term, tf in term_freq.items():
                    self.dictionary[term].append((doc_id, tf))

                # Calculate document length for normalization (lnc)
                doc_length = math.sqrt(
                    sum((1 + math.log10(tf)) ** 2 for tf in term_freq.values())
                )
                self.doc_lengths[doc_id] = doc_length

        self.N = doc_id  # Total number of documents

    def calculate_document_weight(self, tf):
        # Logarithmic term frequency
        return 1 + math.log10(tf)

    def calculate_query_weight(self, tf, df):
        # Logarithmic term frequency and idf
        tf_weight = 1 + math.log10(tf)
        idf = math.log10(self.N / df)
        return tf_weight * idf

    def search(self, query):
        query_terms = self.preprocess(query)
        query_vector = defaultdict(float)
        query_term_weights = {}

        # Build query vector (ltc)
        for term in query_terms:
            df = len(self.dictionary.get(term, []))
            if df == 0:
                continue
            tf = query_terms.count(term)
            query_weight = self.calculate_query_weight(tf, df)
            query_vector[term] = query_weight
            query_term_weights[term] = query_weight

        # Normalize query vector
        query_length = math.sqrt(sum(weight**2 for weight in query_vector.values()))
        for term in query_vector:
            query_vector[term] /= query_length

        # Document ranking
        scores = defaultdict(float)
        for term, q_weight in query_vector.items():
            postings = self.dictionary.get(term, [])
            df = len(postings)
            for doc_id, tf in postings:
                doc_weight = self.calculate_document_weight(tf)
                scores[doc_id] += q_weight * (doc_weight / self.doc_lengths[doc_id])

        # Sort and return top 10 documents
        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:10]

        # Display the document names and their scores
        results = [
            (self.doc_id_to_name[doc_id], score) for doc_id, score in ranked_docs
        ]
        return results

    def run(self):
        print("Building index...")
        self.build_index()
        print("Index built.")
        while True:
            query = input("Enter your search query (or type 'exit' to quit): ")
            if query.lower() == "exit":
                break
            results = self.search(query)
            if not results:
                print("No relevant documents found.")
            else:
                print("Top documents (Name, Score):")
                for name, score in results:
                    print(f"{name}: {score:.4f}")


# Running the model
if __name__ == "__main__":
    corpus_path = "Corpus"  # Update this path to your corpus
    vsm = VSM(corpus_path)
    vsm.run()
