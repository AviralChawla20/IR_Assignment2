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
    """
    A class representing a Vector Space Model (VSM) for Information Retrieval (IR).
    This implementation includes the use of lnc.ltc weighting scheme, lemmatization, and stop word removal.
    """

    def __init__(self, corpus_path):
        """
        Initialize the VSM class.
        :param corpus_path: Path to the directory containing text documents for the corpus.
        """
        self.corpus_path = corpus_path  # Path to the corpus folder
        self.dictionary = defaultdict(
            list
        )  # Inverted index: term -> list of (doc_id, tf)
        self.doc_lengths = {}  # Stores document lengths for normalization (lnc)
        self.doc_id_to_name = {}  # Maps document IDs to file names
        self.N = 0  # Total number of documents in the corpus
        self.lemmatizer = WordNetLemmatizer()  # Used for lemmatizing words
        self.stop_words = set(
            stopwords.words("english")
        )  # Set of stop words to be removed

    def preprocess(self, text):
        """
        Preprocess a given text by tokenizing, removing stop words, lemmatizing, and eliminating punctuation.
        :param text: The raw document text.
        :return: A list of preprocessed terms.
        """
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
        """
        Build the inverted index (dictionary) and compute document lengths for normalization.
        """
        doc_id = 0  # Document ID counter
        # Iterate over all files in the corpus directory
        for filename in os.listdir(self.corpus_path):
            doc_id += 1
            self.doc_id_to_name[doc_id] = filename  # Map document ID to file name
            # Read the document content
            with open(
                os.path.join(self.corpus_path, filename), "r", encoding="utf-8"
            ) as file:
                try:
                    text = file.read()  # Read the document text
                except UnicodeDecodeError as e:
                    print(f"Error decoding {filename}: {e}")
                    continue  # Skip files that are unreadable

                # Preprocess the text
                terms = self.preprocess(text)
                term_freq = defaultdict(
                    int
                )  # Dictionary to store term frequencies for the document

                # Calculate term frequency
                for term in terms:
                    term_freq[term] += 1

                # Build the postings list (term -> list of (doc_id, tf))
                for term, tf in term_freq.items():
                    self.dictionary[term].append((doc_id, tf))

                # Calculate document length for normalization using lnc scheme
                doc_length = math.sqrt(
                    sum((1 + math.log10(tf)) ** 2 for tf in term_freq.values())
                )
                self.doc_lengths[doc_id] = doc_length  # Store the document length

        self.N = doc_id  # Store the total number of documents

    def calculate_document_weight(self, tf):
        """
        Calculate the logarithmic term frequency (lnc) for a document.
        :param tf: Term frequency in the document.
        :return: Logarithmic term frequency (lnc).
        """
        return 1 + math.log10(tf)  # Logarithmic scaling of term frequency

    def calculate_query_weight(self, tf, df):
        """
        Calculate the query term weight using logarithmic tf and inverse document frequency (ltc scheme).
        :param tf: Term frequency in the query.
        :param df: Document frequency (number of documents containing the term).
        :return: The tf-idf weight for the query term.
        """
        tf_weight = 1 + math.log10(tf)  # Logarithmic term frequency
        idf = math.log10(self.N / df)  # Inverse document frequency
        return tf_weight * idf  # Calculate tf-idf weight

    def search(self, query):
        """
        Search the corpus for documents most relevant to the query using cosine similarity.
        :param query: The search query.
        :return: A list of top 10 relevant documents and their similarity scores.
        """
        query_terms = self.preprocess(query)  # Preprocess the query
        query_vector = defaultdict(float)  # Query vector for storing weights
        query_term_weights = {}

        # Build query vector using ltc scheme
        for term in query_terms:
            df = len(self.dictionary.get(term, []))  # Document frequency of the term
            if df == 0:
                continue  # Skip terms not found in any document
            tf = query_terms.count(term)  # Term frequency in the query
            query_weight = self.calculate_query_weight(
                tf, df
            )  # Calculate tf-idf weight for the query
            query_vector[term] = query_weight  # Store the query term weight
            query_term_weights[term] = query_weight

        # Normalize the query vector (L2 normalization)
        query_length = math.sqrt(sum(weight**2 for weight in query_vector.values()))
        for term in query_vector:
            query_vector[
                term
            ] /= query_length  # Normalize each term's weight in the query

        # Document ranking using cosine similarity
        scores = defaultdict(float)  # Dictionary to store similarity scores
        for term, q_weight in query_vector.items():
            postings = self.dictionary.get(term, [])  # Get postings list for the term
            df = len(postings)  # Document frequency
            for doc_id, tf in postings:
                doc_weight = self.calculate_document_weight(
                    tf
                )  # Calculate document weight (lnc)
                scores[doc_id] += q_weight * (
                    doc_weight / self.doc_lengths[doc_id]
                )  # Cosine similarity calculation

        # Sort documents by score and return the top 10
        ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:10]

        # Display the document names and their similarity scores
        results = [
            (self.doc_id_to_name[doc_id], score) for doc_id, score in ranked_docs
        ]
        return results

    def run(self):
        """
        Run the VSM model interactively for user queries.
        """
        print("Building index...")
        self.build_index()  # Build the index (inverted index and document lengths)
        print("Index built.")
        # Continuously prompt for user queries
        while True:
            query = input("Enter your search query (or type 'exit' to quit): ")
            if query.lower() == "exit":  # Exit condition
                break
            results = self.search(query)  # Search the corpus for relevant documents
            if not results:
                print("No relevant documents found.")
            else:
                print("Top documents (Name, Score):")
                for name, score in results:
                    print(f"{name}: {score:.4f}")  # Display document names and scores


# Running the model
if __name__ == "__main__":
    corpus_path = "Corpus"  # Path to the corpus directory (update this as needed)
    vsm = VSM(corpus_path)  # Create the VSM object
    vsm.run()  # Run the VSM model interactively
