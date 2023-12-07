import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import re

class GloveVectorizer:
    def __init__(self, glove_file_path, max_len=31):
        self.glove_model = KeyedVectors.load_word2vec_format(glove_file_path, binary=False, encoding='utf-8', unicode_errors='ignore', no_header=True)
        self.max_len = max_len

    def clean_text(self, text):
        # Expand contractions
        text = re.sub(r"can\'t", "cannot", text)
        text = re.sub(r"n\'t", " not", text)
        text = re.sub(r"\'re", " are", text)
        text = re.sub(r"It's", "It is", text)
        text = re.sub(r"let's", "let us", text)
        text = re.sub(r"\'ve", " have", text)
        text = re.sub(r"\'ll", " will", text)

        # Remove email address, urls and links
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\S+@\S+', '', text, flags=re.MULTILINE)

        text = re.sub(r'@\S+', '', text) #Remove words that starts with @ (i.e tweets that start with @ are mostly username that got tagged)

        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)

        text = re.sub(r'\s+', ' ', text) #Remove consecutive whitespace characters

        # Convert to lowercase
        text = text.lower().strip()

        # Remove Emoji
        text = emoji_pattern.sub(r'', text)

        # Tokenization and stopwords removal
        # tokens = word_tokenize(text)
        # filtered_tokens = [token for token in tokens if token not in stop_words]

        # # Part-of-speech tagging
        # pos_tagged = nltk.pos_tag(filtered_tokens)

        # # Lemmatization
        # lemmatized_text = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tagged]

        return text.strip()

    def glove_embed(self, text):
        text_list = word_tokenize(text)
        sentence_embeddings = []

        for token in text_list:
            try:
                sentence_embeddings.append(self.glove_model[token])
            except:
                continue

        return np.array(sentence_embeddings)

    def vectorize_text(self, text):
        cleaned_text = self.clean_text(text)
        embeddings = self.glove_embed(cleaned_text)

        len_difference = self.max_len - embeddings.shape[0]
        pad = np.zeros(shape=(len_difference, 100))  # Assuming each word vector is of size 100
        return np.concatenate([embeddings, pad])

    def vectorize_data(self, data):
        if isinstance(data, pd.Series):
            return np.array([self.vectorize_text(text) for text in data])
        elif isinstance(data, list):
            return np.array([self.vectorize_text(text) for text in data])
        else:
            raise ValueError("Unsupported data type. Please provide a pandas Series or a list.")

# Example usage:
# Assuming df is your DataFrame and 'text_column' is the column you want to process
glove_file_path = '/content/drive/MyDrive/glove.twitter.27B.100d.txt'
vectorizer = GloveVectorizer(glove_file_path)

# Example DataFrame
data_series = pd.Series(["Your text goes here.", "Another example text."])
data_list = ["Yet another text.", "And one more example."]

# Vectorize Series
resulting_array_series = vectorizer.vectorize_data(data_series)
print(resulting_array_series.shape)

# Vectorize List
resulting_array_list = vectorizer.vectorize_data(data_list)
print(resulting_array_list.shape)
