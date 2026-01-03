import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()

model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return stemmer.stem(text)

def predict_sentiment(text):
    text = preprocess(text)
    vector = tfidf.transform([text])
    return model.predict(vector)[0]
