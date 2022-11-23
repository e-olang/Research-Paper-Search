import flask
from flask import Flask, render_template, request
import pickle
from sentence_transformers import SentenceTransformer


# ----------------- load files and model -----------------
with open('titles.pkl', 'rb') as f:
    titles = pickle.load(f)

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('authors.pkl', 'rb') as f:
    authors = pickle.load(f)

with open('years.pkl', 'rb') as f:
    years = pickle.load(f)

with open('summary.pkl', 'rb') as f:
    summary = pickle.load(f)

index = pickle.load(open('Files/index.sav' , 'rb'))

model = pickle.load(open('Files/model.sav' , 'rb'))
#model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# ----------------- flask app -----------------
app = Flask(__name__)

def home():
    return render_template('index.html')



