# set conda environment for current script to '(nlp2)'
# run in terminal: conda activate nlp2

import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import gradio as gr

import warnings
warnings.filterwarnings('ignore')

# ----------------- load data -----------------

with open('Files/titles.pkl', 'rb') as f:
    titles = pickle.load(f)

with open('Files/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('Files/authors.pkl', 'rb') as f:
    authors = pickle.load(f)

with open('Files/years.pkl', 'rb') as f:
    years = pickle.load(f)

with open('Files/summary.pkl', 'rb') as f:
    summary = pickle.load(f)

index = pickle.load(open('Files/index.sav' , 'rb'))

# ----------------- model  -----------------
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# ----------------- search  -----------------

def retrieve(query, k=5):
    xq = model.encode([query])
    D, I = index.search(xq, k)

    results = {'Result 1': [], 'Result 2': [], 'Result 3': [], 'Result 4': [], 'Result 5': []}
    for i in range(k):
        results['Result '+str(i+1)].append(titles[I[0][i]])
        results['Result '+str(i+1)].append(authors[I[0][i]])
        results['Result '+str(i+1)].append(years[I[0][i]])
        results['Result '+str(i+1)].append(summary[I[0][i]])
    return results


# ----------------- gradio app  -----------------
interface = gr.Interface(
    fn=retrieve,
    inputs = gr.inputs.Textbox(lines=1, placeholder="Enter Query...", label="Query text"),
    # a json output with 4 keys: titles, authors, years, summary
    outputs = gr.outputs.JSON(label="Similar Documents"),
    title="Semantic Search",
    description="Search for similar documents using semantic search.",
    allow_flagging=False,
    examples=[
        ["Mathematical models of the spread of infectious diseases in humans and animals"],
        ["A new method for solving the nonlinear eigenvalue problem"]
    ]
)

interface.launch(inline=False)