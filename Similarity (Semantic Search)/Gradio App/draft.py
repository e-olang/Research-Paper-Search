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

with open('titles.pkl', 'rb') as f:
    titles = pickle.load(f)

with open('embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)


# convert to numpy array
titles = np.array(titles)
embeddings = np.array(embeddings)

# ----------------- model  -----------------
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

# ----------------- faiss_index  -----------------
nlist = 50
m = 8
bits = 8
dimensions = embeddings.shape[1]

quantizer = faiss.IndexFlatIP(dimensions)
index = faiss.IndexIVFPQ(quantizer, dimensions, nlist, m, bits)
index.train(embeddings)
index.add(embeddings)

# ----------------- search  -----------------

def get_embeddings(query):
    xq = model.encode([query])
    return xq

def get_similar_documents(query):
    k = 10
    xq = get_embeddings(str(query))
    D, I = index.search(xq, k)
    # return list of similar documents
    return [titles[i] for i in I[0]]


# ----------------- gradio app  -----------------
iface = gr.Interface(
    fn=get_similar_documents,
    inputs = gr.Textbox(lines=1, placeholder="Enter Query...", label="Query text"),

    # variable number of outputs based on the number of results argument 
    outputs = [ gr.outputs.Textbox(label="First similar document"),
                gr.outputs.Textbox(label="Second similar document"),
                gr.outputs.Textbox(label="Third similar document"),
                gr.outputs.Textbox(label="Fourth similar document"),
                gr.outputs.Textbox(label="Fifth similar document"),
                gr.outputs.Textbox(label="Sixth similar document"),
                gr.outputs.Textbox(label="Seventh similar document"),
                gr.outputs.Textbox(label="Eighth similar document"),
                gr.outputs.Textbox(label="Ninth similar document"),
                gr.outputs.Textbox(label="Tenth similar document")],
   
    title="Search Engine",
    description="Search Engine using Sentence Transformers and Faiss Indexing",
    allow_flagging=False,
    allow_screenshot=False,
    allow_output_caching=False,

    )

iface.launch(inline=False)