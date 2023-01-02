import pickle
import numpy as np
import json
from sentence_transformers import SentenceTransformer
import streamlit as st

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

model = pickle.load(open('Files/model.sav' , 'rb'))

def retrieve(query, k=5):
    xq = model.encode([query])
    D, I = index.search(xq, k)

    results = []
    for i in range(k):
        results.append(
            { 
                'Title': titles[I[0][i]],
                'Author': authors[I[0][i]],
                'Year': years[I[0][i]],
                'Summary': summary[I[0][i]]
            }
        )

    return results

st.title('Semantic Search')
st.write('This is a semantic search engine that uses sentence embeddings to find similar documents. The model used is the DistilBERT model trained on the NLI dataset. The dataset used is the arXiv dataset. The dataset can be found [here](https://www.kaggle.com/Cornell-University/arxiv).')

# single inptu text box
query = st.text_input('Enter your query')

# button to search
if st.button('Search'):
    results = retrieve(query)
    for result in results:
        st.write(result['Title'])
        st.write(result['Author'])
        st.write(result['Year'])
        st.write(result['Summary'])
        st.write('---------------------------------')