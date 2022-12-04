import flask
from flask import Flask, render_template, request, jsonify
import pickle
import json
from sentence_transformers import SentenceTransformer

# Running the flask app
app = Flask(__name__)

#load model using pickle
with open('resources/titles.pkl', 'rb') as f:
    titles = pickle.load(f)

with open('resources/embeddings.pkl', 'rb') as f:
    embeddings = pickle.load(f)

with open('resources/authors.pkl', 'rb') as f:
    authors = pickle.load(f)

with open('resources/years.pkl', 'rb') as f:
    years = pickle.load(f)

with open('resources/summary.pkl', 'rb') as f:
    summary = pickle.load(f)

index = pickle.load(open('resources/index.sav' , 'rb'))

#model = pickle.load(open('resources/model.sav' , 'rb'))
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

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
    
    # conver to json
    #results = json.dumps(results)
    return results


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    query_text = [str(x) for x in request.form.values()]
    query = query_text[0]
    results = retrieve(query)
    return render_template('index.html', search_results=results)




# Run the app
if __name__ == '__main__':
    app.run(debug=True)