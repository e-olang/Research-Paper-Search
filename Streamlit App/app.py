import pickle
import json
import streamlit as st
import pandas as pd


#load files/resources using pickle
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

with open('resources/titles_urls.pkl', 'rb') as f:
    title_urls = pickle.load(f)


index = pickle.load(open('resources/index.sav' , 'rb'))
model = pickle.load(open('resources/model.sav' , 'rb'))


# Query Function
def query(query_text, k=5):
    enc = model.encode([query_text])
    D, I = index.search(enc, k)

    results = []
    for i in range(k):
        results.append(
            {
                "Title" : titles[I[0][i]],
                "Author" : authors[I[0][i]],
                "Year" : years[I[0][i]],
                "Summary" : summary[I[0][i]],
                "Title URL" : title_urls[I[0][i]]
            }
        )
    
    return results


def main():
    # Streamlit App
    st.title("Search Engine")

    query_text = st.text_input("Enter your query")

    if st.button("Search"):
        results = query(query_text)
        
        for result in results:
            st.write("Title:   " + result["Title"])
            st.write("Author:   " + result["Author"])
            st.write("Year Published:   " +  str(result["Year"]))
            
            # justified text for summary
            st.write("Abstract")
            st.markdown(f"<p style='text-align: justify;'>{result['Summary']}</p>", unsafe_allow_html=True)
            st.markdown(f"<a href='{result['Title URL']}'>Google Search</a>", unsafe_allow_html=True)
            st.write("--" * 50)

if __name__ == "__main__":
    main()
