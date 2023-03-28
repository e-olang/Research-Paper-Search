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
    st.title("Research Paper/ Publication Search Engine")

    # app description
    st.markdown(f"<p style='text-align: justify;'>This is a search engine that uses a pre-trained sentence transformer model to encode the user query and the abstracts of the papers. The encoded vectors are then used to find the most similar papers to the user query.", unsafe_allow_html=True)

    # DISCLAIMER: Data capture
    st.markdown(f"<p style='text-align: justify;'>Please note that the search engine captures the user query and stores it in a text file. This is done to build a collection of user queries for future use when building a bigger data pool for the search engine.  \nline", unsafe_allow_html=True)
    

    query_text = st.text_input("Enter your query e.g. NLP")

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
        

        # store user query in a text file, 
        # The purpose of this is to build a collection of user queries for future use when building a bigger data pool for the search engine
        with open('resources/user_query.txt', 'a') as f:
            f.write(query_text + "\n")
        

if __name__ == "__main__":
    main()
