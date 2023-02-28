# Utafiti (Swahili for Research)

- This is a Streamlit app that uses the sentence-transformers library to find similar sentences in a corpus of text. It is based on the [Sentence-BERT](https://www.sbert.net/) model.
- The corpus is a collection of research paper titles in the realm of mathematics and computer science. Only the titles are used for purposes of similarity search. The abstracts as thus fetched by matching title to abstract (summary) indices stored as pickle files in the resources directory.
- For query optimization, the faiss (Facebook AI Similarity Search) library is used to index the corpus. Additionally, clustering using Voronoi cells has been implemented to further optimize the search.
- Hosted on HuggingFace Spaces at: https://huggingface.co/spaces/eolang/utafiti

## Additonal details include:
    - Encoding size: 768
    - Corpus size: 28,000
    - Library: Sentence-BERT (Sentence Transformers), Faiss, Torch, NumPy, Streamlit
    - Model: multi-qa-MiniLM-L6-cos-v1

### Possible Fixes (*This is a hobby project, so no promises* :-)
1. Inlculssion of direct links to Axriv, Papers with Code and Google Scholar links for the papers
2. An API of some sort to allow for easy integration with other projects (maybe in the infamous version 2.0)
3. A more robust search engine (maybe in the infamous version 2.0)


### Random things to note
- I'm not running this on Streamlit Sharing, due to an issue with the fails-CPU library. **May** migrate it in case I find a fix.
- I'm not the best developer, so the code is not the best. I'm still learning and I'm open to suggestions and improvements.
- I'm not a professional mathematician &/ researcher &/ scientist, so the corpus, algorithms, etc. may not be the best. I'm still learning and I'm open to suggestions and improvements.

### Acknowledgements
- [Sentence-BERT](https://www.sbert.net/)
- [multi-qa-MiniLM-L6-cos-v1]((https://huggingface.co/sentence-transformers/multi-qa-MiniLM-L6-cos-v1))


## For feedback and suggestions, please contact me on:
- [Twitter](https://twitter.com/Olangjoe)
- [LinkedIn](https://www.linkedin.com/in/eolang/)
- [Email](mailto:oluoch9@gmail.com)
