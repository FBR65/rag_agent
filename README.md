# RAG_AGENT

The study[^1] of the Hebrew University of Jerusalem suggests that limiting the number of documents 
retrieved and made available to the language model and ensuring that these documents are as relevant
as possible may be beneficial for the effectiveness of RAG systems. 

The used Dataset[^2] is published on Github.

## Intention of the Project

The intention of this project is to implement a RAG system based on the results of the study. 

This will be done using agents with the **PydanticAI** framework. 

## How it's done

**data_intake.py**

* The data was read into Qdrant at all possible distances using data_intake.py.
    * read data from File with Apache Tika
    * uses semantic chunking
    * uses EUCLID, DOT, COSINE and MANHATTAN
    * uses dense, sparse and fulltext


**data_retrieve.py**

* For each distance, a hybrid search is performed using dense and sparse vectors and full-text searches based on reciprocal rerank fusion.
* The results of all hybrid searches are summarized, reranked, and then deduplicated


**rag_agent.py**

* LLM generates answer based on query and retrieved information.
* Validator checks answer against query and retrieved information; suggests improvements if needed.
* LLM refines answer (max 3 iterations).
* Present solution.


### Spacy and uv
Normaly you would install the Spacy Model like:

python -m spacy download de_core_news_sm

But if you are using uv this won't work.
You will get the Error Message "No module named pip"

Fortunately, the releases of spaCy models are also released as .whl. So you may install this like:

uv pip install de_core_news_sm@https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl


### Licence and Copyright

The Project is published under the AGPLv3 License.



[^1]: https://arxiv.org/abs/2503.04388

[^2]: https://github.com/shaharl6000/MoreDocsSameLen