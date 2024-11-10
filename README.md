# Document Retrieval System with PyLucene


This repository contains a document retrieval system built with PyLucene for indexing, querying, and evaluating text datasets. It uses Lucene's indexing and search capabilities and computes Mean Average Precision (MAP@K) and Mean Average Recall (MAR@K) to assess retrieval performance.

It is done as part of assignment for the Information Retieval course at the University of Antwerp. (2024/25)


## Prerequisites


- **Python 3.x**
- **PyLucene** installed and configured
- **Java Development Kit (JDK)** configured on your system


To install PyLucene, please follow [PyLucene's official instructions](http://lucene.apache.org/pylucene/).


## Directory Structure


- `index/` : Directory where the indexed documents are stored.
- `full_docs/` : Directory with text files to be indexed.
- `dev_queries.csv` : CSV file with queries for retrieval.
- `dev_query_results.csv` : CSV file with ground truth results for evaluation.
- `result.csv` : Output CSV file where retrieval results are saved.
- `evaluation_result.txt` : File where evaluation metrics (MAP@K and MAR@K) are saved.


## Overview of Key Components


### 1. **Indexing Documents**


Documents in `full_docs/` are indexed using Lucene’s `IndexWriter`. Text content and unique document IDs are added to the index.


#### Usage:
Set the path to your dataset in `dataset_path`:
```python
dataset_path = "full_docs"  # Change this to your dataset directory
```


Run the indexing function:
```python
index_documents(writer, dataset_path)
```


### 2. **Querying the Index**


Queries from a CSV file (`dev_queries.csv`) are processed, and the top 10 document IDs for each query are retrieved. Special characters in queries are handled by the `clean_query` function.


The search function can use different similarity measures:
- **BM25Similarity**
- **LMDirichletSimilarity**
- **LMJelinekMercerSimilarity**


#### Usage:
Set the similarity measure in the `search` function by uncommenting the desired option:
```python
searcher.setSimilarity(BM25Similarity())
# searcher.setSimilarity(LMDirichletSimilarity())
# searcher.setSimilarity(LMJelinekMercerSimilarity(0.7))
```


### 3. **Processing Queries and Saving Results**


Processes queries from a CSV file and saves the top 10 results for each query to a CSV file (`result.csv`).


#### Usage:
```python
process_queries_csv(query_file="dev_queries.csv", analyzer=analyzer, index_path="index", output_file="result.csv")
```


### 4. **Evaluation (MAP@K and MAR@K)**


Calculates MAP@K and MAR@K for a range of `K` values using the ground truth data from `dev_query_results.csv`.


#### Usage:
```python
evaluate(output_file="result.csv", ground_truth_file="dev_query_results.csv", k_values=[1, 3, 5, 10], result_file="evaluation_result.txt")
```


## Detailed Usage Instructions


1. **Set Up Environment**: Install required libraries and ensure PyLucene and JDK are properly configured.
2. **Indexing**: Run the `index_documents` function if the index doesn’t exist.
3. **Running Queries**: Set your query file and similarity measure in the `search` function, then run `process_queries_csv`.
4. **Evaluate Results**: Run the `evaluate` function to compute MAP@K and MAR@K and save results to `evaluation_result.txt`.


## Additional Notes


- **Dataset and Indexing**: Modify `dataset_path` as needed.
- **Evaluation Ground Truth**: Update `ground_truth_file` if using a different ground truth dataset.
- **Result Output**: Customize `output_file` and `result_file` paths as required.
