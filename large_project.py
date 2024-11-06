import lucene
lucene.initVM()

from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.core import LowerCaseFilter, StopFilter
from org.apache.lucene.analysis import Analyzer
from org.apache.lucene.analysis.miscellaneous import ASCIIFoldingFilter
from org.apache.lucene.document import Document, Field, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.search import IndexSearcher
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.store import FSDirectory
from java.nio.file import Paths
import pandas as pd
import os
import re
import csv

# Set up directory and analyzer with custom filters
index_path = "index"
directory = FSDirectory.open(Paths.get(index_path))
analyzer = EnglishAnalyzer()  # Use EnglishAnalyzer directly
config = IndexWriterConfig(analyzer)
writer = IndexWriter(directory, config)

# Function to index a single document
def index_document(doc_id, content, writer):
    doc = Document()
    doc.add(TextField("content", content, Field.Store.YES))
    doc.add(TextField("doc_id", doc_id, Field.Store.YES))
    writer.addDocument(doc)

# Function to index all documents in a directory
def index_documents(writer, dataset_path):
    for filename in os.listdir(dataset_path):
        if filename.endswith(".txt"):
            with open(os.path.join(dataset_path, filename), "r") as file:
                content = file.read()
                index_document(filename, content, writer)
    writer.close()

# Choose the dataset path based on the dataset size you want to work with
dataset_path = "full_docs"  # Change this to your dataset directory
index_documents(writer, dataset_path)
print("Indexing completed.")

# Function to safely escape special characters in the query
def clean_query(query_str):
    query_str = re.sub(r'[^\w\s]', ' ', query_str)
    query_str = re.sub(r'\s+', ' ', query_str)
    return query_str.strip()

# Search function to retrieve top n document IDs for a query
def search(query_str, analyzer, index_path, top_n=10):
    query_str = clean_query(query_str)
    directory = FSDirectory.open(Paths.get(index_path))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())  # Set BM25 Similarity

    query = QueryParser("content", analyzer).parse(query_str)
    hits = searcher.search(query, top_n).scoreDocs  # Retrieve exactly top_n documents
    
    results = []
    for hit in hits:
        doc_id = hit.doc
        doc = reader.storedFields().document(doc_id)
        doc_unique_id = doc.get("doc_id")
        
        # Extract only the integer part from the document ID
        doc_integer_id = int(re.search(r'\d+', doc_unique_id).group())
        results.append(doc_integer_id)  # Append doc ID to results list
    
    reader.close()
    return results

# Process queries and save top 10 results for each query to a CSV
def process_queries_csv(query_file, analyzer, index_path, output_file="output_results.csv"):
    queries = pd.read_csv(query_file, sep="\t", header=0, names=["Query_number", "Query"])

    # Open the output file in write mode
    with open(output_file, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Query_number", "Doc_number"])  # Write headers

        for idx, row in queries.iterrows():
            query_id = row["Query_number"]
            query_text = row["Query"]

            print(f"Processing Query ID {query_id}: {query_text.strip()}")
            results = search(query_text.strip(), analyzer, index_path, top_n=10)  # Get top 10 unique results

            for doc_id in results:
                writer.writerow([query_id, doc_id])

    print(f"Query processing completed. Results saved to {output_file}.")

# Specify the query file and process it
query_file = "dev_queries.tsv"  # Replace with your query file path
process_queries_csv(query_file, analyzer, index_path, output_file="final_output_large.csv")

# Evaluation functions remain unchanged, so you can reuse your evaluation code here.
# Ensure you run the evaluate function as needed.


# Evaluation functions
def load_results(output_file):
    """Load the output results from a CSV file and standardize column names."""
    results = pd.read_csv(output_file)
    results.columns = results.columns.str.lower()  # Convert column names to lowercase
    return results

def load_ground_truth(ground_truth_file):
    """Load the ground truth data from a CSV file and standardize column names."""
    ground_truth = pd.read_csv(ground_truth_file)
    ground_truth.columns = ground_truth.columns.str.lower()  # Convert column names to lowercase
    return ground_truth

def compute_average_precision(retrieved, relevant):
    """Compute the average precision for a single query."""
    if len(relevant) == 0:
        return 0.0

    ap = 0.0
    relevant_count = 0

    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_count += 1
            ap += relevant_count / (i + 1)  # Precision at this rank

    return ap / len(relevant)  # Normalize by the number of relevant documents

def compute_mean_average_precision(results, ground_truth, k):
    """Compute MAP@K."""
    average_precisions = []

    for query_id in results["query_number"].unique():
        # Get relevant documents for this query
        relevant = set(
            ground_truth[ground_truth["query_number"] == query_id]["doc_number"]
        )

        # Get top K retrieved documents for this query
        retrieved = (
            results[results["query_number"] == query_id]["doc_number"].head(k).tolist()
        )

        ap = compute_average_precision(retrieved, relevant)
        average_precisions.append(ap)

    return (
        sum(average_precisions) / len(average_precisions) if average_precisions else 0.0
    )

def compute_average_recall(retrieved, relevant):
    """Compute the average recall for a single query."""
    if len(relevant) == 0:
        return 0.0

    retrieved_set = set(retrieved)
    relevant_set = set(relevant)

    # Recall = TP / (TP + FN)
    tp = len(retrieved_set.intersection(relevant_set))  # True positives
    recall = tp / len(relevant_set)

    return recall

def compute_mean_average_recall(results, ground_truth, k):
    """Compute MAR@K."""
    average_recalls = []

    for query_id in results["query_number"].unique():
        # Get relevant documents for this query
        relevant = set(
            ground_truth[ground_truth["query_number"] == query_id]["doc_number"]
        )

        # Get top K retrieved documents for this query
        retrieved = (
            results[results["query_number"] == query_id]["doc_number"].head(k).tolist()
        )

        recall = compute_average_recall(retrieved, relevant)
        average_recalls.append(recall)

    return sum(average_recalls) / len(average_recalls) if average_recalls else 0.0

def evaluate(output_file, ground_truth_file, k_values, result_file):
    """Evaluate the results and print MAP@K and MAR@K, saving results to a file."""
    results = load_results(output_file)
    ground_truth = load_ground_truth(ground_truth_file)

    with open(result_file, "w") as f:
        for k in k_values:
            map_k = compute_mean_average_precision(results, ground_truth, k)
            mar_k = compute_mean_average_recall(results, ground_truth, k)

            output_str = f"MAP@{k}: {map_k:.4f}\nMAR@{k}: {mar_k:.4f}\n"
            print(output_str)
            f.write(output_str)

# Run the evaluation
evaluate('final_output_large.csv', 'dev_query_results.csv', [3, 10], 'results_large.txt')
