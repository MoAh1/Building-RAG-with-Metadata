import llama_index
import chromadb
import pickle
from llama_index.legacy.vector_stores import ChromaVectorStore
from llama_index.legacy.query_engine.retriever_query_engine import RetrieverQueryEngine
from llama_index.legacy.schema import MetadataMode
from llama_index.legacy import (
    VectorStoreIndex,
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
    Document,
)
from pathlib import Path
from tqdm import tqdm
from llama_index.legacy.node_parser import SimpleNodeParser
import re
from tenacity import retry, stop_after_attempt, wait_random_exponential
import os
import pandas as pd

def normalize_string(s):
    """Helper function to normalize strings for comparison."""
    if isinstance(s, str):
        s = s.lower()
        s = re.sub(r'[^\w\s]', '', s)
        s = re.sub(r'\s+', ' ', s)
        return s.strip()
    return ""

def get_metadata_for_file(filename, metadata_df=None):
    """
    Retrieves metadata for a given file by matching it with entries in a metadata DataFrame.
    Parameters:
    - filename (str): The name of the file for which metadata is being retrieved.
    - metadata_df (DataFrame, optional): The DataFrame containing metadata.

    Returns:
    A dictionary containing metadata if a match is found; otherwise, an empty dict.
    """
    if metadata_df is None:
        return {}

    if filename.endswith('.pdf') or filename.endswith('.txt'):
        base_filename = os.path.splitext(filename)[0]
        normalized_filename = normalize_string(base_filename)
        if 'normalized_file_name' not in metadata_df.columns:
            if 'file_name' in metadata_df.columns:
                metadata_df['normalized_file_name'] = metadata_df['file_name'].apply(normalize_string)
            else:
                print("Metadata DataFrame does not contain 'file_name' column.")
                return {}
        matched_rows = metadata_df[metadata_df['normalized_file_name'] == normalized_filename]
        if not matched_rows.empty:
            row = matched_rows.iloc[0]
            metadata = {}
            fields = {
                'Year': 'Publication Year',
                'Author': 'Author',
                'Title': 'Title',
                'Link': 'Url'
            }
            for new_key, old_key in fields.items():
                value = row.get(old_key, None)
                if pd.notna(value): 
                    if isinstance(value, (int, float)):
                        metadata[new_key] = value
                    else:
                        metadata[new_key] = str(value)

            print(f"Metadata found for: {filename}")
            return metadata
        else:
            print(f"No metadata match found for: {filename}")
    return {}

@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(6))
def building_vector_with_metadata(chroma_save_path, files_source, llm, embedding, metadata_df=None):
    """
    Constructs a vector database with optional metadata support.
    Keeps track of processed files so it can resume if interrupted.

    Parameters:
    - chroma_save_path (str): The path where the ChromaDB database should be saved.
    - files_source (str): Path to the directory containing the source files to be processed.
    - llm (object): Language model object for prediction.
    - embedding (object): Embedding model for document representation.
    - metadata_df (DataFrame, optional): DataFrame containing metadata for files.

    Returns: chromadb vector store
    """
    chroma_client = chromadb.PersistentClient(path=chroma_save_path)
    chroma_collection = chroma_client.get_or_create_collection(chroma_save_path)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    service_context = ServiceContext.from_defaults(
        llm=llm,
        embed_model=embedding
    )
    source_directory = Path(files_source)
    all_files = [f.name for f in source_directory.iterdir() if f.is_file()]
    log_file = 'processed_files_log.txt'
    processed_files = set()
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f)

    def split_text_into_pages(text):
        pages = text.split('<!-- PageBreak -->')
        return [page.strip() for page in pages if page.strip()]

    for filename in tqdm(all_files, desc="Processing files"):
        if filename in processed_files:
            continue
        metadata_info = get_metadata_for_file(filename, metadata_df)
        file_metadata_callable = lambda x: metadata_info
        reader = SimpleDirectoryReader(
            input_files=[source_directory / filename],
            file_metadata=file_metadata_callable
        )
        doc = reader.load_data()
        if not doc:
            print(f"Could not load file {filename}. Skipping.")
            continue
        print(f'\n{filename}: {len(doc[0].text)} characters')
        pages = split_text_into_pages(doc[0].text)
        if not pages:
            pages = [doc[0].text]
            print(f"No page markers found in {filename}, using entire document as one page.")
        else:
            print(f'Number of pages found: {len(pages)}')
        documents = []
        for i, page_text in enumerate(tqdm(pages, desc=f"Processing pages in {filename}", leave=False), start=1):
            page_doc = Document(
                text=page_text,
                doc_id=f"{doc[0].doc_id}_page_{i}",
                extra_info=metadata_info
            )
            documents.append(page_doc)
        parser = SimpleNodeParser.from_defaults()
        nodes = parser.get_nodes_from_documents(documents)
        if nodes:
            curr_index = VectorStoreIndex(
                nodes,
                service_context=service_context,
                storage_context=storage_context
            )
            storage_context.persist()
        else:
            print(f"No valid nodes found in {filename}. Skipping.")
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(filename + '\n')
        processed_files.add(filename)

if __name__ == "__main__":
    from config_llm import llm_gpt4_mini, embeddings
    import pandas as pd
    
    # Configuration
    chroma_save_path = "RagwithMedata"
    files_source = "TextFiles"
    llm = llm_gpt4_mini
    embedding = embeddings
    try:
        metadata_df = pd.read_csv('metadata.csv')
    except FileNotFoundError:
        print("Metadata file not found. Proceeding without metadata.")
        metadata_df = None
    building_vector_with_metadata(chroma_save_path, files_source, llm, embedding, metadata_df=metadata_df)
