import os
import json
import time
import numpy as np
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import faiss
import hnswlib
import asyncio
from tqdm import tqdm

# LlamaIndex Imports
from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
    Settings,
    PromptTemplate,
    load_index_from_storage,
)
from llama_index.core.schema import NodeWithScore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.hnswlib import HnswlibVectorStore
from llama_index.vector_stores.faiss import FaissVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from google.genai.types import EmbedContentConfig

# --- Configuration ---


class Config:
    """Configuration class for the benchmark."""

    API_MODEL_NAME = "gpt-5-mini"
    EMBEDDING_MODEL_NAME = "text-embedding-3-small"
    EMBED_DIM = 384  # all-MiniLM-L6-v2
    JSON_FILE_PATH = r"C:\Users\yalkh\Documents\Projects\rag-index-benchmark\data\2wikimultihop\dev.json"
    MAX_QUERIES = -1  # Set to -1 to use all queries
    TOP_K = 10
    CHUNK_SIZE = 200
    CHUNK_OVERLAP = 20
    CONCURRENT_REQUESTS = 25

    # HNSWlib Params
    HNSW_M = 128
    HNSW_EF_CONSTRUCTION = 256
    HNSW_EF_SEARCH = 256
    HNSW_SPACE = "cosine"

    # IVF Params
    IVF_NLIST = 10  # Number of clusters
    IVF_NPROBE = 4
    FAISS_METRIC = faiss.METRIC_INNER_PRODUCT


# --- 1. Data Loading ---


def load_data(
    json_path: str, max_queries: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads the dataset from JSON and returns the full data and the query subset."""
    print(f"Loading dataset from: {json_path}...")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            full_data = json.load(f)
        print(f"Loaded {len(full_data)} examples from file.")

        if 0 < max_queries < len(full_data):
            query_subset = full_data[:max_queries]
        else:
            query_subset = full_data

        # For this setup, we index *only* the context from the subset
        index_data_subset = query_subset

        print(f"Using first {len(query_subset)} examples for queries.")
        print(f"Using first {len(index_data_subset)} examples for index context.")
        return index_data_subset, query_subset

    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}")
        exit()
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from file {json_path}.")
        exit()


# --- 2. Document Preparation (Chunking) ---


def prepare_documents_chunked(
    data_subset: List[Dict[str, Any]], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """Converts data subset into LlamaIndex Document nodes using static chunking."""
    print("Preparing documents for indexing...")
    documents = []
    skipped_rows = 0

    for item in data_subset:
        question_id = item.get("_id")
        context_data = item.get("context", None)

        # Create one Document per *paragraph* (title)
        for title, sentences in context_data:
            # Combine sentences into a single text block for chunking
            full_paragraph_text = " ".join(sentences)

            try:
                # Clean text for safe encoding
                cleaned_text = full_paragraph_text.encode(
                    "utf-8", "surrogateescape"
                ).decode("utf-8", "replace")
            except UnicodeEncodeError:
                cleaned_text = full_paragraph_text.encode("ascii", "ignore").decode(
                    "ascii"
                )

            if not cleaned_text.strip():
                continue

            documents.append(
                Document(
                    text=cleaned_text,
                    metadata={
                        "title": title,
                        "question_id": question_id,  # Link chunk to its original question
                    },
                )
            )

    print(
        f"Finished preparing {len(documents)} chunk-documents. Skipped {skipped_rows} items."
    )

    # Apply static chunking
    parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = parser.get_nodes_from_documents(documents, show_progress=True)

    print(f"Chunked into {len(nodes)} nodes (chunk_size={chunk_size}).")
    return nodes


def prepare_documents_sentence(data_subset: List[Dict[str, Any]]) -> List[Document]:
    """Converts data subset into LlamaIndex Document nodes using static chunking."""
    print("Preparing documents for indexing...")
    documents = []
    skipped_rows = 0

    for item in data_subset:
        question_id = item.get("_id")
        context_data = item.get("context", None)

        # Create one Document per *paragraph* (title)
        for title, sentences in context_data:
            for sent_id, sentence_text in enumerate(sentences):
                try:
                    # Clean text for safe encoding
                    cleaned_text = sentence_text.encode(
                        "utf-8", "surrogateescape"
                    ).decode("utf-8", "replace")
                except UnicodeEncodeError:
                    cleaned_text = sentence_text.encode("ascii", "ignore").decode(
                        "ascii"
                    )

                if not cleaned_text.strip():
                    continue

                formatted_text = f"title: {title or 'none'} | text: {cleaned_text}"

                documents.append(
                    Document(
                        text=cleaned_text,
                        metadata={
                            "title": title,
                            "question_id": question_id,
                            "sent_id": sent_id,
                        },
                    )
                )

    print(
        f"Finished preparing {len(documents)} sentence-documents. Skipped {skipped_rows} items."
    )

    return documents


# --- 3. Index Abstraction & Factory ---


class IndexFactory:
    """Builds or loads a specific vector store index."""

    def __init__(self, config: Config, persist_dir: str):
        self.config = config
        self.persist_dir = persist_dir

    def _build_index(self, vector_store, documents: List[Document]) -> VectorStoreIndex:
        """Helper to build a new index."""
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        print("Building vector store index (this may take time)...")
        index = VectorStoreIndex(
            documents,  # Use nodes from prepare_documents
            storage_context=storage_context,
            show_progress=True,
        )
        print(f"Index built. Persisting index to {self.persist_dir}...")
        index.storage_context.persist(persist_dir=self.persist_dir)
        time.sleep(1)  # Ensure files are written
        print("Index persisted.")
        return index

    def _load_index(self, index_type: str) -> Optional[VectorStoreIndex]:
        """Helper to load an existing index using explicit component loading."""
        print(f"Loading index from {self.persist_dir}...")
        try:
            # 1. Load the specific BINARY vector store
            if index_type == "hnsw":
                vector_store = HnswlibVectorStore.from_persist_dir(self.persist_dir)
            elif index_type == "ivf":
                vector_store = FaissVectorStore.from_persist_dir(self.persist_dir)
            else:
                raise ValueError(f"Unknown index_type for loading: {index_type}")

            # 2. Load the JSON-based stores explicitly (Your snippet)
            print("Loading docstore and index_store from disk...")
            storage_context = StorageContext.from_defaults(
                docstore=SimpleDocumentStore.from_persist_dir(
                    persist_dir=self.persist_dir
                ),
                vector_store=vector_store,  # Use the specific store loaded above
                index_store=SimpleIndexStore.from_persist_dir(
                    persist_dir=self.persist_dir
                ),
            )

            # 3. Load the final index from the reconstituted storage
            index = load_index_from_storage(
                storage_context=storage_context,
                embed_model=Settings.embed_model,
            )
            print("Index and StorageContext loaded successfully.")
            return index

        except Exception as e:
            print(f"Error loading index from storage: {e}")
            print("Will attempt to rebuild...")
            return None  # Signal to rebuild

    def get_index(self, index_type: str, documents: List[Document]) -> VectorStoreIndex:
        """Gets or builds the specified index."""
        if os.path.exists(self.persist_dir):
            index = self._load_index(index_type)
            if index:
                # Check if vector store type matches
                if index_type == "hnsw" and isinstance(
                    index.vector_store, HnswlibVectorStore
                ):
                    return index
                if index_type == "ivf" and isinstance(
                    index.vector_store, FaissVectorStore
                ):
                    return index
                print("Warning: Index type mismatch, rebuilding...")

        # Build new index
        if index_type == "hnsw":
            hnswlib_index = hnswlib.Index(
                space=self.config.HNSW_SPACE, dim=self.config.EMBED_DIM
            )
            hnswlib_index.init_index(
                max_elements=len(documents),
                ef_construction=self.config.HNSW_EF_CONSTRUCTION,
                M=self.config.HNSW_M,
            )
            vector_store = HnswlibVectorStore(hnswlib_index)
            return self._build_index(vector_store, documents)

        elif index_type == "ivf":
            quantizer = faiss.IndexFlat(self.config.EMBED_DIM, self.config.FAISS_METRIC)
            faiss_index = faiss.IndexIVFFlat(
                quantizer,
                self.config.EMBED_DIM,
                self.config.IVF_NLIST,
                self.config.FAISS_METRIC,
            )

            # Train IVF
            print("Generating embeddings for IVF training...")
            embeddings = Settings.embed_model.get_text_embedding_batch(
                [doc.get_content() for doc in documents], show_progress=True
            )
            embeddings_np = np.array(embeddings, dtype="float32")

            if embeddings_np.shape[0] >= self.config.IVF_NLIST:
                print("Training IVF index...")
                faiss_index.train(embeddings_np)
                print("IVF index trained.")
            else:
                print(
                    f"Warning: Not enough documents ({embeddings_np.shape[0]}) to train IVF with nlist={self.config.IVF_NLIST}."
                )

            vector_store = FaissVectorStore(faiss_index=faiss_index)
            # We pass documents (nodes) to from_documents, which will embed *again* to add
            return self._build_index(vector_store, documents)

        else:
            raise ValueError(f"Unknown index_type: {index_type}")

    @staticmethod
    def get_retriever_kwargs(index_type: str, config: Config) -> Dict[str, Any]:
        """Gets query-time args for the specified index."""
        if index_type == "hnsw":
            print(f"Using HNSW efSearch: {config.HNSW_EF_SEARCH}")
            return {"vector_store_query_args": {"hnsw_ef": config.HNSW_EF_SEARCH}}
        elif index_type == "ivf":
            print(f"Using IVF nprobe: {config.IVF_NPROBE}")
            return {"vector_store_query_args": {"nprobe": config.IVF_NPROBE}}
        return {}


# --- 4. Recall Measurement ---


def calculate_recall(
    retrieved_nodes: List[NodeWithScore], gold_sp_list: List[List[Any]]
) -> float:
    """Calculates paragraph-level recall."""
    if not gold_sp_list:
        return 1.0  # Technically 100% recall if no facts were required

    retrieved_facts = [
        [node.node.metadata.get("title"), node.node.metadata.get("sent_id")]
        for node in retrieved_nodes
    ]
    required_facts = [[title, sent_id] for title, sent_id in gold_sp_list]

    if not required_facts:
        return 1.0  # Also 100% recall if facts list was empty/invalid

    found_count = sum(
        1 for title_sent_id in required_facts if title_sent_id in retrieved_facts
    )
    return found_count / len(required_facts)


# --- 5. RAG Pipeline ---


async def run_pipeline(
    config: Config, index_type: str, index_data: List[Dict], query_data: List[Dict]
):
    """Runs the full RAG pipeline and evaluation for a given index type."""

    print(f"\n--- Running Pipeline for Index Type: {index_type.upper()} ---")

    # Setup Index
    persist_dir = f"./{index_type}_index_{len(query_data)}"
    factory = IndexFactory(config, persist_dir)
    # nodes = prepare_documents_chunked(
    #     index_data, config.CHUNK_SIZE, config.CHUNK_OVERLAP
    # )
    nodes = prepare_documents_sentence(index_data)
    index = factory.get_index(index_type, nodes)

    # Setup Query Engine with custom prompt
    retriever_kwargs = factory.get_retriever_kwargs(index_type, config)
    prompt_template = PromptTemplate(
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Be concise. Provide *only* the specific answer requested (e.g., just the name, date, location, 'yes', or 'no').\n"
        "Do not include explanations, introductory phrases, or conversational text.\n"
        "Query: {query_str}\n"
        "Answer: "
    )
    query_engine = index.as_query_engine(
        llm=Settings.llm,
        similarity_top_k=config.TOP_K,
        text_qa_template=prompt_template,
        **retriever_kwargs,
    )

    # Run queries
    print(f"Retrieving documents for {len(query_data)} questions...")
    total_recall = 0
    recall_found_at_least_1 = 0
    retrieval_tasks = []

    for item in tqdm(enumerate(query_data), total=len(query_data)):
        question_id = item.get("_id")
        question = item.get("question")
        gt_sp = item.get("supporting_facts")
        question_type = item.get("type", "unknown")

        task = f"task: search result | query: {question}"

        retrieved_nodes = query_engine.retrieve(question)
        retrieved_nodes.sort(key=lambda x: x.get_score(), reverse=True)

        recall = calculate_recall(retrieved_nodes, gt_sp)
        total_recall += recall
        recall_found_at_least_1 += 1 if recall >= 0.5 else 0

        retrieval_tasks.append(
            {
                "question_id": question_id,
                "question": question,
                "nodes": retrieved_nodes,
                "recall": recall,
                "question_type": question_type,
            }
        )

    # === ASYNC LLM ANSWERING ===

    # 1. Create a Semaphore to limit concurrency
    semaphore = asyncio.Semaphore(config.CONCURRENT_REQUESTS)

    # 2. Define a helper async function to wrap the query with the semaphore
    async def process_item(task_data):
        """Acquires semaphore, synthesizes, and releases."""
        question_id = task_data["question_id"]
        question = task_data["question"]
        nodes = task_data["nodes"]

        async with semaphore:
            try:
                # Run the actual LLM call
                response = await query_engine.asynthesize(question, nodes=nodes)
                generated_answer = response.response.strip()
                # Return all data needed for processing
                return question_id, generated_answer, task_data
            except Exception as e:
                print(f"  ERROR processing question {question_id}: {e}")
                # Return error and original data to avoid breaking the loop
                return question_id, "[ERROR]", task_data

    # 3. Create the list of tasks to run
    tasks_to_run = []
    for task_data in retrieval_tasks:
        tasks_to_run.append(process_item(task_data))

    print(
        f"\nRunning {len(tasks_to_run)} async LLM queries (Concurrency: {config.CONCURRENT_REQUESTS})..."
    )

    # 4. Run tasks with asyncio.as_completed and tqdm for monitoring
    processed_count = 0
    predictions = {"answer": {}, "sp": {}}

    for future in tqdm(asyncio.as_completed(tasks_to_run), total=len(tasks_to_run)):
        # Get results as they complete
        question_id, generated_answer, task_data = await future

        # print(f"\nQ:{task_data['question']}\nA: {generated_answer}")

        # Store the answer
        predictions["answer"][question_id] = generated_answer
        processed_count += 1

        predicted_sp = []
        if task_data["nodes"]:
            top_n = 4 if task_data["question_type"] == "bridge-comparison" else 2
            for node_with_score in task_data["nodes"][:top_n]:
                metadata = node_with_score.node.metadata
                title = metadata.get("title")
                sent_id = metadata.get("sent_id")

                sp_entry = [title, sent_id]
                if sp_entry not in predicted_sp:
                    predicted_sp.append(sp_entry)

        # Store predicted SP
        predictions["sp"][question_id] = predicted_sp

    # === SAVE RESULTS ===

    output_file = f"predictions_{index_type}_{len(query_data)}.json"
    print(
        f"\nFormatting and saving {len(predictions['answer'])} results to {output_file}..."
    )
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2, ensure_ascii=False)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    # Print summary
    avg_recall = (total_recall / processed_count) * 100 if processed_count > 0 else 0
    avg_recall_at_least_1 = (
        (recall_found_at_least_1 / processed_count) * 100 if processed_count > 0 else 0
    )
    print(f"\n--- Summary for {index_type.upper()} ---")
    print(f"Average Recall@{config.TOP_K}: {avg_recall:.3f}%")
    print(f"Questions with Recall@{config.TOP_K} >= 50%: {avg_recall_at_least_1:.3f}%")
    print(f"Written prediction results to: {output_file}")


# --- 6. Main Execution ---


def main():
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY not found in environment variables or .env file."
        )
    gemini_api_key = os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found in environment variables or .env file."
        )

    config = Config()

    # Setup global LlamaIndex settings
    # Settings.llm = OpenAI(model=config.API_MODEL_NAME, api_key=openai_api_key)
    Settings.llm = GoogleGenAI(model="gemini-2.5-flash", api_key=gemini_api_key)

    # Settings.embed_model = OpenAIEmbedding(
    #     model=config.EMBEDDING_MODEL_NAME, api_key=openai_api_key
    # )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # model_name="google/embeddinggemma-300m",
        device="cuda",
        embed_batch_size=256,
    )

    # Load data
    index_data, query_data = load_data(config.JSON_FILE_PATH, config.MAX_QUERIES)

    # --- Run for HNSW ---
    asyncio.run(run_pipeline(config, "hnsw", index_data, query_data))

    # --- Run for IVF ---
    # asyncio.run(run_pipeline(config, "ivf", index_data, query_data))


if __name__ == "__main__":
    main()
