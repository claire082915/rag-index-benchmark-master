import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
import asyncio
from tqdm import tqdm
from google import genai
from openai import OpenAI, AsyncOpenAI
import random

# --- Configuration ---


class Config:
    """Configuration class for the benchmark."""

    JSON_FILE_PATH = r"C:\Users\yalkh\Documents\Projects\rag-index-benchmark\data\hotpotqa\hotpot_dev_distractor_v1.json"
    MAX_QUERIES = 10  # Set to -1 to use all queries
    CONCURRENT_REQUESTS = 25


# --- 1. Data Loading ---


def load_data(
    json_path: str, max_queries: int
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Loads the dataset from JSON and returns the full data and the query subset."""

    with open(json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    print(f"Loaded {len(full_data)} examples from file.")

    if 0 < max_queries < len(full_data):
        query_subset = full_data[:max_queries]
    else:
        query_subset = full_data

    print(f"Using first {len(query_subset)} examples for queries.")

    return query_subset


def retrieve_documents(target_recall, gold_documents, distractor_documents):
    documents = []
    recall = 0
    num_required_facts = len(gold_documents)

    for i in range(num_required_facts):
        ground_truth = gold_documents[i]
        noise = random.choice(distractor_documents)
        picked = random.choices(
            [ground_truth, noise], weights=[target_recall, 1 - target_recall], k=1
        )
        if picked[0] == ground_truth:
            recall += 1

        documents.append(picked[0])

    return documents, recall / num_required_facts


async def run_pipeline(
    config: Config, query_data: List[Dict], client: OpenAI, target_recall
):
    """Runs the full RAG pipeline and evaluation for a given index type."""

    # Setup Query Engine with custom prompt
    prompt_template = (
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

    # Run queries
    retrieval_tasks = []
    total_recall = 0

    for item in query_data:
        question_id = item.get("_id")
        question = item.get("question")
        gold_titles = [gold[0] for gold in item.get("supporting_facts")]

        distractor_documents = []
        gold_documents = []
        for doc in item.get("context"):
            title = doc[0]
            context = doc[1]
            if title not in gold_titles:
                distractor_documents.append([title, context])
            else:
                gold_documents.append([title, context])

        retrieved_documents, recall = retrieve_documents(
            target_recall, gold_documents, distractor_documents
        )
        total_recall += recall

        retrieval_tasks.append(
            {
                "question_id": question_id,
                "question": question,
                "retrieved_documents": retrieved_documents,
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
        retrieved_documents: list = task_data["retrieved_documents"]
        retrieved_documents_str = "\n---\n".join(
            ["\n".join(doc[1]) for doc in retrieved_documents]
        )

        async with semaphore:
            try:
                # Run the actual LLM call
                prompt = prompt_template.format(
                    context_str=retrieved_documents_str, query_str=question
                )
                # response = await client.aio.models.generate_content(
                #     model="gemini-2.5-flash", contents=prompt
                # )
                # generated_answer = response.text
                response = await client.responses.create(
                    model="gpt-5-mini", input=prompt
                )
                generated_answer = response.output_text

                # Return all data needed for processing
                return question_id, generated_answer, task_data

            except Exception as e:
                print(f"  ERROR processing question {question_id}: {e}")
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
    predictions = {"answer": {}, "sp": {}, "evidence": {}}

    for future in tqdm(asyncio.as_completed(tasks_to_run), total=len(tasks_to_run)):
        # Get results as they complete
        question_id, generated_answer, task_data = await future

        # Store the answer
        predictions["answer"][question_id] = generated_answer
        processed_count += 1

    # === SAVE RESULTS ===

    output_file = f"predictions_{len(query_data)}_recall_{target_recall}.json"
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
    avg_recall = total_recall / processed_count

    print(f"\n--- Result Summary ---")
    print(f"Target Recall: {target_recall:.3f}")
    print(f"Average Recall: {avg_recall:.3f}")
    print(f"Written prediction results to: {output_file}")
    print("=== FINISHED ===")


# --- 6. Main Execution ---


def main():
    load_dotenv()
    config = Config()
    # llm = genai.Client()
    llm = AsyncOpenAI()
    query_data = load_data(config.JSON_FILE_PATH, config.MAX_QUERIES)

    target_recall_sweep = [1]
    for target_recall in target_recall_sweep:
        asyncio.run(run_pipeline(config, query_data, llm, target_recall))


if __name__ == "__main__":
    main()
