"""
HotpotQA Recall Benchmark — Gemini version
Adapted from Yousif's 2wiki_recall_benchmark.py

Goal: Sweep target_recall from 0 → 1 and measure how EM/F1 change.
This simulates what happens to RAG quality as vector DB recall drops.

Setup:
    1. Get a free Gemini API key at https://aistudio.google.com → "Get API key"
    2. Add to your .env file:  GOOGLE_API_KEY=your-key-here
    3. pip install google-genai python-dotenv tqdm
    4. Set HOTPOT_JSON_FILE_PATH below to your local hotpot_dev_distractor_v1.json
    5. Run: python hotpot_recall_benchmark.py

Evaluate each output file with:
    python hotpot_evaluate_v1.py hotpot_predictions_1000_recall_1.000.json hotpot_dev_distractor_v1.json 1000
"""

import os
import json
import asyncio
import random
from typing import List, Dict, Any

from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types

# --- Configuration ---


class Config:
    """Configuration class for the benchmark."""

    # !! UPDATE THIS PATH to your local hotpot_dev_distractor_v1.json !!
    HOTPOT_JSON_FILE_PATH = r"../data/hotpotqa/hotpot_dev_distractor_v1.json"

    MAX_QUERIES = 1000  # Match Yousif's 2wiki run; set to -1 for all ~7405 examples

    CONCURRENT_REQUESTS = 5    # Gemini free tier: 15 RPM, keep this conservative

    LLM_MODEL = "gemini-2.5-flash"   # Fast, free tier, good quality

    # Recall values to sweep — mirrors the 2wiki experiment data points
    # TARGET_RECALL_SWEEP = [0.0, 0.2, 0.4, 0.6, 0.8, 0.816, 0.839, 0.874, 0.897,
    #                        0.928, 0.943, 0.979, 1.0]
    TARGET_RECALL_SWEEP = [0.8, 0.9, 0.95, 1.0]



# --- 1. Data Loading ---


def load_data(json_path: str, max_queries: int) -> List[Dict[str, Any]]:
    """Loads the HotpotQA dataset and returns the query subset."""
    with open(json_path, "r", encoding="utf-8") as f:
        full_data = json.load(f)

    print(f"Loaded {len(full_data)} examples from file.")

    if 0 < max_queries < len(full_data):
        query_subset = full_data[:max_queries]
    else:
        query_subset = full_data

    print(f"Using first {len(query_subset)} examples for queries.")
    return query_subset


# --- 2. Controlled Recall Retrieval ---


def retrieve_documents(target_recall: float, gold_documents: list, distractor_documents: list):
    """
    Simulates retrieval at a given recall level.
    For each required gold document, picks it with probability=target_recall,
    otherwise substitutes a random distractor.

    Returns: (retrieved_documents, actual_recall)
    """
    documents = []
    recall_hits = 0
    num_required = len(gold_documents)

    for gold_doc in gold_documents:
        noise = random.choice(distractor_documents)
        picked = random.choices(
            [gold_doc, noise],
            weights=[target_recall, 1 - target_recall],
            k=1
        )[0]

        if picked == gold_doc:
            recall_hits += 1

        documents.append(picked)

    actual_recall = recall_hits / num_required if num_required > 0 else 1.0
    return documents, actual_recall


# --- 3. RAG Pipeline ---


PROMPT_TEMPLATE = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Be concise. Provide *only* the specific answer requested "
    "(e.g., just the name, date, location, 'yes', or 'no').\n"
    "Do not include explanations, introductory phrases, or conversational text.\n"
    "Query: {query_str}\n"
    "Answer: "
)


async def run_pipeline(
    config: Config,
    query_data: List[Dict],
    client: genai.Client,
    target_recall: float,
):
    """Runs the full RAG pipeline for a single target_recall value."""

    print(f"\n{'='*60}")
    print(f"Running pipeline | target_recall={target_recall:.3f} | N={len(query_data)}")
    print(f"{'='*60}")

    # --- Build retrieval tasks ---
    retrieval_tasks = []
    total_recall = 0.0

    for item in query_data:
        question_id = item["_id"]
        question = item["question"]
        gold_titles = {title for title, _ in item.get("supporting_facts", [])}

        gold_documents = []
        distractor_documents = []

        for doc in item.get("context", []):
            title, sentences = doc[0], doc[1]
            if title in gold_titles:
                gold_documents.append([title, sentences])
            else:
                distractor_documents.append([title, sentences])

        # Edge case: if no distractors, use gold docs as distractors too
        if not distractor_documents:
            distractor_documents = gold_documents

        retrieved_documents, actual_recall = retrieve_documents(
            target_recall, gold_documents, distractor_documents
        )
        total_recall += actual_recall

        retrieval_tasks.append({
            "question_id": question_id,
            "question": question,
            "retrieved_documents": retrieved_documents,
        })

    # --- Async LLM answering ---
    semaphore = asyncio.Semaphore(config.CONCURRENT_REQUESTS)

    async def process_item(task_data: dict):
        question_id = task_data["question_id"]
        question = task_data["question"]
        docs = task_data["retrieved_documents"]

        # Format context: join all sentences of each retrieved doc
        context_str = "\n---\n".join(
            "\n".join(sentences) for _, sentences in docs
        )

        prompt = PROMPT_TEMPLATE.format(context_str=context_str, query_str=question)

        async with semaphore:
            try:
                response = await client.aio.models.generate_content(
                    model=config.LLM_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=256,
                        temperature=0,
                    ),
                )
                # print(f"DEBUG {question_id}: finish_reason={response.candidates[0].finish_reason if response.candidates else 'NO_CANDIDATES'}, text={response.text!r}")
                generated_answer = (response.text or "").strip() or "[NO_ANSWER]"
            except Exception as e:
                print(f"  ERROR on {question_id}: {e}")
                generated_answer = "[ERROR]"

        return question_id, generated_answer

    tasks_to_run = [process_item(t) for t in retrieval_tasks]
    print(f"Running {len(tasks_to_run)} async LLM queries (concurrency={config.CONCURRENT_REQUESTS})...")

    predictions = {"answer": {}, "sp": {}}
    processed_count = 0

    for future in tqdm(asyncio.as_completed(tasks_to_run), total=len(tasks_to_run)):
        question_id, generated_answer = await future
        predictions["answer"][question_id] = generated_answer
        # HotpotQA eval also needs 'sp' — leave empty; focus is on answer metrics
        predictions["sp"][question_id] = []
        processed_count += 1

    # --- Save ---
    output_file = f"hotpot_predictions_{len(query_data)}_recall_{target_recall:.3f}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    avg_recall = total_recall / processed_count if processed_count > 0 else 0.0
    print(f"\n--- Done | target_recall={target_recall:.3f} ---")
    print(f"Average actual recall : {avg_recall:.4f}")
    print(f"Predictions saved to  : {output_file}")
    print(f"\nTo evaluate, run:")
    print(f"  python hotpot_evaluate_v1.py {output_file} hotpot_dev_distractor_v1.json {len(query_data)}")

    return output_file, avg_recall


# --- 4. Main ---


def main():
    load_dotenv()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Get a free key at https://aistudio.google.com "
            "and add it to your .env file as: GOOGLE_API_KEY=your-key-here"
        )

    config = Config()
    client = genai.Client(api_key=api_key)

    query_data = load_data(config.HOTPOT_JSON_FILE_PATH, config.MAX_QUERIES)

    results_summary = []

    for target_recall in config.TARGET_RECALL_SWEEP:
        output_file, avg_recall = asyncio.run(
            run_pipeline(config, query_data, client, target_recall)
        )
        results_summary.append({
            "target_recall": target_recall,
            "avg_actual_recall": avg_recall,
            "output_file": output_file,
        })

    print("\n" + "="*60)
    print("SWEEP COMPLETE — evaluate all files with:")
    for r in results_summary:
        n = config.MAX_QUERIES if config.MAX_QUERIES > 0 else len(query_data)
        print(f"  python hotpot_evaluate_v1.py {r['output_file']} {config.HOTPOT_JSON_FILE_PATH} {n}")
    print("="*60)


if __name__ == "__main__":
    main()