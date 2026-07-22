"""
The RAGAS evaluation harness needs 4 elements:
    - question
    - response: pipeline's answer
    - retrieved context
    - reference context
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from openai import OpenAI
from ragas import EvaluationDataset, evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall

load_dotenv()

EVAL_PROVIDER = os.getenv("EVAL_PROVIDER", "local")
if EVAL_PROVIDER == "openai":
    BASE_URL, API_KEY, MODEL = None, os.getenv("OPENAI_API_KEY"), "gpt-4o-mini"
else:
    BASE_URL, API_KEY, MODEL = "http://localhost:11434/v1", "ollama", "qwen3:8b"

CLIENT = OpenAI(base_url=BASE_URL, api_key=API_KEY)
JUDGE = LangchainLLMWrapper(ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL))
GEN = CLIENT

BASE_DIR = Path(__file__).parent
EVAL_DIR = BASE_DIR / "evals"
GOLDEN_EVAL = EVAL_DIR / "golden.jsonl"
DATA = BASE_DIR / "data.jsonl"


CORPUS = [
    json.loads(line)["text"] for line in DATA.read_text().splitlines() if line.strip()
]
print((CORPUS)) 


# The judge is part of the ruler — pin model + temperature.
JUDGE = LangchainLLMWrapper(ChatOpenAI(base_url=BASE_URL, api_key=API_KEY, model=MODEL))


def retrieve(question: str, corpus: list[str], top_k: int = 3) -> list[str]:
    """
    Basic 'retrieve' function for Retrieval-Augmented Generation (RAG).

    This baseline retriever ignores the question and simply pulls passages from the corpus.
    "Structured" can mean either "is it an ordered list?" or "is it a dict or some other data structure?"—
    here, we just use a list, in input order.

    Args:
        question (str): The user’s question (ignored for now).
        corpus (list[str]): The collection of documents or passages.
        top_k (int): Number of passages to return.

    Returns:
        list[str]: A list of retrieved passages, in the order they appear in the corpus.
    """
    scored = sorted(
        corpus,
        key=lambda d: len(set(question.lower().split()) & set(d.lower().split())),
        reverse=True,
        # The [:top_k] slices out the top N passages after sorting,
        # where N is given by top_k (default: 3).
        # Passages are ranked by the number of words they have in common with the question.
        # That count is computed by the len() function:
        #   len(set(question words) & set(passage words))
        # So, a higher len means more overlapping words.
    )[:top_k]
    return scored


def rag_pipeline(question: str) -> tuple[str, list[str]]:
    ctx = retrieve(question, CORPUS)
    joined = "\n".join(ctx)
    resp = GEN.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {
                "role": "user",
                "content": (
                    "Answer ONLY from the context below. "
                    "If the answer is not there, reply exactly: NOT_IN_CORPUS\n\n"
                    f"Context:\n{joined}\n\nQuestion: {question}"
                ),
            }
        ],
    )
    return resp.choices[0].message.content or "", ctx


def build_dataset(path: str | Path) -> EvaluationDataset:
    """
    Loads an evaluation set (one example per line, with question and ground_truth)
    and runs the RAG pipeline on each input. Packages everything into an EvaluationDataset
    object in the 4-field structure that RAGAS expects:
      - question (the user input)
      - response (the model's answer)
      - contexts (retrieved context passages)
      - ground_truths (the reference answers)
    """
    rows = []
    for line in Path(path).read_text().splitlines():
        ex = json.loads(line)
        question = ex["question"]
        reference = ex["ground_truth"]
        answer, retrieved_contexts = rag_pipeline(question)
        row = {
            "user_input": question,
            "response": answer,
            "retrieved_contexts": retrieved_contexts,
            "reference": reference,
        }
        rows.append(row)
    return EvaluationDataset.from_list(rows)


def run_eval(path: str | Path = GOLDEN_EVAL) -> dict[str, float]:
    """
    Runs the end-to-end evaluation using the RAGAS framework.
    Returns a dict mapping metric names to their scores.
    """
    result = evaluate(
        dataset=build_dataset(path),
        metrics=[Faithfulness(llm=JUDGE), LLMContextRecall(llm=JUDGE)],
    )
    return {k: float(v) for k, v in result._repr_dict.items()}  # noqa: SLF001


if __name__ == "__main__":
    # This block allows the script to be run from the command line.
    # - If a filename is passed as the first argument (after the script name), that file is used as the eval set.
    # - Otherwise, it uses the default GOLDEN_EVAL dataset.
    # It then runs the evaluation and prints the resulting metrics.
    eval_path = Path(sys.argv[1]) if len(sys.argv) > 1 else GOLDEN_EVAL
    print(run_eval(eval_path))
