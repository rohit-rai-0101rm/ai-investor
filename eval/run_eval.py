"""
Quality/regression eval harness (Phase 3).

Runs the fixed question set in eval_set.py against the real /api/chat
endpoint (via FastAPI's TestClient - in-process, no server needed, but it's
the actual route code, not a reimplementation of it), scores each answer two
ways, and stores one row per question in Postgres so quality trend is
visible over time, not just pass/fail on the latest run.

Scoring:
  1. Keyword/refusal check (cheap, deterministic) - see _score_keywords().
  2. Groq-as-judge (see _judge_grounding()) - a second, independent opinion
     on whether the answer is actually grounded in what was retrieved, since
     keyword presence alone can't catch "right topic, fabricated number".

Usage:
    python -m eval.run_eval
Exit code 0 if the run meets MIN_PASS_RATE and has zero adversarial
failures; 1 otherwise - this is what Phase 4 will wire into CI to gate
deploys on.
"""
import json
import os
import subprocess
import time
import uuid

from fastapi.testclient import TestClient
from sqlalchemy import text

from app import app
from database.postgres_sql import get_engine
from eval.eval_set import EVAL_SET
from llm.azure_openai import get_openai_client
from langchain_huggingface import HuggingFaceEmbeddings
from vectorstore.azure_ai_search import ChromaVectorStore, Retriever

# A hallucinating small model rephrases refusals in a lot of different ways;
# this list is deliberately broad rather than exact, since a false "did not
# refuse" reading (missing a real refusal) is the failure mode that matters
# here - not missing a rephrasing is safer than wrongly flagging a refusal.
REFUSAL_PHRASES = [
    "does not contain", "insufficient information", "not contain enough",
    "no information", "cannot determine", "not available in the",
    "context does not", "does not provide", "not enough information",
    "unable to determine", "no data", "not specified in the",
    "unable to find", "unable to provide", "unable to answer",
    "unable to verify", "not present in", "don't have", "doesn't have",
    "not mentioned in", "not included in", "no mention of",
    "not able to find",
]

# Matches routes/chat.py's own retrieval settings - the judge needs to see
# the same context the live endpoint actually used, not different chunks,
# or it would be judging groundedness against the wrong evidence.
RETRIEVAL_TOP_K = 6
MAX_CHARS_PER_CHUNK = 1200

MIN_PASS_RATE = 0.8


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def _contains_any(text_lower: str, phrases: list[str]) -> bool:
    return any(phrase.lower() in text_lower for phrase in phrases)


def _score_keywords(item: dict, answer: str) -> bool:
    """
    Deterministic pass/fail, no LLM involved. Meaning of "pass" differs by
    question type - see each branch.
    """
    answer_lower = answer.lower()

    if item["type"] == "factual":
        # Every concept-group needs at least one matching phrasing present.
        return all(
            _contains_any(answer_lower, group)
            for group in item["concepts"]
        )

    if item["type"] == "adversarial":
        # Earlier version also required the answer avoid an item-specific
        # "forbidden" phrase list (e.g. banning the mention of "2021 annual
        # report"). That penalized the model for naturally referencing the
        # entity it was asked about while honestly refusing to answer -
        # "I don't have info about Tesla's 2021 annual report" is a correct,
        # non-fabricated refusal, not a violation. Refusal detection alone is
        # the right deterministic signal here; _judge_grounding() is what
        # actually catches a *confidently fabricated* answer, which needs
        # semantic understanding a substring ban can't provide.
        return _contains_any(answer_lower, REFUSAL_PHRASES)

    if item["type"] == "meta":
        # Regression test for the bug where meta questions got refused -
        # passing means it did NOT refuse.
        return not _contains_any(answer_lower, REFUSAL_PHRASES)

    raise ValueError(f"Unknown eval question type: {item['type']!r}")


def _judge_grounding(question: str, context: str, answer: str) -> tuple[bool, str]:
    """
    Ask Groq itself whether the answer is grounded in the retrieved context
    (or an honest admission that the context is insufficient) - a second
    opinion that can catch fabrication a keyword check would miss, e.g. the
    right topic with a wrong/invented number attached.
    """
    prompt = (
        "You are a strict fact-checker for a RAG system. Given the context "
        "that was retrieved and the answer that was generated, judge whether "
        "the ANSWER is grounded - do not judge whether the context happens "
        "to contain the requested information.\n\n"
        "Mark grounded=true if EITHER:\n"
        "(a) every factual claim in the answer is directly supported by the "
        "context, OR\n"
        "(b) the answer honestly says the context does not contain enough "
        "information to answer - this is a CORRECT, grounded response even "
        "though the context lacks the requested fact. A model that admits "
        "it doesn't know is behaving correctly, not failing.\n\n"
        "Mark grounded=false ONLY if the answer confidently asserts specific "
        "facts, numbers, dates, or citations that are NOT present in the "
        "context - i.e. it fabricated something instead of admitting the "
        "gap.\n\n"
        f"Context:\n{context if context else '(no context retrieved)'}\n\n"
        f"Question: {question}\n\n"
        f"Answer: {answer}\n\n"
        "Respond with JSON only: {\"grounded\": true or false, \"reason\": "
        "\"<one sentence>\"}"
    )

    client = get_openai_client()
    model = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-8b-instant")

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0
    )

    try:
        parsed = json.loads(response.choices[0].message.content)
        return bool(parsed.get("grounded", False)), str(parsed.get("reason", ""))
    except Exception as e:
        # A judge that fails to respond in the right shape shouldn't silently
        # pass the answer - treat it as ungrounded and say why.
        return False, f"judge response could not be parsed: {e}"


def _retrieve_context(retriever: Retriever, question: str, company: str | None, year: int | None) -> str:
    if company and year:
        docs = retriever.invoke(query=question, company=company, year=year, top_k=RETRIEVAL_TOP_K)
    else:
        docs = retriever.invoke(query=question, top_k=RETRIEVAL_TOP_K)
    return "\n\n".join(doc.page_content[:MAX_CHARS_PER_CHUNK] for doc in docs)


def _store_result(engine, run_id: str, git_sha: str, item: dict, answer: str,
                   keyword_pass: bool, grounded: bool | None, judge_reason: str, passed: bool) -> None:
    query = text("""
        INSERT INTO eval_results (
            run_id, git_sha, question, company, year, question_type,
            answer, keyword_pass, grounded, judge_reason, passed
        ) VALUES (
            :run_id, :git_sha, :question, :company, :year, :question_type,
            :answer, :keyword_pass, :grounded, :judge_reason, :passed
        )
    """)
    with engine.begin() as connection:
        connection.execute(query, {
            "run_id": run_id,
            "git_sha": git_sha,
            "question": item["question"],
            "company": item.get("company"),
            "year": str(item["year"]) if item.get("year") is not None else None,
            "question_type": item["type"],
            "answer": answer,
            "keyword_pass": keyword_pass,
            "grounded": grounded,
            "judge_reason": judge_reason,
            "passed": passed,
        })


def run_eval() -> bool:
    run_id = str(uuid.uuid4())
    git_sha = _git_sha()
    engine = get_engine()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = ChromaVectorStore(
        persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "investor-intelligence")
    )
    retriever = Retriever(vector_store.collection, embeddings)

    results = []
    adversarial_failures = 0

    # `with` triggers app.py's startup event (DB/table/collection setup) the
    # same way a real deployment would - a bare TestClient(app) call skips
    # FastAPI's lifespan handling entirely.
    with TestClient(app) as client:
        for item in EVAL_SET:
            response = client.post("/api/chat", json={
                "question": item["question"],
                "company": item.get("company"),
                "year": item.get("year"),
            })
            answer = response.json().get("answer", "") if response.status_code == 200 else f"[HTTP {response.status_code}]"

            keyword_pass = _score_keywords(item, answer)

            grounded, judge_reason = (None, "")
            if item["type"] != "meta":
                context = _retrieve_context(retriever, item["question"], item.get("company"), item.get("year"))
                grounded, judge_reason = _judge_grounding(item["question"], context, answer)

            passed = keyword_pass if grounded is None else (keyword_pass and grounded)

            if item["type"] == "adversarial" and not passed:
                adversarial_failures += 1

            _store_result(engine, run_id, git_sha, item, answer, keyword_pass, grounded, judge_reason, passed)

            results.append({
                "question": item["question"],
                "type": item["type"],
                "passed": passed,
                "keyword_pass": keyword_pass,
                "grounded": grounded,
                "judge_reason": judge_reason,
            })

            status = "PASS" if passed else "FAIL"
            print(f"[{status}] ({item['type']}) {item['question']}")
            if not passed:
                print(f"         answer: {answer[:200]}")
                print(f"         keyword_pass={keyword_pass} grounded={grounded} reason={judge_reason}")

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    pass_rate = passed_count / total if total else 0.0

    print(f"\n{passed_count}/{total} passed ({pass_rate:.0%}). "
          f"Adversarial failures: {adversarial_failures}. run_id={run_id} git_sha={git_sha}")

    return pass_rate >= MIN_PASS_RATE and adversarial_failures == 0


if __name__ == "__main__":
    import sys
    ok = run_eval()
    sys.exit(0 if ok else 1)
