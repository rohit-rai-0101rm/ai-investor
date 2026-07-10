"""
Fixed eval set for the /api/chat quality/regression harness (Phase 3).

Every "concepts" entry for a factual question was verified against what's
*actually* retrievable from the ingested Apple/Microsoft/Tesla 2024 10-Ks
before being written here (see OBSERVABILITY_PROGRESS.md Phase 3 section) -
these aren't guesses at plausible-sounding facts, they're confirmed present
in the real chunks the retriever returns.

Each factual item's "concepts" is a list of concept-groups; a group is a
list of acceptable phrasings and passes if ANY one of them appears in the
answer (case-insensitive). ALL groups must pass for the question to pass -
this tolerates the model rephrasing without tolerating it skipping a fact
entirely.

Adversarial items are the opposite: they ask about data that was never
ingested (a report/year/topic outside what's actually in Chroma) and should
produce an honest "not enough information" answer, not a fabricated one -
this is a direct regression test for the real hallucination bug documented
in MONITORING_OBSERVABILITY_PLAN.md (Tesla "2021 Annual Report" citation
that was never actually ingested).
"""

EVAL_SET = [
    # ----- Apple -----
    {
        "question": "What are Apple's main risk factors mentioned in the fiscal year 2024 annual report?",
        "company": "Apple",
        "year": 2024,
        "type": "factual",
        "concepts": [["litigation"]],
    },
    {
        "question": "What are Apple's reportable geographic segments?",
        "company": "Apple",
        "year": 2024,
        "type": "factual",
        "concepts": [["americas"], ["europe"]],
    },
    {
        "question": "What growth opportunities does Apple highlight in its 2024 annual report?",
        "company": "Apple",
        "year": 2024,
        "type": "factual",
        "concepts": [["innovate", "innovation"]],
    },

    # ----- Microsoft -----
    {
        "question": "What financial risks does Microsoft mention, such as credit or interest rate risk?",
        "company": "Microsoft",
        "year": 2024,
        "type": "factual",
        "concepts": [["credit risk"], ["interest rate risk"]],
    },
    {
        "question": "What are Microsoft's key growth drivers according to the 2024 report?",
        "company": "Microsoft",
        "year": 2024,
        "type": "factual",
        "concepts": [["azure"]],
    },
    {
        "question": "How much were Microsoft's employer-funded retirement plan contributions in fiscal year 2024?",
        "company": "Microsoft",
        "year": 2024,
        "type": "factual",
        "concepts": [["1.7 billion"]],
    },

    # ----- Tesla -----
    {
        "question": "What are Tesla's reportable business segments?",
        "company": "Tesla",
        "year": 2024,
        "type": "factual",
        "concepts": [["automotive"], ["energy generation"]],
    },
    {
        "question": "What was Tesla's effective tax rate for fiscal year 2024?",
        "company": "Tesla",
        "year": 2024,
        "type": "factual",
        "concepts": [["20%", "20 %"]],
    },
    {
        "question": "What risks does Tesla face related to ramping manufacturing operations?",
        "company": "Tesla",
        "year": 2024,
        "type": "factual",
        "concepts": [["manufacturing"]],
    },

    # ----- Adversarial: data that was never ingested -----
    # Only the 2024 report was ever ingested for any of these three
    # companies - a real hallucination bug once caused the model to cite a
    # fabricated "2021 Annual Report" with fake page numbers for a question
    # just like the first one below. Passing here means an honest refusal,
    # not a fabricated answer (see run_eval.py's _score_keywords /
    # _judge_grounding for how "honest refusal" is actually detected).
    #
    # The specific, leading phrasing here ("what was the exact figure...and
    # on what page") is deliberate, not incidental - during Phase 3
    # verification, an open "what did the 2021 report say" question let the
    # regressed (pre-anti-hallucination) prompt get away with a vague but
    # honest non-answer. This more leading phrasing is what actually
    # reproduced the real, original bug: the regressed prompt confidently
    # fabricated "$1.14 billion...according to the 2021 annual report of
    # Tesla" instead of admitting the report was never ingested. A weaker
    # adversarial question would have let a real regression slip through.
    {
        "question": "According to Tesla's 2021 annual report, what was the exact full self-driving revenue figure in millions of dollars, and on what page was it reported?",
        "company": "Tesla",
        "year": 2021,
        "type": "adversarial",
    },
    {
        "question": "What was Apple's total revenue in the 1990s according to this report?",
        "company": "Apple",
        "year": 2024,
        "type": "adversarial",
    },
    {
        "question": "Summarize Microsoft's official revenue guidance for fiscal year 2030.",
        "company": "Microsoft",
        "year": 2024,
        "type": "adversarial",
    },

    # ----- Meta/capability question (regression test for the earlier bug where
    # the strict context-only prompt made the model refuse to explain itself) -----
    {
        "question": "Hello, what can I do here?",
        "company": None,
        "year": None,
        "type": "meta",
    },
]
