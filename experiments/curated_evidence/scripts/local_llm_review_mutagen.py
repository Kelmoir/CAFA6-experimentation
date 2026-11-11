"""Item 15, tier 1: judge each MUTAGEN review task (built by build_mutagen_review_tasks.py)
against a local LLM, via LM Studio's OpenAI-compatible server -- same approach and the same
reused HTTP/retry machinery as experiments/mf_span/scripts/local_llm_review_loop.py, which the
mf_span project already proved works well for this kind of free-text-vs-GO-term judgment call.

Why an LLM and not a bigger dictionary: the plain token-overlap matcher in mine_mutagen_evidence.py
can't tell a negated clause ("does not abolish X") from a real loss, and can't bridge a synonym/
paraphrase the note uses instead of a GO term's literal name -- both need real text understanding,
not more keywords. The model is never asked to recall what a GO term means from its own training:
every task hands it the protein's own real molecular-function terms with their official
go-basic.obo name + definition already in context (see build_mutagen_review_tasks.py), so this is
closed-set matching against text that's fully present in the prompt, not open recall.

Run this ON THE MACHINE THAT HOSTS LM STUDIO (or anywhere with LAN access to it), NOT inside the
Claude Code sandbox -- same reason as local_llm_review_loop.py: the sandbox's network namespace has
no route to the host's LAN.

Usage:
  python3 experiments/curated_evidence/scripts/local_llm_review_mutagen.py \
      --host http://192.168.1.122:1234 --concurrency 4

Resumable: results append to mutagen_review_results.jsonl, one line per completed task, keyed by
task id. Re-running only re-queries tasks that are missing or whose last recorded verdict was
UNPARSED/INVALID (a genuine NONE or a real GO id is never re-queried) -- safe to Ctrl-C any time.

Whenever a response doesn't parse to a clean VERDICT, the raw model text is saved to
mutagen_review_debug.log (not just a canned reason string) -- the first smoke test against a
native reasoning model showed exactly why this matters: its own internal <think>-style
deliberation ran past max_tokens before ever reaching VERDICT, and without the raw text saved
there was no way to tell that apart from a genuine formatting failure without digging the
transcript out of LM Studio's own UI by hand.
"""
import argparse
import concurrent.futures
import json
import re
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "mf_span/scripts"))
from local_llm_review_loop import LocalLLMError, call_local_llm, discover_model  # noqa: E402

TASKS_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_tasks.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_results.jsonl"
DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_debug.log"

TASK_DESCRIPTION = (
    "You are reviewing a site-directed mutagenesis (MUTAGEN) annotation from UniProt, to decide "
    "whether it provides trustworthy evidence that a SPECIFIC molecular function is lost in the "
    "mutant protein. You will be shown the mutation and UniProt's free-text description of its "
    "effect, plus a list of molecular-function GO terms this protein is already known to carry "
    "(each with its official name and definition)."
)

INSTRUCTIONS = (
    "Decide which ONE of the listed terms (if any) is described as being clearly and "
    "unconditionally LOST or ABOLISHED by this specific mutation.\n"
    "- If the description negates the loss (e.g. \"no loss of X\", \"no significant loss of X\", "
    "\"does not abolish Y\", \"no effect on Z activity\", \"normal X\"), that term is NOT lost -- "
    "do not select it. Read the whole clause: \"no\"/\"not\"/\"normal\" immediately changes the "
    "meaning of the loss-language that follows it.\n"
    "- If the loss is described as depending on OTHER mutations elsewhere in the protein (e.g. "
    "\"when associated with...\", \"in combination with...\"), select the term but set CONDITIONAL "
    "to YES -- this single mutation alone may not be sufficient evidence on its own.\n"
    "- The note may use different wording than a term's official name (a synonym, abbreviation, "
    "or more specific/general phrasing) -- still match it if the underlying biological activity is "
    "clearly the same.\n"
    "- A description that only names a SPECIFIC interaction partner or process (e.g. \"loss of "
    "interaction with MPDZ\") does NOT by itself confirm any listed term unless that term's own "
    "definition is clearly about that same kind of interaction -- a loss affecting one specific, "
    "unlisted partner/process does not transfer onto an unrelated listed term just because both "
    "involve this protein.\n"
    "- A PARTIAL loss, reduction, or decrease (e.g. \"partial loss of X\", \"reduced X\", \"X "
    "activity decreased\") is NOT a clear, unconditional loss, even though the words \"loss of\" "
    "appear -- do not select a term for it. Only a complete or near-complete loss counts (e.g. "
    "\"loss of X\", \"abolished\", \"no detectable X\", or a tiny residual percent of wild-type).\n"
    "- If the description only names a GENERIC class of activity (e.g. \"catalytic activity\", "
    "\"enzymatic activity\", \"function\") without saying which SPECIFIC chemistry, and more than "
    "one listed term could plausibly fall under that generic class, you cannot tell which one is "
    "meant -- answer NONE rather than guessing which specific one.\n"
    "- That said, when exactly ONE candidate term remains plausible and it is a well-established, "
    "specific instance of the general activity named (e.g. a phosphatase IS a kind of hydrolase; a "
    "kinase IS a kind of transferase -- basic, uncontroversial biochemistry, not a guess), that "
    "counts as a confident match. Being conservative does not mean rejecting a clear is-a "
    "relationship just because the note used the broader word.\n"
    "- A compound note may describe SEVERAL separate findings. Judge each clause on its own: a "
    "clause using ongoing/unchanged-behavior language (\"inhibits X\", \"promotes Y\", \"maintains "
    "Z\", stated without comparison to the wild-type) is NOT itself evidence of a loss, even if it "
    "immediately follows an unrelated negated clause -- it does not inherit loss-status from a "
    "neighboring clause about a different finding, and does not gain it either.\n"
    "- Most tasks genuinely have NO correct match among the listed terms -- answering NONE is the "
    "expected, common outcome, not a failure to find something. Never force a weak or speculative "
    "match just to avoid answering NONE."
)

ANSWER_FORMAT = (
    "Respond with EXACTLY three lines, nothing else, no markdown, no commentary:\n"
    "VERDICT: <GO id exactly as listed above, or NONE>\n"
    "CONDITIONAL: YES or NO (NO if VERDICT is NONE)\n"
    "REASONING: <one sentence, plain text>"
)

THINKING_INSTRUCTIONS = (
    "In the THINKING block, follow these steps in order, and do not repeat a step once it is done:\n"
    "- Split the MUTAGEN description into its separate clauses/findings, and restate each in your "
    "own words -- a compound note often reports several unrelated things. For each clause, mark it "
    "as: NEGATED (\"no loss\", \"no significant loss\", \"no effect\", \"does not abolish\", "
    "\"normal\"), PARTIAL (\"partial loss\", \"reduced\", \"decreased\" -- not complete), ONGOING/"
    "UNCHANGED (describes current behavior with no comparison to wild-type, e.g. \"inhibits X\", "
    "\"promotes Y\"), or a GENUINE COMPLETE LOSS. If every clause is NEGATED, PARTIAL, or ONGOING/"
    "UNCHANGED -- none is a genuine complete loss -- stop here and go straight to VERDICT: NONE.\n"
    "- Otherwise, using only the genuine-complete-loss clause(s), for each listed GO term in turn, "
    "EXACTLY ONCE: state whether that clause's own wording -- or a very well-known, unambiguous "
    "fact, not a guess -- connects it to that term's specific definition, and rate your confidence 1 "
    "(no real connection) to 10 (specific/mechanistic match). A clause naming only a specific "
    "interaction partner or process, without itself stating a general class of activity, does not "
    "confirm a listed term just because both involve the same protein -- rate it low unless you are "
    "genuinely confident the named partner/process matches that term's own definition. Conversely, "
    "if the clause names a GENERIC activity class (e.g. \"catalytic activity\") and only one listed "
    "term is a well-established, specific instance of it, that single exact match counts as high "
    "confidence, not a guess -- being conservative does not mean rejecting a sound is-a "
    "relationship.\n"
    "- If the clause names a generic activity class and MORE THAN ONE listed term could plausibly "
    "be the specific one meant, rate all of them low -- you cannot tell which one, so this should "
    "still end in NONE.\n"
    "- Once every term has been rated exactly once, stop analyzing. Use the highest-scoring term if "
    "its score is >=7; otherwise answer NONE. Do not revisit or re-score a term already rated, and "
    "do not force an answer just to avoid NONE -- most tasks genuinely have no correct match."
)

COT_ANSWER_FORMAT = (
    "Respond in this order, starting with THINKING (nothing before it):\n"
    "THINKING: <follow the numbered steps given above. Can span as many lines as needed; there is "
    "no length limit, but each term is rated once, not revisited.>\n"
    "VERDICT: <GO id exactly as listed above, or NONE>\n"
    "CONDITIONAL: YES or NO (NO if VERDICT is NONE)\n"
    "REASONING: <one sentence, plain text, summarizing the conclusion your THINKING reached>"
)


def build_system_prompt(chain_of_thought: bool) -> str:
    if chain_of_thought:
        return f"{TASK_DESCRIPTION}\n\n{INSTRUCTIONS}\n\n{THINKING_INSTRUCTIONS}\n\n{COT_ANSWER_FORMAT}"
    return f"{TASK_DESCRIPTION}\n\n{INSTRUCTIONS}\n\n{ANSWER_FORMAT}"


def build_prompt(task: dict) -> str:
    lines = [
        f"Protein {task['accession']}, mutation at residues {task['start'] + 1}-{task['end']} "
        f"(1-indexed):",
        "",
        f'MUTAGEN description: "{task["note"]}"',
        "",
        "This protein's known molecular-function GO terms:",
    ]
    for c in task["candidates"]:
        lines.append(f"- {c['go_id']}: {c['name']}")
        lines.append(f"  Definition: {c['def'] or '(no definition available)'}")
    return "\n".join(lines)


def _last_match(pattern, text, flags=0):
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1] if matches else None


def parse_response(text: str, valid_go_ids: set):
    v = _last_match(r"VERDICT:\s*(\S+)", text, re.I)
    c = _last_match(r"CONDITIONAL:\s*(YES|NO)", text, re.I)
    r = _last_match(r"REASONING:\s*(.+)", text, re.I)
    raw_verdict = v.group(1).rstrip(".,:;") if v else None
    if raw_verdict is None:
        verdict = "UNPARSED"
    elif raw_verdict.upper() == "NONE":
        verdict = "NONE"
    elif raw_verdict.upper() in valid_go_ids:
        verdict = raw_verdict.upper()
    else:
        verdict = "INVALID_VERDICT"  # hallucinated an id not in this task's candidate list
    conditional = (c.group(1).upper() == "YES") if c else False
    reasoning = r.group(1).strip().split("\n")[0][:400] if r else "(no reasoning extracted)"
    return verdict, conditional, reasoning


_write_lock = threading.Lock()
_debug_lock = threading.Lock()


def log_debug(task_id, accession, tag, raw_text):
    """Saves the raw model response whenever parsing didn't produce a clean, usable verdict --
    without this, diagnosing a failure (e.g. a native reasoning model burning its whole token
    budget on internal deliberation before ever reaching VERDICT) means digging the transcript
    out of LM Studio's own UI by hand, which is what happened before this existed."""
    with _debug_lock:
        with open(DEBUG_LOG_PATH, "a") as f:
            f.write(f"=== task {task_id} ({accession}) [{tag}] ===\n{raw_text}\n\n")


def load_done(results_path: Path) -> dict:
    """task id -> last recorded result. UNPARSED/INVALID_VERDICT results are kept in the file for
    visibility but intentionally excluded from the returned dict's "done" sense via the caller
    checking the verdict -- see main()."""
    done = {}
    if results_path.exists():
        with open(results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
    return done


def review_task(host, model, system_prompt, task, request_timeout, max_tokens,
                 capacity_retries, capacity_retry_delay):
    valid_go_ids = {c["go_id"] for c in task["candidates"]}
    prompt = build_prompt(task)
    try:
        raw, finish_reason = call_local_llm(
            host, model, system_prompt, prompt, max_tokens, timeout=request_timeout,
            capacity_retries=capacity_retries, capacity_retry_delay=capacity_retry_delay)
    except LocalLLMError as e:
        log_debug(task["id"], task["accession"], f"HTTP {e.status} {e.reason}", e.body)
        return {"id": task["id"], "verdict": "UNPARSED", "conditional": False,
                "reasoning": f"HTTP {e.status} {e.reason}"}
    except (OSError, TimeoutError) as e:
        return {"id": task["id"], "verdict": "UNPARSED", "conditional": False,
                "reasoning": f"request failed: {e}"}
    verdict, conditional, reasoning = parse_response(raw, valid_go_ids)
    if verdict in ("UNPARSED", "INVALID_VERDICT"):
        if verdict == "UNPARSED" and finish_reason == "length":
            reasoning = f"hit max_tokens ({max_tokens}) before VERDICT appeared"
        log_debug(task["id"], task["accession"], f"{verdict} finish_reason={finish_reason}", raw)
    return {"id": task["id"], "accession": task["accession"], "verdict": verdict,
            "conditional": conditional, "reasoning": reasoning}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="LM Studio server, e.g. http://192.168.1.122:1234")
    ap.add_argument("--model", default=None, help="Model id; auto-detected from /v1/models if omitted")
    ap.add_argument("--concurrency", type=int, default=4,
                     help="Matches the throughput sweet spot found for LM Studio in the mf_span "
                          "review loop -- 4 concurrent requests measurably raised tokens/sec there.")
    ap.add_argument("--request-timeout", type=int, default=180)
    ap.add_argument("--max-tokens", type=int, default=1024,
                     help="Generation budget per task. A NATIVE REASONING model (QwQ, "
                          "DeepSeek-R1-distill, Qwen3-thinking, etc.) burns hundreds-to-thousands "
                          "of tokens on its own internal <think> deliberation before it ever "
                          "reaches the requested VERDICT line -- regardless of --chain-of-thought, "
                          "which only adds a SEPARATE scaffolded THINKING field for models that "
                          "DON'T already reason internally. For a native reasoning model, raise "
                          "this further (2048+) if mutagen_review_debug.log shows real responses "
                          "getting cut off mid-thought (finish_reason=length), and check whether "
                          "that model's chat template supports a lightweight no-think directive "
                          "(e.g. Qwen3's \"/no_think\" suffix) to skip the internal deliberation "
                          "entirely for a task this size.")
    ap.add_argument("--chain-of-thought", action="store_true",
                     help="Adds a scaffolded THINKING field to the requested answer format -- for "
                          "models that DON'T already reason internally on their own. Do not combine "
                          "with a native reasoning model; it already deliberates before answering, "
                          "and stacking a second, separately-prompted reasoning field on top of that "
                          "just adds more unbounded text for it to get lost in.")
    ap.add_argument("--capacity-retries", type=int, default=3)
    ap.add_argument("--capacity-retry-delay", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None, help="Cap total tasks processed (smoke test)")
    args = ap.parse_args()

    tasks = json.loads(TASKS_PATH.read_text())
    if args.limit:
        tasks = tasks[:args.limit]
    done = load_done(RESULTS_PATH)

    def has_valid_verdict(task_id):
        result = done.get(task_id)
        return result is not None and result["verdict"] not in ("UNPARSED", "INVALID_VERDICT")

    todo = [t for t in tasks if not has_valid_verdict(t["id"])]
    print(f"[INFO] {len(tasks)} total tasks, {len(todo)} to process "
          f"({len(tasks) - len(todo)} already have a valid verdict)")

    model = args.model or discover_model(args.host)
    system_prompt = build_system_prompt(args.chain_of_thought)
    print(f"[INFO] Using model: {model} @ {args.host}"
          + (" (chain-of-thought enabled)" if args.chain_of_thought else ""))

    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(review_task, args.host, model, system_prompt, t,
                                    args.request_timeout, args.max_tokens,
                                    args.capacity_retries, args.capacity_retry_delay)
                   for t in todo]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            with _write_lock:
                with open(RESULTS_PATH, "a") as f:
                    f.write(json.dumps(result) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                print(f"[INFO] {n_done}/{len(todo)} processed", flush=True)

    print(f"[INFO] Done. {n_done} tasks processed this run -> {RESULTS_PATH}")
    if DEBUG_LOG_PATH.exists():
        print(f"[INFO] Some responses didn't parse cleanly -- raw text saved to {DEBUG_LOG_PATH} "
              f"for inspection. Those tasks are UNPARSED/INVALID_VERDICT in the results file and "
              f"will be retried automatically next run.")


if __name__ == "__main__":
    main()
