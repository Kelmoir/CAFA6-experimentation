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
    "- If the description negates the loss (e.g. \"no loss of X\", \"does not abolish Y\", \"no "
    "effect on Z activity\", \"normal X\"), that term is NOT lost -- do not select it.\n"
    "- If the loss is described as depending on OTHER mutations elsewhere in the protein (e.g. "
    "\"when associated with...\", \"in combination with...\"), select the term but set CONDITIONAL "
    "to YES -- this single mutation alone may not be sufficient evidence on its own.\n"
    "- The note may use different wording than a term's official name (a synonym, abbreviation, "
    "or more specific/general phrasing) -- still match it if the underlying biological activity is "
    "clearly the same.\n"
    "- If no listed term is clearly and specifically described as lost, or the match is too vague "
    "to be confident, answer NONE -- do not force a match."
)

ANSWER_FORMAT = (
    "Respond with EXACTLY three lines, nothing else, no markdown, no commentary:\n"
    "VERDICT: <GO id exactly as listed above, or NONE>\n"
    "CONDITIONAL: YES or NO (NO if VERDICT is NONE)\n"
    "REASONING: <one sentence, plain text>"
)

COT_ANSWER_FORMAT = (
    "Respond in this order, starting with THINKING (nothing before it):\n"
    "THINKING: <briefly consider each candidate term in turn against the note; name which one (if "
    "any) matches and say plainly whether the note negates it or makes it conditional on another "
    "mutation. Then settle on an answer below.>\n"
    "VERDICT: <GO id exactly as listed above, or NONE>\n"
    "CONDITIONAL: YES or NO (NO if VERDICT is NONE)\n"
    "REASONING: <one sentence, plain text, summarizing the conclusion your THINKING reached>"
)


def build_system_prompt(chain_of_thought: bool) -> str:
    fmt = COT_ANSWER_FORMAT if chain_of_thought else ANSWER_FORMAT
    return f"{TASK_DESCRIPTION}\n\n{INSTRUCTIONS}\n\n{fmt}"


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
        return {"id": task["id"], "verdict": "UNPARSED", "conditional": False,
                "reasoning": f"HTTP {e.status} {e.reason}"}
    except (OSError, TimeoutError) as e:
        return {"id": task["id"], "verdict": "UNPARSED", "conditional": False,
                "reasoning": f"request failed: {e}"}
    verdict, conditional, reasoning = parse_response(raw, valid_go_ids)
    if verdict == "UNPARSED" and finish_reason == "length":
        reasoning = f"hit max_tokens ({max_tokens}) before VERDICT appeared"
    return {"id": task["id"], "accession": task["accession"], "verdict": verdict,
            "conditional": conditional, "reasoning": reasoning}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="LM Studio server, e.g. http://192.168.1.122:1234")
    ap.add_argument("--model", default=None, help="Model id; auto-detected from /v1/models if omitted")
    ap.add_argument("--concurrency", type=int, default=4,
                     help="Matches the throughput sweet spot found for LM Studio in the mf_span "
                          "review loop -- 4 concurrent requests measurably raised tokens/sec there.")
    ap.add_argument("--request-timeout", type=int, default=120)
    ap.add_argument("--max-tokens", type=int, default=256,
                     help="Short answer (VERDICT/CONDITIONAL/REASONING, no reasoning trace by "
                          "default) -- raise substantially if --chain-of-thought is set.")
    ap.add_argument("--chain-of-thought", action="store_true",
                     help="Let the model reason in a THINKING field before answering -- recommended "
                          "for smaller/non-reasoning models; needs --max-tokens raised accordingly.")
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


if __name__ == "__main__":
    main()
