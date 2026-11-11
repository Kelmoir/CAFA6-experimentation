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
    "Decide which of the listed terms are described as being clearly and unconditionally LOST or "
    "ABOLISHED by this specific mutation. Usually this is zero or one term -- only list more than "
    "one when the note describes multiple SEPARATE, independently-supported losses (different "
    "clauses, different activities), never as a way to hedge; each term you list must "
    "independently satisfy every rule below on its own.\n"
    "- If the description negates the loss (e.g. \"no loss of X\", \"no significant loss of X\", "
    "\"does not abolish Y\", \"no effect on Z activity\", \"normal X\"), that term is NOT lost -- "
    "do not select it. Read the whole clause: \"no\"/\"not\"/\"normal\" immediately changes the "
    "meaning of the loss-language that follows it.\n"
    "- If a term's loss is described as depending on OTHER mutations elsewhere in the protein (e.g. "
    "\"when associated with...\", \"in combination with...\"), still list it but mark it "
    "\"(conditional)\" -- this single mutation alone may not be sufficient evidence on its own.\n"
    "- An INCOMPLETE loss is NOT a clear, unconditional loss, even though the word \"loss\" "
    "appears -- do not select a term for it. Words that mean INCOMPLETE: \"partial\", \"reduced\", "
    "\"decreased\", \"diminished\", \"mild\", \"modest\", or a percentage that is not near zero. "
    "Words that mean NEAR-TOTAL and DO count as a clear loss, same as a bare \"loss of X\": "
    "\"complete\"/\"almost complete\"/\"near-complete\"/\"nearly complete\", \"strong\", "
    "\"severe\", \"substantial\", \"significant\", \"drastic\", \"abolished\", \"no detectable X\", "
    "or a tiny residual percent of wild-type (e.g. 0.02%). Do not treat every magnitude-qualified "
    "phrase as automatically insufficient -- only the INCOMPLETE words above are; the NEAR-TOTAL "
    "words are not weaker evidence than a bare \"loss of X\".\n"
    "- CHECK THIS FIRST, before the priority rule below: a clause that only names ONE SPECIFIC "
    "interaction partner or process (e.g. \"loss of interaction with MPDZ\", \"loss of binding "
    "ABL1/NOXA1\") is talking about that ONE relationship, not a general class of activity -- even "
    "when words like \"binding\"/\"interaction\" in the clause happen to overlap with a broad, "
    "nearly-universal candidate term's name (\"binding\", \"protein binding\"), that overlap is NOT "
    "a real match and the priority rule below does NOT apply to it. Losing one specific partner "
    "does not mean the protein's general binding capability (which covers every other partner too) "
    "was lost. Only select such a broad catch-all term if the clause itself states the GENERAL "
    "class in its own words (e.g. \"loss of DNA-binding activity\", not \"loss of binding to "
    "DNMT1\"); a bare named partner never qualifies, no matter which words it happens to share with "
    "a broad term's name.\n"
    "- PRIORITY RULE -- otherwise check this before any inference below: the listed terms include "
    "every ancestor of this protein's real annotations in the GO hierarchy, not just the most "
    "specific ones, so the exact general activity a note names is very often itself sitting in the "
    "list as its own term (e.g. a note saying \"catalytic activity\" with \"catalytic activity\" "
    "itself listed; \"hydrolase activity\" with \"hydrolase activity\" itself listed). If the "
    "note's own wording directly names a GENERAL CLASS of activity (not a specific named partner -- "
    "see the rule above) that matches one of the listed terms (exact name, or an obvious synonym/"
    "abbreviation of it), that is a confident match. Only fall through to the rules below when no "
    "listed term directly matches the note's own wording this way.\n"
    "- SPECIFICITY: once you have a confident match (from the priority rule or the is-a fallback "
    "below), ALWAYS check whether a MORE SPECIFIC listed term is also validly supported -- either "
    "by its own direct name match, or because it is this protein's own real annotation that the "
    "matched general term is an ancestor of. If so, prefer the MORE SPECIFIC term. Specificity is "
    "strictly more informative, and under GO's true-path rule a specific term being true already "
    "implies its ancestors are true too -- there is never a reason to settle for a general match "
    "when an equally well-supported, more specific one is available. (e.g. if \"loss of activity\" "
    "matches both the general listed \"catalytic activity\" AND this protein's own single, specific "
    "real enzyme activity, report the specific enzyme activity, not the general one.)\n"
    "- If the description names a GENERIC class of activity with NO direct name match (the priority "
    "rule above didn't apply) and more than one listed term could plausibly be the specific one "
    "meant, you cannot tell which -- answer NONE rather than guessing.\n"
    "- That said, when exactly ONE candidate term remains plausible AND it is a well-established, "
    "specific instance of the SAME general activity class the description itself names (e.g. the "
    "description says \"hydrolase activity\", no listed term is named exactly that, but the one "
    "candidate IS a specific hydrolase -- basic, uncontroversial biochemistry, not a guess), that "
    "counts as a confident match. This is NOT permission to pick the sole listed term by default "
    "whenever only one term happens to exist for the protein -- if the description names a "
    "DIFFERENT, unrelated activity or process (e.g. \"mitogenic activity\" when the one listed term "
    "is an unrelated specific enzyme activity), being the only option on the list is not itself a "
    "reason to select it; it must still genuinely match what was described.\n"
    "- A compound note may describe SEVERAL separate findings. Judge each clause on its own: a "
    "clause using ongoing/unchanged-behavior language (\"inhibits X\", \"promotes Y\", \"maintains "
    "Z\", stated without comparison to the wild-type) is NOT itself evidence of a loss, even if it "
    "immediately follows an unrelated negated clause -- it does not inherit loss-status from a "
    "neighboring clause about a different finding, and does not gain it either. If more than one "
    "clause each independently supports its OWN distinct term, list all of them. But never list "
    "both a term and its own ancestor for the SAME single clause/finding -- that is not two "
    "independent losses, just one finding at two levels of generality; list only the more specific "
    "one for that finding (see SPECIFICITY above).\n"
    "- Most tasks genuinely have NO correct match among the listed terms -- answering NONE is the "
    "expected, common outcome, not a failure to find something. Never force a weak or speculative "
    "match just to avoid answering NONE."
)

ANSWER_FORMAT = (
    "Respond with EXACTLY two lines, nothing else, no markdown, no commentary:\n"
    "VERDICTS: <comma-separated GO ids exactly as listed above, each optionally followed by "
    "\"(conditional)\", or the single word NONE>\n"
    "REASONING: <one sentence, plain text>"
)

THINKING_INSTRUCTIONS = (
    "In the THINKING block, follow these steps in order, and do not repeat a step once it is done:\n"
    "- Split the MUTAGEN description into its separate clauses/findings, and restate each in your "
    "own words -- a compound note often reports several unrelated things. For each clause, mark it "
    "as: NEGATED (\"no loss\", \"no significant loss\", \"no effect\", \"does not abolish\", "
    "\"normal\"), INCOMPLETE (\"partial\", \"reduced\", \"decreased\", \"diminished\", \"mild\", "
    "\"modest\", or a not-near-zero percentage -- not complete), ONGOING/UNCHANGED (describes "
    "current behavior with no comparison to wild-type, e.g. \"inhibits X\", \"promotes Y\"), or a "
    "GENUINE LOSS -- which includes both a bare \"loss of X\" AND a NEAR-TOTAL loss (\"complete\", "
    "\"almost/near-complete\", \"strong\", \"severe\", \"substantial\", \"significant\", "
    "\"drastic\", \"abolished\", \"no detectable X\", or a tiny residual percent of wild-type -- "
    "these are NOT weaker than a bare loss, do not downgrade them to INCOMPLETE). If every clause "
    "is NEGATED, INCOMPLETE, or ONGOING/UNCHANGED -- none is a GENUINE LOSS -- stop here and go "
    "straight to VERDICTS: NONE.\n"
    "- Otherwise, using only the genuine-loss clause(s): if a clause names only ONE SPECIFIC "
    "interaction partner or process (e.g. \"loss of interaction with MPDZ\"), that is NOT a general "
    "class of activity -- do not let a shared word like \"binding\"/\"interaction\" count as a "
    "match to a broad, nearly-universal candidate term (\"binding\", \"protein binding\") just "
    "because it overlaps textually; losing one partner does not mean the protein's general binding "
    "capability across all its other partners was lost too. Skip straight to the per-term scoring "
    "pass below for this clause.\n"
    "- Otherwise, FIRST check every listed term's name directly against the clause's own wording "
    "(exact name, synonym, or abbreviation of a GENERAL CLASS the clause itself states -- e.g. the "
    "clause says \"catalytic activity\" and \"catalytic activity\" is itself one of the listed "
    "terms). The listed terms include every ancestor of this protein's real annotations, not just "
    "the most specific ones, so this direct match is very often available. If found, do NOT stop "
    "yet -- check whether a MORE SPECIFIC listed term is ALSO supported (either its own direct name "
    "match, or this protein's own real annotation that your matched general term is an ancestor "
    "of). If a more specific valid match exists, use that one instead -- specificity is strictly "
    "more informative, and a specific term being true already implies the general one is true too "
    "under GO's true-path rule, so never settle for the general match when a specific one is "
    "equally well-supported. Once you have the most specific valid match, that is the answer at "
    "high confidence -- skip the scoring pass below entirely for this clause, this is more "
    "reliable than any inference.\n"
    "- Only if no listed term directly names what the clause says: for each listed GO term in turn, "
    "EXACTLY ONCE, state whether that clause's own wording -- or a very well-known, unambiguous "
    "fact, not a guess -- connects it to that term's specific definition, and rate your confidence 1 "
    "(no real connection) to 10 (specific/mechanistic match). A clause naming only a specific "
    "interaction partner or process, without itself stating a general class of activity, does not "
    "confirm a listed term just because both involve the same protein -- rate it low unless you are "
    "genuinely confident the named partner/process matches that term's own definition. If the "
    "clause names a GENERIC activity class with no direct name match and only one listed term is a "
    "well-established, specific instance of THAT SAME CLASS, that single exact match counts as high "
    "confidence -- but this is not permission to default to whichever term happens to be the only "
    "one listed when the clause names something else entirely unrelated to it; the sole candidate "
    "must still genuinely match what was described, not just be the only option present. If MORE "
    "THAN ONE listed term could plausibly be the specific one meant, rate all of them low -- you "
    "cannot tell which one, so this should still end in NONE.\n"
    "- Once every term has been rated exactly once, stop analyzing. Collect every term scoring >=7 "
    "across all clauses -- usually zero or one, but list more than one if separate clauses each "
    "independently support a different term. Never include both a term and its own ancestor for "
    "the same single clause/finding -- only the more specific one for that finding. If nothing "
    "scores >=7 anywhere, answer NONE. Do not revisit or re-score a term already rated, and do not "
    "force an answer just to avoid NONE -- most tasks genuinely have no correct match."
)

COT_ANSWER_FORMAT = (
    "Respond in this order, starting with THINKING (nothing before it):\n"
    "THINKING: <follow the numbered steps given above. Can span as many lines as needed; there is "
    "no length limit, but each term is rated once, not revisited.>\n"
    "VERDICTS: <comma-separated GO ids exactly as listed above, each optionally followed by "
    "\"(conditional)\", or the single word NONE>\n"
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


VERDICT_TOKEN_RE = re.compile(r"^(GO:\d+)\s*(\(conditional\))?$", re.I)


def parse_response(text: str, valid_go_ids: set):
    """Returns (status, verdicts, reasoning). status is one of OK/NONE/UNPARSED/INVALID_VERDICT;
    verdicts is a list of {"go_id", "conditional"} dicts (empty for NONE/UNPARSED/INVALID_VERDICT).
    Multiple verdicts are allowed -- a note can genuinely describe more than one separate loss
    (e.g. "strong loss of ATPase AND inorganic triphosphatase activities" supports two distinct
    terms at once); any single hallucinated/unlisted id invalidates the whole response rather than
    silently dropping just that one token, consistent with the single-verdict version's guard."""
    v = _last_match(r"VERDICTS?:\s*(.+)", text, re.I)
    r = _last_match(r"REASONING:\s*(.+)", text, re.I)
    reasoning = r.group(1).strip().split("\n")[0][:400] if r else "(no reasoning extracted)"
    if v is None:
        return "UNPARSED", [], reasoning
    raw = v.group(1).strip().rstrip(".")
    if raw.upper() == "NONE":
        return "NONE", [], reasoning
    verdicts = []
    for token in raw.split(","):
        m = VERDICT_TOKEN_RE.match(token.strip())
        if not m:
            return "INVALID_VERDICT", [], reasoning
        go_id = m.group(1).upper()
        if go_id not in valid_go_ids:
            return "INVALID_VERDICT", [], reasoning  # hallucinated an id not in this task's list
        verdicts.append({"go_id": go_id, "conditional": bool(m.group(2))})
    if not verdicts:
        return "INVALID_VERDICT", [], reasoning
    return "OK", verdicts, reasoning


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
    checking status -- see main()."""
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
        return {"id": task["id"], "accession": task["accession"], "status": "UNPARSED",
                "verdicts": [], "reasoning": f"HTTP {e.status} {e.reason}"}
    except (OSError, TimeoutError) as e:
        return {"id": task["id"], "accession": task["accession"], "status": "UNPARSED",
                "verdicts": [], "reasoning": f"request failed: {e}"}
    status, verdicts, reasoning = parse_response(raw, valid_go_ids)
    if status in ("UNPARSED", "INVALID_VERDICT"):
        if status == "UNPARSED" and finish_reason == "length":
            reasoning = f"hit max_tokens ({max_tokens}) before VERDICTS appeared"
        log_debug(task["id"], task["accession"], f"{status} finish_reason={finish_reason}", raw)
    return {"id": task["id"], "accession": task["accession"], "status": status,
            "verdicts": verdicts, "reasoning": reasoning}


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

    def is_resolved(task_id):
        result = done.get(task_id)
        return result is not None and result["status"] not in ("UNPARSED", "INVALID_VERDICT")

    todo = [t for t in tasks if not is_resolved(t["id"])]
    print(f"[INFO] {len(tasks)} total tasks, {len(todo)} to process "
          f"({len(tasks) - len(todo)} already resolved)")

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
