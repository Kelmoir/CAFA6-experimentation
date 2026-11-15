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

import obonet  # noqa: E402
import networkx as nx  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
from generate_dataset import restrict_to_true_path  # noqa: E402

GO_OBO_PATH = (Path(__file__).resolve().parent.parent.parent.parent
               / "data/cafa-6-protein-function-prediction/Train/go-basic.obo")
TASKS_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_tasks.json"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_results.jsonl"
DEBUG_LOG_PATH = Path(__file__).resolve().parent.parent / "mutagen_review_debug.log"

TASK_DESCRIPTION = (
    "You are reviewing a site-directed mutagenesis (MUTAGEN) annotation from UniProt to decide "
    "whether it provides trustworthy evidence that a SPECIFIC molecular function is lost in the "
    "mutant protein. You will be shown the mutation, UniProt's free-text description of its "
    "effect, and a list of this protein's known molecular-function GO terms, each with its "
    "official name and definition."
)

INSTRUCTIONS = (
    "Decide which of the listed terms are described as being clearly and unconditionally LOST or "
    "ABOLISHED by this specific mutation. Usually this is zero or one term -- only list more than "
    "one when the note describes multiple SEPARATE, independently-supported losses (different "
    "clauses, different activities), never as a way to hedge; each term you list must "
    "independently satisfy every rule below on its own.\n"
    "- If the description negates the loss (e.g. \"no loss of X\", \"no significant loss of X\", "
    "\"does not abolish Y\", \"no effect on Z activity\", \"normal X\"), that term is NOT lost. "
    "Read the whole clause -- \"no\"/\"not\"/\"normal\" immediately change the meaning of the "
    "loss-language that follows.\n"
    "- If a term's loss is described as depending on OTHER mutations elsewhere in the protein (e.g. "
    "\"when associated with...\", \"in combination with...\"), still list it but mark it "
    "\"(conditional)\" -- this mutation alone may not be sufficient evidence on its own.\n"
    "- An INCOMPLETE loss is NOT a clear, unconditional loss, even though the word \"loss\" "
    "appears. INCOMPLETE words (do NOT count): \"partial\", \"reduced\", \"decreased\", "
    "\"diminished\", \"mild\", \"modest\", or a percentage not near zero. NEAR-TOTAL words (DO "
    "count, same as a bare \"loss of X\"): \"complete\"/\"almost complete\"/\"near-complete\"/"
    "\"nearly complete\", \"strong\", \"severe\", \"substantial\", \"significant\", \"drastic\", "
    "\"abolished\", \"no detectable X\", or a tiny residual percent of wild-type (e.g. 0.02%). "
    "NEAR-TOTAL words are never weaker evidence than a bare \"loss of X\" -- do not downgrade "
    "them.\n"
    "- CHECK THIS FIRST, before the priority rule below: a clause naming ONE SPECIFIC interaction "
    "partner or process (e.g. \"loss of interaction with MPDZ\", \"loss of binding to "
    "ABL1/NOXA1\") describes that ONE relationship, not a general activity class -- even when "
    "words like \"binding\"/\"interaction\" textually overlap with a broad, nearly-universal "
    "candidate term's name (\"binding\", \"protein binding\"), that overlap is NOT a match and "
    "the priority rule below does not apply. Losing one named partner does not mean the general "
    "capability covering every other partner was lost too. Only select such a broad term if the "
    "clause itself states the GENERAL class in its own words (e.g. \"loss of DNA-binding "
    "activity\", not \"loss of binding to DNMT1\").\n"
    "- PRIORITY RULE -- check this before any inference below (after the partner-exclusion check "
    "above): the listed terms include every ancestor of this protein's real annotations in the GO "
    "hierarchy, not just the most specific ones, so the exact general activity a note names is "
    "often itself sitting in the list (e.g. a note saying \"catalytic activity\" with \"catalytic "
    "activity\" itself listed). If the note's own wording directly names a GENERAL CLASS of "
    "activity (not a specific named partner -- see above) matching one listed term's exact name, "
    "synonym, or abbreviation -- OR a specific compound/substrate/product the note names (e.g. "
    "\"loss of production of aristolochene\") appears literally in a listed term's DEFINITION text "
    "(not just its name, e.g. a definition reading \"...= (+)-5-epi-aristolochene + ...\") -- that "
    "is a confident match. A bare \"loss of activity\" or \"loss of "
    "function\" (no specific activity named at all) counts as directly naming \"catalytic "
    "activity\" whenever GO:0003824 is itself listed -- treat it exactly like a note that says "
    "\"loss of catalytic activity\", not as an unmatched generic class. Fall through to the rules "
    "below only when no listed term directly matches the note's wording this way.\n"
    "- SPECIFICITY: once you have a confident match (from the priority rule or the is-a fallback "
    "below), check whether a MORE SPECIFIC listed term is also validly supported -- by its own "
    "direct name match, or because it is this protein's own real annotation that the matched "
    "general term is an ancestor of. If so, prefer the MORE SPECIFIC term: under GO's true-path "
    "rule a specific term being true already implies its ancestors are true, so never settle for "
    "a general match when an equally well-supported, more specific one is available (e.g. if "
    "\"loss of activity\" matches both the general listed \"catalytic activity\" and this "
    "protein's own specific real enzyme activity, report the specific enzyme activity, not the "
    "general one). Use each term's own numbered \"is_a (N)\" reference to tell a real single "
    "chain apart from genuine sibling ambiguity: if one candidate's is_a chain leads to another "
    "candidate you're considering -- directly, or by following further is_a references from there "
    "-- they are NOT competing siblings, they're the same chain, so climb to the most specific one "
    "in it. Only treat two candidates as genuinely competing, unrelated siblings when neither "
    "is_a's the other, even by following the chain further. BUT if this check finds TWO OR MORE "
    "genuinely unrelated, equally-plausible specific siblings with no way to tell which one is "
    "meant (e.g. the protein has both a mannosidase and an unrelated xylosidase activity, and the "
    "note just says \"loss of catalytic activity\" with nothing to pick between them), do NOT "
    "discard your original match or answer NONE -- the ambiguity is only about which more-specific "
    "sibling, not about whether the original, already-confirmed general term was lost. First check "
    "whether the competing siblings' is_a chains lead to a shared term that is itself listed and "
    "MORE SPECIFIC than your original general match -- if so, report that shared term instead, "
    "since it's more informative and still unambiguous. Otherwise keep and report the original, "
    "more general match. This is different from climbing a single, unambiguous chain of ancestors "
    "to one specific leaf term (always do that) -- it only applies when there are multiple "
    "competing, unrelated specific siblings.\n"
    "- Decide using only the note's own wording and the listed terms' names/definitions/is_a "
    "references -- never your own outside knowledge about what this specific protein (by its "
    "accession or identity) actually is or does, even if you recognize it. The same applies to any "
    "OTHER gene/protein the note happens to name (e.g. a complementation partner like \"ACS2\") -- "
    "do not use outside knowledge of what THAT gene/protein's function is either, only what the "
    "note's own wording states about the activity being lost; a note naming a second protein as "
    "context for a rescue/complementation experiment is not itself evidence of which specific "
    "activity THIS mutation lost. If the note's wording is generic and the listed terms alone "
    "don't disambiguate which specific one is meant, that is the sibling-ambiguity case above, "
    "regardless of what you believe you know about this particular protein, or any other one it "
    "names, from training.\n"
    "- If the description names a GENERIC class of activity with NO direct name match (the "
    "priority rule above didn't apply) and more than one listed term could plausibly be the "
    "specific one meant, you cannot tell which -- answer NONE rather than guessing.\n"
    "- EXCEPTION: when exactly ONE candidate term remains plausible and it is a well-established, "
    "specific instance of the SAME general activity class the description names (e.g. the "
    "description says \"hydrolase activity\", no listed term is named exactly that, but the one "
    "candidate IS a specific hydrolase -- basic, uncontroversial biochemistry, not a guess), that "
    "counts as a confident match. This is NOT permission to default to the sole listed term "
    "whenever the description names a DIFFERENT, unrelated activity or process (e.g. \"mitogenic "
    "activity\" when the one listed term is an unrelated specific enzyme activity) -- it must "
    "still genuinely match what was described, not just be the only option present.\n"
    "- A compound note may describe SEVERAL separate findings -- judge each clause on its own. A "
    "clause using ongoing/unchanged-behavior language (\"inhibits X\", \"promotes Y\", "
    "\"maintains Z\", stated without comparison to the wild-type) is NOT itself evidence of a "
    "loss, even if it immediately follows an unrelated negated clause -- it neither inherits "
    "loss-status from a neighboring clause about a different finding, nor gains it. If more than "
    "one clause each independently supports its OWN distinct term, list all of them. But never "
    "list both a term and its own ancestor for the SAME single clause/finding -- that is one "
    "finding at two levels of generality, not two independent losses; list only the more specific "
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

# For --prompt-level simplified/minimal, used only with a LEAF-ONLY tasks file (built via
# build_mutagen_review_tasks.py --leaf-only), where every candidate is already this protein's own
# real, directly-asserted MF annotation -- no ancestor closure, so none of the is_a/specificity/
# sibling-ambiguity machinery in INSTRUCTIONS applies or is included. The negation/magnitude-word/
# specific-partner-exclusion/no-outside-knowledge rules are kept here, in full for "simplified" and
# tightened for "minimal" -- these address basic text-comprehension failures an early model showed
# even on trivial candidate lists, not DAG complexity, so they're not simplification targets.
SIMPLIFIED_INSTRUCTIONS = (
    "Decide which of the listed terms are described as being clearly and unconditionally LOST or "
    "ABOLISHED by this specific mutation. Usually this is zero or one term -- only list more than "
    "one when the note describes multiple SEPARATE, independently-supported losses (different "
    "clauses, different activities), never as a way to hedge; each term you list must "
    "independently satisfy every rule below on its own. Every listed term is this protein's own "
    "real, already-known molecular-function annotation (not a generalization of it).\n"
    "- If the description negates the loss (e.g. \"no loss of X\", \"no significant loss of X\", "
    "\"does not abolish Y\", \"no effect on Z activity\", \"normal X\"), that term is NOT lost. "
    "Read the whole clause -- \"no\"/\"not\"/\"normal\" immediately change the meaning of the "
    "loss-language that follows.\n"
    "- If a term's loss is described as depending on OTHER mutations elsewhere in the protein (e.g. "
    "\"when associated with...\", \"in combination with...\"), still list it but mark it "
    "\"(conditional)\" -- this mutation alone may not be sufficient evidence on its own.\n"
    "- An INCOMPLETE loss is NOT a clear, unconditional loss, even though the word \"loss\" "
    "appears. INCOMPLETE words (do NOT count): \"partial\", \"reduced\", \"decreased\", "
    "\"diminished\", \"mild\", \"modest\", or a percentage not near zero. NEAR-TOTAL words (DO "
    "count, same as a bare \"loss of X\"): \"complete\"/\"almost complete\"/\"near-complete\"/"
    "\"nearly complete\", \"strong\", \"severe\", \"substantial\", \"significant\", \"drastic\", "
    "\"abolished\", \"no detectable X\", or a tiny residual percent of wild-type (e.g. 0.02%). "
    "NEAR-TOTAL words are never weaker evidence than a bare \"loss of X\" -- do not downgrade "
    "them.\n"
    "- CHECK THIS FIRST, before matching below: a clause naming ONE SPECIFIC interaction partner, "
    "process, channel, or receptor subtype (e.g. \"loss of interaction with MPDZ\", \"loss of "
    "binding to ABL1/NOXA1\", a toxin blocking one specific ion channel or receptor subtype) "
    "describes that ONE relationship, not a general activity class -- even when words like "
    "\"binding\"/\"interaction\" textually overlap with a broad candidate term's name (\"binding\", "
    "\"protein binding\"), that overlap is NOT a match. Losing one named partner or subtype does "
    "not mean the general capability covering every other partner/subtype was lost too. Only "
    "select such a broad term if the clause itself states the GENERAL class in its own words (e.g. "
    "\"loss of DNA-binding activity\", not \"loss of binding to DNMT1\"), OR if a listed term is "
    "itself specifically about that exact named partner or subtype (e.g. a term literally named "
    "\"eukaryotic initiation factor 4E binding\" for a clause about losing interaction with EIF4E "
    "-- that is a direct match to the named partner, not a forbidden generalization).\n"
    "- MATCHING: decide whether the note's wording identifies one listed term's SPECIFIC activity "
    "as lost -- by its exact name, a well-known synonym/abbreviation, an unambiguous paraphrase, a "
    "specific compound/substrate/product the note names appearing literally in a listed term's "
    "DEFINITION text (not just its name -- e.g. a note about losing \"production of aristolochene\" "
    "matching a term whose definition reads \"...= (+)-5-epi-aristolochene + ...\"), or "
    "a well-known, uncontroversial biochemistry/biology fact connecting what's described to "
    "that term's own definition (not a guess) -- e.g. a toxin's \"loss of inhibition of neuronal "
    "and muscle receptors\" is the standard description of losing nicotinic acetylcholine receptor "
    "inhibition, matching a listed acetylcholine-receptor term even without that exact wording. A "
    "bare \"activity\"/\"function\" with no specific activity named directly matches a listed term "
    "that itself uses that same generic word in its own name (e.g. \"catalytic activity\" listed) "
    "-- this holds no matter how many other, more specific candidates are also listed. Short of "
    "that, it still counts as a match when exactly ONE listed term remains plausible, or when the "
    "note's own wording otherwise clearly singles out one listed term over the others; if it could "
    "plausibly mean more than one listed term and none of them shares the bare wording, a "
    "CATALYTIC/enzymatic term (one describing a chemical reaction) is the natural meaning of "
    "\"activity\", not a BINDING term (binding to a cofactor, metal, or small molecule) that only "
    "supports that catalysis -- prefer the catalytic one unless the note itself mentions binding "
    "specifically. If more than one listed term remains plausible and nothing in the note "
    "distinguishes between them, you cannot tell which one is meant -- answer NONE rather than "
    "guessing.\n"
    "- Decide using only the note's own wording and the listed terms' names/definitions -- never "
    "your own outside knowledge about what this specific protein (by its accession or identity) "
    "actually is or does, even if you recognize it. The same applies to any OTHER gene/protein the "
    "note happens to name (e.g. a complementation partner like \"ACS2\") -- do not use outside "
    "knowledge of what THAT gene/protein's function is either; a note naming a second protein as "
    "context for a rescue/complementation experiment is not itself evidence of which specific "
    "activity THIS mutation lost.\n"
    "- A compound note may describe SEVERAL separate findings -- judge each clause on its own. A "
    "clause using ongoing/unchanged-behavior language (\"inhibits X\", \"promotes Y\", "
    "\"maintains Z\", stated without comparison to the wild-type) is NOT itself evidence of a "
    "loss, even if it immediately follows an unrelated negated clause -- it neither inherits "
    "loss-status from a neighboring clause about a different finding, nor gains it. If more than "
    "one clause each independently supports its OWN distinct term, list all of them.\n"
    "- Most tasks genuinely have NO correct match among the listed terms -- answering NONE is the "
    "expected, common outcome, not a failure to find something. Never force a weak or speculative "
    "match just to avoid answering NONE."
)

# Drops the sole-candidate carve-out and the compound-clause elaboration down to one-liners, but
# keeps negation/magnitude-word/partner-exclusion/no-outside-knowledge in full -- per the explicit
# call that these address basic comprehension, not something to compress further.
MINIMAL_INSTRUCTIONS = (
    "Decide which of the listed terms (each one of this protein's own real, already-known "
    "molecular-function annotations) are clearly and unconditionally LOST by this mutation. "
    "Usually zero or one; list more than one only for genuinely separate, independently-supported "
    "losses.\n"
    "- Negated (\"no loss of X\", \"no significant loss of X\", \"does not abolish Y\", \"no "
    "effect on Z\", \"normal X\") means NOT lost.\n"
    "- A loss that depends on another mutation elsewhere (\"when associated with...\", \"in "
    "combination with...\") is still listed, but marked \"(conditional)\".\n"
    "- INCOMPLETE words (\"partial\", \"reduced\", \"decreased\", \"diminished\", \"mild\", "
    "\"modest\", a percentage not near zero) do NOT count as a loss. NEAR-TOTAL words "
    "(\"complete\", \"almost complete\", \"near-complete\", \"nearly complete\", \"strong\", "
    "\"severe\", \"substantial\", \"significant\", \"drastic\", \"abolished\", \"no detectable X\", "
    "a tiny residual percent) count the same as a bare \"loss of X\" -- never downgrade them.\n"
    "- A clause naming only ONE SPECIFIC interaction partner, channel, or receptor subtype (e.g. "
    "\"loss of binding to ABL1\", a toxin blocking one specific ion channel or receptor subtype) "
    "is not evidence that a general, nearly-universal term like \"protein binding\" or a generic "
    "regulator/inhibitor term covering every subtype was lost, even though the words overlap -- "
    "unless a listed term is itself specifically about that exact named partner or subtype (e.g. "
    "a term literally naming EIF4E, for a clause about losing interaction with EIF4E).\n"
    "- A bare \"activity\"/\"function\" with no specific activity named directly matches a listed "
    "term that itself uses that same generic word in its own name (e.g. \"catalytic activity\" "
    "listed) -- this holds no matter how many other, more specific candidates are also listed. If "
    "instead it could plausibly mean more than one listed term and none of them shares that bare "
    "wording, a CATALYTIC/enzymatic term (one describing a chemical reaction) is the natural "
    "meaning of \"activity\", not a BINDING term (binding to a cofactor, metal, or small molecule) "
    "that only supports that catalysis -- prefer the catalytic one unless the note itself mentions "
    "binding specifically.\n"
    "- Report a term if the note's own wording identifies that specific activity as lost -- by "
    "name, a well-known synonym/abbreviation, an unambiguous paraphrase, a specific compound/"
    "substrate/product the note names appearing literally in a listed term's DEFINITION text (not "
    "just its name -- e.g. a note about losing \"production of aristolochene\" matching a term "
    "whose definition reads \"...= (+)-5-epi-aristolochene + ...\"), OR a well-known, "
    "uncontroversial biochemistry/biology fact connecting what's described to that term's own "
    "definition (not a guess) -- e.g. a toxin's \"loss of inhibition of neuronal and muscle "
    "receptors\" is the standard description of losing nicotinic acetylcholine receptor "
    "inhibition, matching a listed acetylcholine-receptor term even without that exact wording. "
    "Never decide from outside knowledge of what this protein -- or any OTHER gene/protein the "
    "note happens to name, e.g. a complementation partner -- actually is or does; naming a second "
    "protein as context for a rescue/complementation experiment is not evidence of which activity "
    "THIS mutation lost. If you still can't tell which listed term (if any) is meant, answer NONE "
    "rather than guessing.\n"
    "- A note may describe more than one separate loss -- judge each independently. Ongoing/"
    "unchanged language (\"inhibits X\", \"promotes Y\", no comparison to wild-type) is not "
    "evidence of loss.\n"
    "- NONE is the common, expected answer -- don't force a match."
)

THINKING_INSTRUCTIONS = (
    "In the THINKING block, follow these steps in order, and do not repeat a step once it is done:\n"
    "- Split the MUTAGEN description into its separate clauses/findings, and restate each in your "
    "own words -- a compound note often reports several unrelated things. Classify each clause as "
    "NEGATED (per the negation rule above), INCOMPLETE (per the magnitude-word rule above), "
    "ONGOING/UNCHANGED (describes current behavior with no comparison to wild-type, e.g. "
    "\"inhibits X\", \"promotes Y\"), or GENUINE LOSS (a bare \"loss of X\", or any NEAR-TOTAL "
    "word from the rule above -- these are never weaker than a bare loss; do not downgrade them "
    "to INCOMPLETE). If every clause is NEGATED, INCOMPLETE, or ONGOING/UNCHANGED -- none is a "
    "GENUINE LOSS -- stop here and go straight to VERDICTS: NONE.\n"
    "- Otherwise, for each genuine-loss clause: first apply the specific-partner-exclusion rule "
    "above (a clause naming only one interaction partner or process is never a match to a broad, "
    "nearly-universal candidate term just because a word like \"binding\"/\"interaction\" "
    "overlaps textually). Then apply the PRIORITY RULE above -- check every listed term's name "
    "directly against the clause's own wording, remembering a bare \"loss of activity\"/\"loss of "
    "function\" counts as naming \"catalytic activity\" per that rule -- and if a direct match is "
    "found, apply the SPECIFICITY rule above to see whether an even more specific valid match "
    "exists; follow each candidate's own \"is_a (N)\" reference to tell a real single chain apart "
    "from genuine sibling ambiguity before concluding there's a conflict. Use the most specific "
    "one found via a single chain, unless that check turns up multiple genuinely unrelated "
    "siblings (neither is_a's the other, even transitively) with no way to pick between them -- "
    "then check for a shared, more-specific-than-general term their is_a chains both lead to and "
    "report that if one exists, otherwise keep the original, already-confirmed general match "
    "instead (per SPECIFICITY above), don't discard it either way. Never use outside knowledge "
    "about what this specific protein is to break a tie the listed terms themselves don't break. "
    "Once you have a match this way, that is the answer at high confidence -- skip the scoring "
    "pass below entirely for this clause, since this is more reliable than any inference.\n"
    "- Only if no listed term directly names what the clause says: for each listed GO term in "
    "turn, EXACTLY ONCE, state whether the clause's own wording -- or a very well-known, "
    "unambiguous fact, not a guess -- connects it to that term's specific definition, and rate "
    "your confidence 1 (no real connection) to 10 (specific/mechanistic match). A clause naming "
    "only a specific interaction partner or process, without itself stating a general class of "
    "activity, does not confirm a listed term just because both involve the same protein -- rate "
    "it low unless you are genuinely confident the named partner/process matches that term's own "
    "definition. The sole-candidate EXCEPTION above applies here too if it fits. If MORE THAN ONE "
    "listed term could plausibly be the specific one meant, rate all of them low -- you cannot "
    "tell which one, so this should still end in NONE.\n"
    "- Once every term has been rated exactly once, stop analyzing. Collect every term scoring "
    ">=7 across all clauses -- usually zero or one, but list more than one if separate clauses "
    "each independently support a different term. Never include both a term and its own ancestor "
    "for the same single clause/finding -- only the more specific one for that finding. If "
    "nothing scores >=7 anywhere, answer NONE. Do not revisit or re-score a term already rated, "
    "and do not force an answer just to avoid NONE -- most tasks genuinely have no correct match."
)

COT_ANSWER_FORMAT = (
    "Respond in this order, starting with THINKING (nothing before it):\n"
    "THINKING: <follow the numbered steps given above. Can span as many lines as needed; there is "
    "no length limit, but each term is rated once, not revisited.>\n"
    "VERDICTS: <comma-separated GO ids exactly as listed above, each optionally followed by "
    "\"(conditional)\", or the single word NONE>\n"
    "REASONING: <one sentence, plain text, summarizing the conclusion your THINKING reached>"
)


def build_system_prompt(prompt_level: str, chain_of_thought: bool) -> str:
    """chain_of_thought is only meaningful for prompt_level="full" -- the DAG-reasoning steps it
    scaffolds don't exist in the simplified/minimal (leaf-only) prompts, so those two never offer
    a THINKING field; main() rejects --chain-of-thought with a non-full --prompt-level up front."""
    if prompt_level == "full":
        if chain_of_thought:
            return f"{TASK_DESCRIPTION}\n\n{INSTRUCTIONS}\n\n{THINKING_INSTRUCTIONS}\n\n{COT_ANSWER_FORMAT}"
        return f"{TASK_DESCRIPTION}\n\n{INSTRUCTIONS}\n\n{ANSWER_FORMAT}"
    instructions = SIMPLIFIED_INSTRUCTIONS if prompt_level == "simplified" else MINIMAL_INSTRUCTIONS
    return f"{TASK_DESCRIPTION}\n\n{instructions}\n\n{ANSWER_FORMAT}"


_go_graph = None
_ancestor_cache = {}


def _ancestors_of(go_id):
    """True is_a/part_of ancestors of go_id (obonet's child->parent edges, so nx.descendants from
    a child node reaches its ancestors) -- same construction as build_mutagen_review_tasks.py's
    own ancestor lookup, loaded lazily here since not every import of this module needs it."""
    global _go_graph
    if _go_graph is None:
        _go_graph = restrict_to_true_path(obonet.read_obo(str(GO_OBO_PATH)))
    if go_id not in _ancestor_cache:
        _ancestor_cache[go_id] = nx.descendants(_go_graph, go_id) if go_id in _go_graph else set()
    return _ancestor_cache[go_id]


def build_prompt(task: dict, prompt_level: str = "full") -> str:
    header = [
        f"Protein {task['accession']}, mutation at residues {task['start'] + 1}-{task['end']} "
        f"(1-indexed):",
        "",
        f'MUTAGEN description: "{task["note"]}"',
        "",
    ]
    candidates = task["candidates"]

    if prompt_level != "full":
        # Leaf-only candidates (build_mutagen_review_tasks.py --leaf-only): every candidate is
        # already this protein's own real annotation, so there's no ancestor closure to render.
        lines = header + ["This protein's known molecular-function GO terms:"]
        for c in candidates:
            lines.append(f"- {c['go_id']}: {c['name']}")
            lines.append(f"  Definition: {c['def'] or '(no definition available)'}")
        return "\n".join(lines)

    candidate_ids = {c["go_id"] for c in candidates}
    # ancestors of each candidate, restricted to this task's own candidate set
    in_list_ancestors = {c["go_id"]: _ancestors_of(c["go_id"]) & candidate_ids for c in candidates}
    # broad-to-specific: an ancestor always has strictly fewer in-list ancestors than its
    # descendant (true-path rule), so this sort is a valid topological order
    ordered = sorted(candidates, key=lambda c: (len(in_list_ancestors[c["go_id"]]), c["go_id"]))
    index_of = {c["go_id"]: i for i, c in enumerate(ordered, start=1)}

    lines = header + [
        "This protein's known molecular-function GO terms, numbered broadest (1) to most "
        "specific, each followed by \"is_a (N)\" naming the earlier, more general listed term(s) "
        "it is a direct child of -- omitted when this term has no such parent in this list.",
        "Follow is_a references (possibly through more than one step) to tell a real chain of "
        "increasing specificity, which should be climbed to its most specific term, apart from "
        "truly unrelated terms, which share no is_a reference at all.",
        "",
    ]
    for c in ordered:
        go_id = c["go_id"]
        ancestors = in_list_ancestors[go_id]
        # immediate parents only (transitive reduction): an ancestor that is itself an ancestor of
        # another candidate ancestor is not direct -- that other ancestor is the closer parent
        immediate = sorted(
            (a for a in ancestors if not any(a in in_list_ancestors[b] for b in ancestors if b != a)),
            key=lambda a: index_of[a],
        )
        is_a = f" is_a ({', '.join(str(index_of[a]) for a in immediate)})" if immediate else ""
        lines.append(f"{index_of[go_id]}. {go_id}: {c['name']}{is_a}")
        lines.append(f"   Definition: {c['def'] or '(no definition available)'}")
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
                 capacity_retries, capacity_retry_delay, api_key, prompt_level):
    valid_go_ids = {c["go_id"] for c in task["candidates"]}
    prompt = build_prompt(task, prompt_level)
    try:
        raw, finish_reason = call_local_llm(
            host, model, system_prompt, prompt, max_tokens, timeout=request_timeout,
            capacity_retries=capacity_retries, capacity_retry_delay=capacity_retry_delay,
            api_key=api_key)
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
    global DEBUG_LOG_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True, help="LM Studio server, e.g. http://192.168.1.122:1234")
    ap.add_argument("--model", default=None, help="Model id; auto-detected from /v1/models if omitted")
    ap.add_argument("--concurrency", type=int, default=4,
                     help="Matches the throughput sweet spot found for LM Studio in the mf_span "
                          "review loop -- 4 concurrent requests measurably raised tokens/sec there, "
                          "for a smaller model. Every call here reuses the exact same system prompt "
                          "(see call_local_llm's cache_prompt=True), which only pays off within a "
                          "given LM Studio parallel slot -- so this should match LM Studio's own "
                          "configured slot count, not exceed it, or extra requests just start cold "
                          "on a fresh slot instead of hitting cache. For a ~27B-class model (e.g. "
                          "Qwen3), each slot's KV cache is far larger than the smaller model this "
                          "default was tuned against, so check LM Studio's slot count/VRAM budget "
                          "before assuming 4 still fits -- fewer, cache-hitting slots can beat more, "
                          "cache-missing ones.")
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
                          "just adds more unbounded text for it to get lost in. Only valid with "
                          "--prompt-level full -- the simplified/minimal prompts scaffold no DAG "
                          "reasoning to think through.")
    ap.add_argument("--prompt-level", choices=["full", "simplified", "minimal"], default="full",
                     help="\"full\" (default): the current is_a-indexed, ancestor-aware prompt -- "
                          "use with the regular ancestor-expanded tasks.json. \"simplified\"/"
                          "\"minimal\": no is_a/specificity/sibling-ambiguity rules at all -- use "
                          "with a --leaf-only tasks file (build_mutagen_review_tasks.py "
                          "--leaf-only), where every candidate is already this protein's own real "
                          "annotation, so there's no DAG to reason about. Both keep the negation/"
                          "magnitude-word/specific-partner-exclusion/no-outside-knowledge rules in "
                          "full, since those fix basic text-comprehension gaps, not DAG complexity "
                          "-- \"minimal\" additionally compresses the sole-candidate and compound-"
                          "clause guidance down to one-liners, to test how far that can be cut "
                          "without losing correctness.")
    ap.add_argument("--tasks-path", type=Path, default=TASKS_PATH,
                     help="Override the input tasks file -- e.g. mutagen_review_tasks_leaf.json "
                          "for a --leaf-only build, so a leaf-only benchmarking run doesn't need "
                          "the default ancestor-expanded mutagen_review_tasks.json.")
    ap.add_argument("--capacity-retries", type=int, default=3)
    ap.add_argument("--capacity-retry-delay", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=None, help="Cap total tasks processed (smoke test)")
    ap.add_argument("--results-path", type=Path, default=RESULTS_PATH,
                     help="Override the output file -- use a distinct path per independent run "
                          "(e.g. mutagen_review_results_run1.jsonl / _run2.jsonl) so multiple full "
                          "passes over the same tasks can be kept separate and reconciled later, "
                          "instead of each run resuming/overwriting into the same file.")
    ap.add_argument("--debug-log-path", type=Path, default=DEBUG_LOG_PATH,
                     help="Override the raw-response debug log path; defaults alongside --results-path's constant.")
    ap.add_argument("--api-key", default="lm-studio",
                     help="Bearer token for the server's Authorization header. LM Studio ignores "
                          "this (any value works), but some other OpenAI-compatible local servers "
                          "enforce a real token even for local/LAN requests -- set it here instead "
                          "of hardcoding one.")
    ap.add_argument("--print-id", type=int, default=None,
                     help="Print task --print-id's exact prompt (system + user, as the model would "
                          "see it) and its recorded result if any, then exit without contacting any "
                          "model -- for manually checking why UniProt itself doesn't already flag "
                          "the loss this way.")
    args = ap.parse_args()
    if args.chain_of_thought and args.prompt_level != "full":
        ap.error("--chain-of-thought requires --prompt-level full")
    DEBUG_LOG_PATH = args.debug_log_path

    tasks = json.loads(args.tasks_path.read_text())

    if args.print_id is not None:
        task = next((t for t in tasks if t["id"] == args.print_id), None)
        if task is None:
            ap.error(f"id {args.print_id} not found in {args.tasks_path}")
        print("=== SYSTEM PROMPT ===")
        print(build_system_prompt(args.prompt_level, args.chain_of_thought))
        print("\n=== USER PROMPT ===")
        print(build_prompt(task, args.prompt_level))
        result = load_done(args.results_path).get(args.print_id)
        print("\n=== RESULT ===")
        print(json.dumps(result, indent=2) if result else "(no recorded result)")
        return

    if args.limit:
        tasks = tasks[:args.limit]
    done = load_done(args.results_path)

    def is_resolved(task_id):
        result = done.get(task_id)
        return result is not None and result["status"] not in ("UNPARSED", "INVALID_VERDICT")

    todo = [t for t in tasks if not is_resolved(t["id"])]
    print(f"[INFO] {len(tasks)} total tasks, {len(todo)} to process "
          f"({len(tasks) - len(todo)} already resolved)")

    model = args.model or discover_model(args.host, api_key=args.api_key)
    system_prompt = build_system_prompt(args.prompt_level, args.chain_of_thought)
    print(f"[INFO] Using model: {model} @ {args.host}, prompt-level={args.prompt_level}"
          + (" (chain-of-thought enabled)" if args.chain_of_thought else ""))

    n_done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(review_task, args.host, model, system_prompt, t,
                                    args.request_timeout, args.max_tokens,
                                    args.capacity_retries, args.capacity_retry_delay, args.api_key,
                                    args.prompt_level)
                   for t in todo]
        for fut in concurrent.futures.as_completed(futures):
            result = fut.result()
            with _write_lock:
                with open(args.results_path, "a") as f:
                    f.write(json.dumps(result) + "\n")
            n_done += 1
            if n_done % 50 == 0:
                print(f"[INFO] {n_done}/{len(todo)} processed", flush=True)

    print(f"[INFO] Done. {n_done} tasks processed this run -> {args.results_path}")
    if DEBUG_LOG_PATH.exists():
        print(f"[INFO] Some responses didn't parse cleanly -- raw text saved to {DEBUG_LOG_PATH} "
              f"for inspection. Those tasks are UNPARSED/INVALID_VERDICT in the results file and "
              f"will be retried automatically next run.")


if __name__ == "__main__":
    main()
