"""Run the local model against the small Opus-confirmed eval set
(experiments/llm_review_eval_set.json) -- the exact same proteins Haiku reviewed and Opus
independently verified -- and write everything to a dedicated, separate output location for easy
comparison. This never touches experiments/llm_term_review/ (production review state): no merge,
no dictionary write, no already_started() footprint, no raw_debug.log dropped into a production
term directory.

Evidence text is rebuilt directly from positional_mf_raw_features.json (via
prepare_term_review_files.py's own render_review_file/render_protein_section, for byte-identical
formatting to what the production pipeline generates) rather than read back from a term's existing
review.md on disk. This is deliberate, not just for isolation: GO:0000156's on-disk review.md
turned out to list 10 entirely different proteins than the 3 (P0AE88/P0AFB8/P23620) actually
reviewed in its findings.md and recorded in the eval set -- a leftover from the pipeline's
pre-restructuring, one-file-per-protein era (that term's assigned_accessions is empty, a tell that
review.md was regenerated after the fact without ever being reconciled with the original findings).
Rebuilding straight from raw_features.json for exactly the eval set's own protein list sidesteps
that kind of drift entirely: it can never show a protein the eval set doesn't actually claim ground
truth for.

For each protein it also resolves the model's reported feature back to a real annotation record
(same lookup parse_llm_findings.py uses) and checks whether it matches the ground-truth signal
recorded in llm_review_eval_set.json -- via merge_llm_findings.py's own domain-accession-bridging
logic for DOMAIN/text signals, so "does this match" uses the identical resolution the production
merge step would use, not a separate ad hoc comparison.

Usage:
  python3 experiments/scripts/run_local_llm_eval.py --host http://192.168.1.122:1234 \
      --chain-of-thought "Let's think step by step."

Output, under experiments/llm_eval_runs/<run_name>/ (run_name defaults to a timestamp):
  <go_id>_findings.md   -- raw per-protein VERDICT/FEATURE/REASONING blocks, same format as the
                           main pipeline, for manual read-through.
  raw_debug.log         -- unparsed-response dumps, if any (see local_llm_review_loop.py).
  report.json           -- structured per-protein match/no-match verdict against ground truth.
  summary.md            -- human-readable table, the fastest way to eyeball pass/fail.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import local_llm_review_loop as loop
import merge_llm_findings as mlf
import parse_llm_findings as plf
import prepare_term_review_files as ptr

EVAL_SET_PATH = Path("experiments/llm_review_eval_set.json")
RAW_FEATURES_PATH = Path("experiments/positional_mf_raw_features.json")
OUT_ROOT = Path("experiments/llm_eval_runs")


def check_match(raw_features, accession, ftype, start, end, expected):
  """Resolve the model's reported (ftype, start, end) to a real qualifier value, then check it
  against the ground-truth expected signal using the same resolution merge_llm_findings.py would
  use. Returns (matched: bool, detail: str)."""
  match, err = plf.resolve_pointer(raw_features, accession, ftype, start, end)
  if err:
    return False, f"pointer didn't resolve: {err}"
  kind, value = plf.extract_signal_value(match)
  field = expected["field"]
  exp_value = expected["value"]

  if field == "domain_accessions":
    if not (kind == "text" and ftype == "DOMAIN"):
      return False, f"expected a DOMAIN note, model pointed at {ftype} ({kind}={value!r})"
    resolved = mlf.resolve_domain_accessions(raw_features, [accession], value)
    overlap = sorted(set(resolved) & set(exp_value))
    if overlap:
      return True, f"DOMAIN note {value!r} resolved to {resolved}, overlaps expected {exp_value}"
    return False, f"DOMAIN note {value!r} resolved to {resolved}, expected {exp_value} -- no overlap"

  if field == "ligands":
    if kind != "ligand":
      return False, f"expected a ligand, model pointed at {ftype} ({kind}={value!r})"
    matched = value.strip().lower() == str(exp_value).strip().lower()
    return matched, f"ligand {value!r} vs expected {exp_value!r}"

  if field == "text_patterns":
    if kind != "text":
      return False, f"expected a text note, model pointed at {ftype} ({kind}={value!r})"
    norm_v, norm_e = mlf.normalize(value), mlf.normalize(str(exp_value))
    matched = norm_e <= norm_v
    return matched, f"note {value!r} vs expected {exp_value!r}"

  return False, f"unhandled expected field {field!r}"


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--host", required=True, help="LM Studio server, e.g. http://192.168.1.122:1234")
  ap.add_argument("--model", default=None, help="Model id; auto-detected from /v1/models if omitted")
  ap.add_argument("--request-timeout", type=int, default=180)
  ap.add_argument("--chain-of-thought", type=str, default="",
                   help="Same flag as local_llm_review_loop.py -- should match whatever setup "
                        "you're evaluating.")
  ap.add_argument("--max-tokens", type=int, default=1536)
  ap.add_argument("--run-name", default=None, help="Output subdir name; default is a timestamp")
  args = ap.parse_args()

  model = args.model or loop.discover_model(args.host)
  system_prompt = loop.build_system_prompt(args.chain_of_thought)
  print(f"Using model: {model} @ {args.host}"
        + (" (chain-of-thought enabled)" if args.chain_of_thought else ""))

  eval_set = json.loads(EVAL_SET_PATH.read_text())
  raw_features = json.loads(RAW_FEATURES_PATH.read_text())

  run_name = args.run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"
  out_dir = OUT_ROOT / run_name
  out_dir.mkdir(parents=True, exist_ok=True)
  print(f"Output directory: {out_dir}")

  report = {
      "model": model, "host": args.host, "chain_of_thought": args.chain_of_thought,
      "max_tokens": args.max_tokens, "terms": {},
  }
  total, matched_count = 0, 0

  for go_id, term in eval_set["terms"].items():
    print(f"\n=== {go_id} ({term['name']}) ===")
    go_info = {"name": term["name"], "def": term["definition"]}
    review_text = ptr.render_review_file(go_id, go_info, term["proteins"], raw_features)
    findings_path = out_dir / f"{go_id.replace(':', '_')}_findings.md"
    expected = term["expected_signal"]

    protein_results = {}
    for accession in term["proteins"]:
      verdict, ftype, start, end, reasoning, unparsed = loop.review_protein(
          args.host, model, system_prompt, go_id, accession, review_text,
          out_dir, findings_path, args.request_timeout, args.max_tokens)
      total += 1
      matched, detail = (False, "verdict was not LIKELY")
      if verdict == "LIKELY" and not unparsed:
        matched, detail = check_match(raw_features, accession, ftype, int(start), int(end), expected)
      if matched:
        matched_count += 1
      protein_results[accession] = {
          "verdict": verdict, "feature": f"{ftype} {start}-{end}", "reasoning": reasoning,
          "matched_ground_truth": matched, "match_detail": detail,
      }
      print(f"  {accession}: {verdict} ({ftype} {start}-{end}) -> "
            f"{'MATCH' if matched else 'no match'} ({detail})")

    n_matched = sum(1 for r in protein_results.values() if r["matched_ground_truth"])
    report["terms"][go_id] = {
        "name": term["name"], "expected_signal": expected,
        "proteins": protein_results,
        "term_summary": f"{n_matched}/{len(term['proteins'])} proteins matched ground truth",
    }

  report["overall_summary"] = f"{matched_count}/{total} protein reviews matched ground truth"
  (out_dir / "report.json").write_text(json.dumps(report, indent=2))

  summary_lines = [
      f"# Local-model eval run: {run_name}", "",
      f"Model: `{model}` @ {args.host}  ",
      f"Chain-of-thought: {args.chain_of_thought or '(off)'}  ",
      f"Overall: **{report['overall_summary']}**", "",
      "| GO term | Name | Protein | Verdict | Feature | Match | Detail |",
      "|---|---|---|---|---|---|---|",
  ]
  for go_id, t in report["terms"].items():
    for acc, r in t["proteins"].items():
      summary_lines.append(
          f"| {go_id} | {t['name']} | {acc} | {r['verdict']} | {r['feature']} | "
          f"{'✓' if r['matched_ground_truth'] else '✗'} | {r['match_detail']} |")
  (out_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")

  print(f"\n{report['overall_summary']}")
  print(f"-> {out_dir}/report.json, summary.md, <go_id>_findings.md")


if __name__ == "__main__":
  main()
