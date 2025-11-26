#!/usr/bin/env python3
import argparse, json, os
from typing import List, Dict, Any
from huggingface_hub import HfApi, HfFolder


def parse_args():
    p = argparse.ArgumentParser(description="Merge shard inference results (JSON/JSONL) and recompute metrics")
    p.add_argument("--inputs", nargs="+", default=None, help="Explicit list of shard result files (.json preferred; .jsonl also supported)")
    p.add_argument("--dir", default=None, help="Directory to auto-discover shard files when --inputs is omitted")
    p.add_argument("--pattern", default="*.shard*.json", help="Glob pattern within --dir (auto-add .jsonl peers)")
    p.add_argument("--output", required=True, help="Path to combined JSON")
    p.add_argument("--model_repo", default=None)
    p.add_argument("--dataset_repo", default=None)
    p.add_argument("--push_hf", action="store_true")
    p.add_argument("--hf_out_repo", default=None)
    p.add_argument("--hf_out_file", default=None)
    return p.parse_args()


def recompute_metrics(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    parsed = [r for r in items if r is not None]
    n = len(parsed)
    n_gt = sum(1 for r in parsed if r.get("gt"))
    n_parsed = sum(1 for r in parsed if r.get("pred_order"))
    top1_acc = (sum(1 for r in parsed if r.get("top1_correct")) / n_gt) if n_gt else 0.0
    recall5 = (sum(1 for r in parsed if r.get("gt") and r.get("gt_pos", 0) > 0) / n_gt) if n_gt else 0.0
    mrr = 0.0
    cnt = 0
    for r in parsed:
        pos = r.get("gt_pos", 0)
        if pos > 0:
            mrr += 1.0 / float(pos)
            cnt += 1
    mrr = (mrr / cnt) if cnt else 0.0
    return {
        "total": n,
        "with_gt": n_gt,
        "with_parsed_answer": n_parsed,
        "top1_acc": top1_acc,
        "recall_at_5": recall5,
        "mrr": mrr,
    }


def main():
    args = parse_args()
    items: List[Dict[str, Any]] = []
    model = args.model_repo
    dataset = args.dataset_repo
    def _load_items_from_json(path: str) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj.get("items") or []

    def _load_items_from_jsonl(path: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    pass
        return rows

    # Resolve inputs: explicit list or auto-discover from dir/pattern
    inputs: List[str]
    if args.inputs:
        inputs = list(args.inputs)
    else:
        if not args.dir:
            raise SystemExit("Provide --inputs or --dir for auto-discovery")
        # Gather both .json and .jsonl candidates, deduplicate by stem, prefer .json
        base_dir = os.path.abspath(args.dir)
        patt = os.path.join(base_dir, args.pattern)
        files = set(glob.glob(patt))
        # Add jsonl variant of the pattern when applicable
        if args.pattern.endswith(".json"):
            files.update(glob.glob(os.path.join(base_dir, args.pattern[:-5] + ".jsonl")))
        elif args.pattern.endswith(".jsonl"):
            files.update(glob.glob(os.path.join(base_dir, args.pattern[:-6] + ".json")))
        else:
            files.update(glob.glob(os.path.join(base_dir, "*.jsonl")))
            files.update(glob.glob(os.path.join(base_dir, "*.json")))

        # Deduplicate by stem; prefer .json if both exist
        by_stem: Dict[str, str] = {}
        for fp in files:
            stem, ext = os.path.splitext(fp)
            prev = by_stem.get(stem)
            if not prev:
                by_stem[stem] = fp
            else:
                # Prefer .json over .jsonl
                if prev.endswith(".jsonl") and fp.endswith(".json"):
                    by_stem[stem] = fp
        inputs = sorted(by_stem.values())

    loaded = 0
    for p in inputs:
        candidate_paths = []
        # primary: JSON path as given
        candidate_paths.append((p, "json"))
        # fallback: same stem with .jsonl
        if p.endswith(".json"):
            candidate_paths.append((p[:-5] + ".jsonl", "jsonl"))
        elif p.endswith(".jsonl"):
            candidate_paths.append((p, "jsonl"))

        loaded_any = False
        for cp, kind in candidate_paths:
            if not os.path.isfile(cp) or os.path.getsize(cp) == 0:
                continue
            try:
                if kind == "json":
                    shard_items = _load_items_from_json(cp)
                else:
                    shard_items = _load_items_from_jsonl(cp)
            except Exception as e:
                print(f"[warn] failed to read {cp}: {e}")
                continue

            if kind == "json":
                # Only JSON contains model/dataset metadata
                try:
                    with open(cp, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                    if model is None:
                        model = obj.get("model")
                    if dataset is None:
                        dataset = obj.get("dataset")
                except Exception:
                    pass

            items.extend(shard_items)
            loaded += 1
            loaded_any = True
            break

        if not loaded_any:
            print(f"[warn] missing or empty: {p} (and no jsonl fallback) — skipping")

    if loaded == 0:
        raise SystemExit("No shard files could be loaded. Nothing to merge.")

    metrics = recompute_metrics(items)
    out = {"model": model, "dataset": dataset, "items": items, "metrics": metrics}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("Merged", len(args.inputs), "files →", args.output)
    print("Metrics:", json.dumps(metrics, indent=2))

    if args.push_hf:
        repo = args.hf_out_repo
        out_file = args.hf_out_file or os.path.basename(args.output)
        if not repo:
            # infer namespace from token
            try:
                api = HfApi()
                token = HfFolder.get_token()
                who = api.whoami(token) if token else {}
                user = who.get("name") or who.get("username")
                if not user:
                    print("[warn] No HF user; skip push.")
                    return
                repo = f"{user}/{os.path.splitext(os.path.basename(args.output))[0]}"
            except Exception:
                print("[warn] Could not resolve HF user; skip push.")
                return
        api = HfApi()
        api.create_repo(repo_id=repo, repo_type="dataset", exist_ok=True)
        api.upload_file(
            path_or_fileobj=args.output,
            path_in_repo=out_file,
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"Upload merged results ({len(args.inputs)} shards)",
        )
        print(f"Pushed to https://huggingface.co/datasets/{repo}")


if __name__ == "__main__":
    main()
