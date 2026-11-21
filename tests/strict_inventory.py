"""The CPU-execution inventory from ``strict_sweep.py``'s rows: one markdown table, sorted by impact.
    python3 tests/strict_inventory.py SWEEP_DIR [FIXES.json] > table.md
An entry is a (source, op) pair; ``cpu op`` rows are grouped by the caller line (one library line makes several
ops). Impact = the sum over (model, mode) of the count per step, training steps and inference forwards together;
the table also gives the models touched per mode. FIXES.json maps an entry key ``source|op`` to its proposed route."""
import collections
import glob
import json
import os
import sys

MODES = ("eager", "lazy", "compile")


def load(directory):
    rows = []
    for path in sorted(glob.glob(os.path.join(directory, "*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r.get("status") != "started":
                rows.append(r)
    return rows


BY_CALLER = ("cpu op", "host sync", "host copy")


def key(row):
    if row["source"] in BY_CALLER:  # one generic op from many places: the place is the entry
        return row["source"], row["where"]
    return row["source"], row["op"]


def collect(results):
    entries = collections.defaultdict(lambda: dict(impact=0.0, models={m: set() for m in MODES}, infer=set(),
                                                   per_step={}, where=collections.Counter(), tensors=set(),
                                                   ops=set(), families=set()))
    for r in results:
        for part in ("step_rows", "infer_rows", "trace_rows"):
            for row in r.get(part) or []:
                k = key(row)
                e = entries[k]
                n = row["per_step"]
                if part != "trace_rows":
                    e["impact"] += n
                if part == "infer_rows":
                    e["infer"].add(r["model"])
                elif part == "step_rows" or part == "trace_rows":
                    e["models"][r["mode"]].add(r["model"])
                if part == "step_rows":
                    e["per_step"].setdefault(r["mode"], []).append(n)
                e["where"][row["where"]] += 1
                e["tensors"].add(row["tensors"])
                e["ops"].add(row["op"])
                e["families"].add(r["family"])
    return entries


def status_table(results):
    table = collections.defaultdict(dict)
    for r in results:
        table[(r["family"], r["model"])][r["mode"]] = r
    return table


def main():
    results = load(sys.argv[1])
    fixes = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else {}
    entries = collect(results)
    out = ["| # | source | CPU op | impact | models eager / lazy / compile / inference | count per step "
           "(max) | example caller | proposed fix |", "|---|---|---|---|---|---|---|---|"]
    for i, (k, e) in enumerate(sorted(entries.items(), key=lambda kv: -kv[1]["impact"]), 1):
        counts = " / ".join(str(len(e["models"][m])) for m in MODES) + f" / {len(e['infer'])}"
        peak = max((max(v) for v in e["per_step"].values()), default=0)
        where = k[1] if k[0] in BY_CALLER else (e["where"].most_common(1)[0][0] if e["where"] else "-")
        op = k[1] if k[0] not in BY_CALLER else ", ".join(sorted(e["ops"]))[:120]
        fix = fixes.get(f"{k[0]}|{k[1]}", "")
        out.append(f"| {i} | {k[0]} | `{op}` | {e['impact']:g} | {counts} | {peak:g} | `{where}` | {fix} |")
    print("\n".join(out))
    print()
    print("| family | model | " + " | ".join(f"{m} (ms/step, status)" for m in MODES) + " |")
    print("|---|---|" + "---|" * len(MODES))
    for (family, model), modes in sorted(status_table(results).items()):
        cells = []
        for m in MODES:
            r = modes.get(m)
            if r is None:
                cells.append("not run")
            elif r["status"] == "ok":
                cells.append(f"{r['ms_per_step']}")
            else:
                cells.append(f"{r['status']}: {r.get('stage', '')} {r.get('error', '')[:70]}".replace("|", "/"))
        print(f"| {family} | {model} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
