"""Markdown table of repro/sweep_conv_native.sh results (PROBE lines): one row per (x, w) shape: the strides (along W; "sH,sW"
otherwise) at which the native op ran with relative errors < 1e-3 against CPU float64, and those at which it failed graph
compile, gave wrong values ("bad") or died ("died": no output, in this sweep "Synapse detected a device critical error").
    python3 repro/sweep_conv_native_table.py repro/sweep_conv_native_results.jsonl"""
import collections
import json
import sys

cells = collections.defaultdict(dict)
strides = set()
for line in open(sys.argv[1]):
    if not line.startswith("PROBE {"):
        continue
    d = json.loads(line[6:])
    if "x" not in d:  # the process died (timeout or a device critical error): rebuild the case from its arguments
        v = [int(a) for a in d["case"].split()[:10]]
        d.update(x=v[0:4], w=[v[4], v[1], v[5], v[6]], stride=v[7:9], xgrad=len(v) < 10 or v[9] != 0,
                 lazy=d["lazy"] not in ("0", False), stage="died")
    row = (tuple(d["x"]), tuple(d["w"]), d["xgrad"])
    s = tuple(d["stride"])
    strides.add(s)
    mode = "L" if d["lazy"] else "E"
    cells[row].setdefault(s, {})[mode] = "ok" if d["ok"] else {"bwd": "bad", "died": "died"}.get(d.get("stage"), "fail")

def fmt(s):
    return str(s[1]) if s[0] == 1 else f"{s[0]},{s[1]}"


print("| x (B, C, H, W) | w (O, C, kH, kW) | eager runs | eager fails | lazy runs | lazy fails |")
print("|---|---|---|---|---|---|")
for (x, w, xgrad), by_stride in sorted(cells.items(), key=lambda kv: (kv[0][1][1], kv[0][1][2] > kv[0][1][3],
                                                                       kv[0][1][3], kv[0][0][3], not kv[0][2])):
    cols = []
    for mode in "EL":
        ran = [fmt(s) for s in sorted(by_stride) if by_stride[s].get(mode) == "ok"]
        bad = [fmt(s) + ("" if by_stride[s][mode] == "fail" else f" ({by_stride[s][mode]})")
               for s in sorted(by_stride) if by_stride[s].get(mode, "ok") != "ok"]
        cols += [", ".join(ran), ", ".join(bad)]
    tag = "" if xgrad else " (no x grad)"
    print(f"| {', '.join(map(str, x))}{tag} | {', '.join(map(str, w))} | " + " | ".join(cols) + " |")
