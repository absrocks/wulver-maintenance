#!/usr/bin/env python3
"""
Collect Wulver maintenance benchmark results into a normalized JSON layout
that index.html reads.

Usage:
    python3 collect_maint.py --raw <mirror of /project/hpcadmins/am2455/maint/2026> --out data

Input layout (per maintenance month, any nesting):
    gromacs*/.../gmx_performance*.json | gmx_performance*.csv
    hpcg*/hpcg_gflops_results.json
    nccl*/nccl_performance*.json  (+ nccl<job>_<node>_<task>.err for node mapping)
                                  or raw nccl*.out files
    torch*/torch<job>_<node>_<task>.out        (torch-comm.py output)
    nvlink*/nvlink<job>_<node>_<task>.out      (nvbandwidth output -> uses nvlink_results.json if present)
    p2p*/nvlink<job>_<node>_<task>.out         (CUDA p2pBandwidthLatencyTest output)

Output layout:
    data/index.json                      list of periods for the dropdown
    data/<period>/manifest.json          tests available in that period
    data/<period>/<test>.json            {"test":..., "metrics":[...], "rows":[{"node":..., ...}]}
"""
import argparse, csv, glob, json, os, re, statistics, sys
from collections import OrderedDict

PERIODS = [
    # folder,        id,            label,                       date
    ("Jan-maint",   "2026-01-13", "Jan 13, 2026 — Scheduled",      "2026-01-13"),
    ("March-maint", "2026-03-10", "Mar 10, 2026 — Scheduled",      "2026-03-10"),
    ("Apr-maint",   "2026-04-14", "Apr 14, 2026 — Scheduled",      "2026-04-14"),
    ("May-maint",   "2026-05-12", "May 12, 2026 — Scheduled",      "2026-05-12"),
    ("Jun-maint",   "2026-06-09", "Jun 9, 2026 — Scheduled",       "2026-06-09"),
    ("July-maint",  "2026-07-14", "Jul 14, 2026 — Scheduled",      "2026-07-14"),
    ("Aug-maint",   "2026-08-11", "Aug 11, 2026 — Scheduled",      "2026-08-11"),
    ("Sep",         "2026-09-08", "Sep 8, 2026 — Scheduled",       "2026-09-08"),
    ("Sep-Slurm",   "2026-09-22", "Sep 22, 2026 — Emergency (Slurm)", "2026-09-22"),
    ("Oct-maint",   "2026-10-13", "Oct 13, 2026 — Scheduled",      "2026-10-13"),
]

NODE_RE = re.compile(r"_(n\d{4})_(\d+)\.(out|err)$")
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def node_key(n):
    m = re.match(r"n(\d+)", n or "")
    return int(m.group(1)) if m else 10**9


def gpu_type_from_path(rel):
    """Return a short variant label like a100 / l40 / mig_10g / cpu."""
    low = rel.lower()
    m = re.search(r"mig[/_](\d+g)", low)
    if m:
        return f"mig_{m.group(1)}"
    if "l40" in low:
        return "l40"
    if "a100" in low or "gpu" in low:
        return "a100"
    return "a100"


# ---------------------------------------------------------------- GROMACS
def gromacs_from_err(dirpath):
    """node -> ns/day parsed from GROMACS stderr logs (latest job id wins)."""
    found = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "gromacs*_n*_*.err"))):
        m = NODE_RE.search(f)
        if not m:
            continue
        txt = open(f, errors="replace").read()
        pm = re.findall(r"Performance:\s+(" + NUM + r")\s+(" + NUM + ")", txt)
        if pm:
            found[m.group(1)] = (float(pm[-1][0]), float(pm[-1][1]))
    return found


def load_gromacs(path):
    rows = []
    if path.endswith(".json"):
        for r in json.load(open(path)):
            rows.append({"node": r["node"],
                         "ns_per_day": float(r["performance_ns_per_day"]),
                         "hour_per_ns": float(r["hour_per_ns"])})
    else:
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                rows.append({"node": r["Node Number"],
                             "ns_per_day": float(r["Performance (ns/day)"]),
                             "hour_per_ns": float(r["Hour/ns"])})
    rows = [r for r in rows if re.match(r"^n\d+$", str(r["node"]))]  # drop malformed node names
    have = {r["node"] for r in rows}
    for node, (ns, hpn) in gromacs_from_err(os.path.dirname(path)).items():
        if node not in have:
            rows.append({"node": node, "ns_per_day": ns, "hour_per_ns": hpn, "from_log": True})
    return rows


# ---------------------------------------------------------------- HPCG
def load_hpcg(path):
    return [{"node": r["Node Number"], "gflops": float(r["GFLOPS"])}
            for r in json.load(open(path))]


# ---------------------------------------------------------------- NCCL
def task_to_node_map(dirpath):
    m = {}
    for f in os.listdir(dirpath):
        mm = NODE_RE.search(f)
        if mm:
            m[mm.group(2)] = mm.group(1)
    return m


def parse_nccl_out(path):
    avg = None
    node = None
    rows = []
    for line in open(path, errors="replace"):
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            m = re.search(r"Avg bus bandwidth\s*:\s*(" + NUM + ")", s)
            if m:
                avg = float(m.group(1))
            m = re.search(r"Rank\s+0\s+.*?\bon\s+(n\d+)\b", s)
            if m:
                node = m.group(1)
            continue
        p = s.split()
        if len(p) < 13:
            continue
        try:
            size = int(p[0])
        except ValueError:
            continue
        rows.append({"size": size,
                     "oop": {"time": float(p[5]), "algbw": float(p[6]), "busbw": float(p[7]), "wrong": p[8]},
                     "ip": {"time": float(p[9]), "algbw": float(p[10]), "busbw": float(p[11]), "wrong": p[12]}})
    return node, avg, rows


def load_nccl_dir(dirpath):
    t2n = task_to_node_map(dirpath)
    jsons = glob.glob(os.path.join(dirpath, "nccl_performance*.json"))
    rows = []
    if jsons:
        for r in json.load(open(jsons[0])):
            tid = str(r.get("task_id"))
            res = r.get("results") or []
            by_size = {int(x["size_bytes"]): x for x in res}
            big = by_size.get(1073741824) or (res[-1] if res else None)
            mid = by_size.get(536870912)
            wrong = sum(int(x["out_of_place"]["wrong"]) + int(x["in_place"]["wrong"]) for x in res) if res else None
            rows.append({
                "node": t2n.get(tid, f"task{tid}"),
                "task_id": tid,
                "avg_busbw": r.get("avg_bus_bandwidth_gbps"),
                "busbw_1g": big["in_place"]["busbw_gbps"] if big else None,
                "algbw_1g": big["in_place"]["algbw_gbps"] if big else None,
                "time_us_1g": big["in_place"]["time_us"] if big else None,
                "busbw_512m": mid["in_place"]["busbw_gbps"] if mid else None,
                "wrong": wrong,
            })
    else:
        for f in sorted(glob.glob(os.path.join(dirpath, "nccl*.out"))):
            tid = re.search(r"_(\d+)\.out$", f).group(1)
            node, avg, res = parse_nccl_out(f)
            if not res and avg is None:
                continue
            big = next((x for x in res if x["size"] == 1073741824), res[-1] if res else None)
            mid = next((x for x in res if x["size"] == 536870912), None)
            rows.append({
                "node": t2n.get(tid) or node or f"task{tid}",
                "task_id": tid,
                "avg_busbw": avg,
                "busbw_1g": big["ip"]["busbw"] if big else None,
                "algbw_1g": big["ip"]["algbw"] if big else None,
                "time_us_1g": big["ip"]["time"] if big else None,
                "busbw_512m": mid["ip"]["busbw"] if mid else None,
                "wrong": sum(int(x["oop"]["wrong"]) + int(x["ip"]["wrong"]) for x in res) if res else None,
            })
    return rows


# ---------------------------------------------------------------- torch-comm
PAIR_RE = re.compile(r"cuda:(\d+)\s*[↔<>-]+\s*cuda:(\d+)")


def parse_torch_out(path):
    txt = open(path, errors="replace").read()
    node = None
    m = NODE_RE.search(os.path.basename(path))
    if m:
        node = m.group(1)
    sections = re.split(r"═{10,}\n\s*(.+?)\n═{10,}", txt)
    # sections: [pre, title1, body1, title2, body2, ...]
    sec = {}
    for i in range(1, len(sections) - 1, 2):
        sec[sections[i].strip()] = sections[i + 1]

    gpu_name = None
    mm = re.search(r"cuda:0\s+\(physical GPU \d+\)\s+(.+)", txt)
    if mm:
        gpu_name = mm.group(1).strip()
    n_gpus = len(re.findall(r"^\s*cuda:\d+\s+\(physical GPU", txt, re.M))

    pairs = OrderedDict()
    # Test 1 / Summary: fwd/bwd + status
    summ = next((v for k, v in sec.items() if k.startswith("Summary")), "")
    for line in summ.splitlines():
        pm = PAIR_RE.search(line)
        if not pm:
            continue
        rest = line.split("cuda:" + pm.group(2), 1)[1]
        rest = re.sub(r"[A-Za-z/]+\d*[A-Za-z/]*", " ", rest.split("✓")[0].split("✗")[0])  # drop mode word e.g. NVLink/P2P
        nums = re.findall(NUM, rest)
        status = "PASS" if "PASS" in line else ("FAIL" if "FAIL" in line else None)
        mode = "NVLink/P2P" if "P2P" in line else ("PCIe" if "PCIe" in line or "memcpy" in line.lower() else None)
        key = f"{pm.group(1)}-{pm.group(2)}"
        pairs.setdefault(key, {})
        if len(nums) >= 2:
            pairs[key]["fwd"] = float(nums[0])
            pairs[key]["bwd"] = float(nums[1])
        pairs[key]["status"] = status
        pairs[key]["mode"] = mode
    if not pairs:  # fall back to Test 1 block
        t1 = next((v for k, v in sec.items() if k.startswith("Test 1")), "")
        lines = t1.splitlines()
        for i, line in enumerate(lines):
            pm = PAIR_RE.search(line)
            if pm and i + 1 < len(lines):
                nums = re.findall(NUM, lines[i + 1])
                key = f"{pm.group(1)}-{pm.group(2)}"
                pairs[key] = {"fwd": float(nums[0]) if nums else None,
                              "bwd": float(nums[1]) if len(nums) > 1 else None,
                              "status": "PASS" if "PASS" in lines[i + 1] else "FAIL" if "FAIL" in lines[i + 1] else None,
                              "mode": "NVLink/P2P" if "P2P" in line else None}
    # Test 2: bandwidth sweep -> take 1 GB row per pair
    t2 = next((v for k, v in sec.items() if k.startswith("Test 2")), "")
    cur = None
    for line in t2.splitlines():
        pm = PAIR_RE.search(line)
        if pm:
            cur = f"{pm.group(1)}-{pm.group(2)}"
            continue
        if cur and re.match(r"^\s*1 GB\s", line):
            nums = re.findall(NUM, line.replace("1 GB", ""))
            pairs.setdefault(cur, {})
            if len(nums) >= 2:
                pairs[cur]["sweep_1g_fwd"] = float(nums[0])
                pairs[cur]["sweep_1g_bwd"] = float(nums[1])
            pairs[cur]["sweep_ok"] = ("✓" in line) and ("✗" not in line)
    # Test 3: bidirectional concurrent
    t3 = next((v for k, v in sec.items() if k.startswith("Test 3")), "")
    for line in t3.splitlines():
        pm = PAIR_RE.search(line)
        if not pm:
            continue
        rest = line.split("cuda:" + pm.group(2), 1)[1]
        nums = re.findall(NUM, rest)
        key = f"{pm.group(1)}-{pm.group(2)}"
        pairs.setdefault(key, {})
        if nums:
            pairs[key]["bidir"] = float(nums[0])
        pairs[key]["bidir_ok"] = ("✓" in rest) and ("✗" not in rest)

    overall = None
    om = re.search(r"Overall\s*:\s*(.+)", txt)
    if om:
        overall = "PASS" if "ALL PASSED" in om.group(1) else "FAIL"
    if not pairs:
        return None

    fwds = [p["fwd"] for p in pairs.values() if p.get("fwd") is not None]
    bwds = [p["bwd"] for p in pairs.values() if p.get("bwd") is not None]
    s1f = [p["sweep_1g_fwd"] for p in pairs.values() if p.get("sweep_1g_fwd") is not None]
    s1b = [p["sweep_1g_bwd"] for p in pairs.values() if p.get("sweep_1g_bwd") is not None]
    bid = [p["bidir"] for p in pairs.values() if p.get("bidir") is not None]
    fails = [k for k, p in pairs.items() if p.get("status") == "FAIL"
             or p.get("sweep_ok") is False or p.get("bidir_ok") is False]
    return {
        "node": node,
        "gpu": gpu_name,
        "n_gpus": n_gpus,
        "overall": overall,
        "fail_pairs": len(fails),
        "p2p_fwd_min": min(fwds) if fwds else None,
        "p2p_fwd_mean": round(statistics.fmean(fwds), 2) if fwds else None,
        "p2p_bwd_min": min(bwds) if bwds else None,
        "p2p_bwd_mean": round(statistics.fmean(bwds), 2) if bwds else None,
        "sweep_1g_fwd_min": min(s1f) if s1f else None,
        "sweep_1g_bwd_min": min(s1b) if s1b else None,
        "bidir_min": min(bid) if bid else None,
        "bidir_max": max(bid) if bid else None,
        "pairs": pairs,
    }


# ---------------------------------------------------------------- p2pBandwidthLatencyTest
def parse_matrix(txt, header):
    """Return NxN float matrix following a header line."""
    i = txt.find(header)
    if i < 0:
        return None
    lines = txt[i:].splitlines()[1:]
    mat = []
    for line in lines:
        s = line.strip()
        if not s:
            if mat:
                break
            continue
        p = s.split()
        if p[0] in ("D\\D", "GPU", "CPU"):
            if mat:
                break
            continue
        if not re.match(r"^\d+$", p[0]):
            break
        try:
            mat.append([float(x) for x in p[1:]])
        except ValueError:
            break
    return mat or None


def offdiag(mat):
    return [mat[i][j] for i in range(len(mat)) for j in range(len(mat[i])) if i != j]


def parse_p2p_out(path):
    txt = open(path, errors="replace").read()
    if "P2P Connectivity Matrix" not in txt:
        return None
    m = NODE_RE.search(os.path.basename(path))
    node = m.group(1) if m else None
    uni = parse_matrix(txt, "Unidirectional P2P=Enabled Bandwidth (P2P Writes) Matrix (GB/s)")
    bi = parse_matrix(txt, "Bidirectional P2P=Enabled Bandwidth Matrix (GB/s)")
    uni_off = parse_matrix(txt, "Unidirectional P2P=Disabled Bandwidth Matrix (GB/s)")
    lat_sec = txt.split("P2P=Enabled Latency (P2P Writes) Matrix (us)", 1)
    lat = parse_matrix(lat_sec[1], "GPU") if len(lat_sec) > 1 else None
    conn = parse_matrix(txt, "P2P Connectivity Matrix")
    n_gpus = len(re.findall(r"^Device: \d+,", txt, re.M))
    row = {"node": node, "n_gpus": n_gpus}
    if conn:
        row["p2p_links"] = int(sum(offdiag(conn)))
        row["p2p_links_expected"] = len(conn) * (len(conn) - 1)
    if uni:
        o = offdiag(uni); row["uni_min"] = min(o); row["uni_mean"] = round(statistics.fmean(o), 2)
    if bi:
        o = offdiag(bi); row["bi_min"] = min(o); row["bi_mean"] = round(statistics.fmean(o), 2)
    if uni_off:
        o = offdiag(uni_off); row["uni_p2p_disabled_mean"] = round(statistics.fmean(o), 2)
    if lat:
        o = offdiag(lat); row["lat_max"] = max(o); row["lat_mean"] = round(statistics.fmean(o), 2)
    row["uni_matrix"] = uni
    row["bi_matrix"] = bi
    row["lat_matrix"] = lat
    return row


# ---------------------------------------------------------------- nvbandwidth (nvlink_results.json)
def load_nvbandwidth_json(path):
    rows = []
    for r in json.load(open(path)):
        row = {"node": r["node"]}
        tests = r.get("tests", {})
        errs = []
        for name, t in tests.items():
            if t.get("status") == "error" or "sum_bandwidth_gbps" not in t:
                errs.append(name)
                row[name] = None
            else:
                row[name] = t["sum_bandwidth_gbps"]
        row["errors"] = len(errs)
        row["error_tests"] = errs
        rows.append(row)
    return rows


# ---------------------------------------------------------------- driver
def add_test(period_tests, key, label, metrics, rows, variant=None, source=None, group=None):
    rows = [r for r in rows if r and r.get("node")]
    rows.sort(key=lambda r: node_key(r["node"]))
    if not rows:
        return
    period_tests[key] = {
        "test": key, "label": label, "group": group, "variant": variant,
        "metrics": metrics, "source": source, "rows": rows,
    }


GMX_METRICS = [
    {"key": "ns_per_day", "label": "Performance (ns/day)", "higher_is_better": True, "primary": True},
    {"key": "hour_per_ns", "label": "Hour / ns", "higher_is_better": False},
]
HPCG_METRICS = [{"key": "gflops", "label": "HPCG GFLOP/s", "higher_is_better": True, "primary": True}]
NCCL_METRICS = [
    {"key": "avg_busbw", "label": "Avg bus BW (GB/s)", "higher_is_better": True, "primary": True},
    {"key": "busbw_1g", "label": "Bus BW @1 GiB in-place (GB/s)", "higher_is_better": True},
    {"key": "algbw_1g", "label": "Alg BW @1 GiB in-place (GB/s)", "higher_is_better": True},
    {"key": "busbw_512m", "label": "Bus BW @512 MiB in-place (GB/s)", "higher_is_better": True},
    {"key": "time_us_1g", "label": "Time @1 GiB (µs)", "higher_is_better": False},
    {"key": "wrong", "label": "#wrong (sum)", "higher_is_better": False, "expect_zero": True},
]
TORCH_METRICS = [
    {"key": "sweep_1g_fwd_min", "label": "Sweep 1 GB → min over pairs (GB/s)", "higher_is_better": True, "primary": True},
    {"key": "sweep_1g_bwd_min", "label": "Sweep 1 GB ← min over pairs (GB/s)", "higher_is_better": True},
    {"key": "fail_pairs", "label": "Failed pairs", "higher_is_better": False, "expect_zero": True},
    {"key": "bidir_min", "label": "Bidirectional concurrent min (GB/s)", "higher_is_better": True},
    {"key": "bidir_max", "label": "Bidirectional concurrent max (GB/s)", "higher_is_better": True},
    {"key": "p2p_fwd_min", "label": "Test 1 P2P → min (GB/s)", "higher_is_better": True},
    {"key": "p2p_fwd_mean", "label": "Test 1 P2P → mean (GB/s)", "higher_is_better": True},
    {"key": "p2p_bwd_min", "label": "Test 1 P2P ← min (GB/s)", "higher_is_better": True},
    {"key": "p2p_bwd_mean", "label": "Test 1 P2P ← mean (GB/s)", "higher_is_better": True},
]
P2P_METRICS = [
    {"key": "uni_min", "label": "Unidir P2P min (GB/s)", "higher_is_better": True, "primary": True},
    {"key": "uni_mean", "label": "Unidir P2P mean (GB/s)", "higher_is_better": True},
    {"key": "bi_min", "label": "Bidir P2P min (GB/s)", "higher_is_better": True},
    {"key": "bi_mean", "label": "Bidir P2P mean (GB/s)", "higher_is_better": True},
    {"key": "uni_p2p_disabled_mean", "label": "Unidir P2P-disabled mean (GB/s)", "higher_is_better": True},
    {"key": "lat_max", "label": "P2P latency max (µs)", "higher_is_better": False},
    {"key": "lat_mean", "label": "P2P latency mean (µs)", "higher_is_better": False},
    {"key": "p2p_links", "label": "P2P links OK", "higher_is_better": True},
]


def nvb_metrics(rows):
    names = []
    for r in rows:
        for k in r:
            if k not in ("node", "errors", "error_tests") and k not in names:
                names.append(k)
    mets = [{"key": n, "label": n.replace("_", " ") + " (GB/s)", "higher_is_better": True, "primary": i == 0}
            for i, n in enumerate(names)]
    mets.append({"key": "errors", "label": "Tests with error", "higher_is_better": False, "expect_zero": True})
    return mets


def collect_period(folder, pid):
    tests = OrderedDict()
    for root, dirs, files in os.walk(folder):
        rel = os.path.relpath(root, folder)
        low = rel.lower()
        # skip per-run gromacs dirs and hpcg node dirs
        if re.search(r"(^|/)\d+$", rel) or re.search(r"(^|/)n\d{4}_\d+$", rel):
            continue
        fjson = [f for f in files if f.endswith(".json")]
        fcsv = [f for f in files if f.endswith(".csv")]
        fout = [f for f in files if f.endswith(".out")]

        # GROMACS
        gj = [f for f in fjson if f.startswith("gmx_performance")]
        gc = [f for f in fcsv if f.startswith("gmx_performance")]
        if gj or gc:
            src = os.path.join(root, (gj or gc)[0])
            variant = gpu_type_from_path(rel)
            if "mig" in low and not re.search(r"mig[/_]\d+g", low):
                variant = "mig_mixed"
            key = f"gromacs_{variant}"
            if key in tests:  # keep the larger set
                if len(tests[key]["rows"]) >= len(load_gromacs(src)):
                    continue
            add_test(tests, key, f"GROMACS ({variant.replace('_', ' ').upper()})", GMX_METRICS,
                     load_gromacs(src), variant=variant, source=os.path.relpath(src, folder), group="GROMACS")
            continue

        # HPCG
        hj = [f for f in fjson if f.startswith("hpcg_gflops")]
        if hj:
            src = os.path.join(root, hj[0])
            add_test(tests, "hpcg_cpu", "HPCG (CPU nodes)", HPCG_METRICS, load_hpcg(src),
                     variant="cpu", source=os.path.relpath(src, folder), group="HPCG")
            continue

        # NCCL
        if "nccl" in low and (any(f.startswith("nccl_performance") for f in fjson) or any(f.startswith("nccl") for f in fout)):
            variant = gpu_type_from_path(rel)
            rows = load_nccl_dir(root)
            add_test(tests, f"nccl_{variant}", f"NCCL all_reduce ({variant.replace('_', ' ').upper()})", NCCL_METRICS,
                     rows, variant=variant, source=rel, group="NCCL")
            continue

        # torch-comm
        tout = [f for f in fout if f.startswith("torch")]
        if tout:
            variant = gpu_type_from_path(rel)
            rows = [parse_torch_out(os.path.join(root, f)) for f in sorted(tout)]
            add_test(tests, f"torch_{variant}", f"PyTorch P2P comm ({variant.replace('_', ' ').upper()})", TORCH_METRICS,
                     rows, variant=variant, source=rel, group="PyTorch")
            continue

        # nvbandwidth json
        nj = [f for f in fjson if f.startswith("nvlink_results")]
        if nj:
            src = os.path.join(root, nj[0])
            variant = gpu_type_from_path(rel)
            rows = load_nvbandwidth_json(src)
            add_test(tests, f"nvbandwidth_{variant}", f"nvbandwidth ({variant.replace('_', ' ').upper()})", nvb_metrics(rows),
                     rows, variant=variant, source=os.path.relpath(src, folder), group="NVLink")
            continue

        # p2pBandwidthLatencyTest
        nout = [f for f in fout if f.startswith("nvlink")]
        if nout:
            rows = [parse_p2p_out(os.path.join(root, f)) for f in sorted(nout)]
            rows = [r for r in rows if r]
            if rows:
                variant = gpu_type_from_path(rel)
                tag = os.path.basename(root)
                key = f"p2p_{variant}" if "p2pbandwidth" in low else f"p2p_{variant}_{tag}"
                if key in tests and len(tests[key]["rows"]) >= len(rows):
                    continue
                label = f"CUDA P2P bandwidth/latency ({variant.upper()})" + ("" if "p2pbandwidth" in low else f" [{tag}]")
                add_test(tests, key, label, P2P_METRICS,
                         rows, variant=variant, source=rel, group="NVLink")
            continue
    return tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default="data")
    args = ap.parse_args()

    index = []
    for folder, pid, label, date in PERIODS:
        src = os.path.join(args.raw, folder)
        if not os.path.isdir(src):
            print(f"skip {folder}: not found", file=sys.stderr)
            continue
        tests = collect_period(src, pid)
        if not tests:
            print(f"skip {folder}: no tests parsed", file=sys.stderr)
            continue
        outdir = os.path.join(args.out, pid)
        os.makedirs(outdir, exist_ok=True)
        manifest = {"id": pid, "label": label, "date": date, "source_folder": folder, "tests": []}
        for key, t in tests.items():
            with open(os.path.join(outdir, f"{key}.json"), "w") as f:
                json.dump(t, f, indent=1)
            manifest["tests"].append({"key": key, "label": t["label"], "group": t["group"],
                                      "variant": t["variant"], "nodes": len(t["rows"]), "file": f"{key}.json"})
        with open(os.path.join(outdir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        index.append({"id": pid, "label": label, "date": date, "tests": len(tests),
                      "nodes": sum(len(t["rows"]) for t in tests.values())})
        print(f"{pid:12s} {label:36s} " + ", ".join(f"{k}({len(t['rows'])})" for k, t in tests.items()))
    # legacy period: the Dec 2025 files that the old index.html used
    leg_g = os.path.join(args.out, "gmx_performance_results.json")
    leg_h = os.path.join(args.out, "hpl_gflops_per_gpu.json")
    if os.path.exists(leg_g) or os.path.exists(leg_h):
        pid, label, date = "2025-12-09", "Dec 9, 2025 — Scheduled (legacy)", "2025-12-09"
        tests = OrderedDict()
        if os.path.exists(leg_g):
            add_test(tests, "gromacs_a100", "GROMACS (A100)", GMX_METRICS, load_gromacs(leg_g),
                     variant="a100", source="gmx_performance_results.json", group="GROMACS")
        if os.path.exists(leg_h):
            rows = [{"node": r["Node Number"], "gflops_per_gpu": float(r["Gflops_per_GPU"])} for r in json.load(open(leg_h))]
            add_test(tests, "hpl_a100", "NVIDIA HPL (A100, per GPU)",
                     [{"key": "gflops_per_gpu", "label": "HPL GFLOP/s per GPU", "higher_is_better": True, "primary": True}],
                     rows, variant="a100", source="hpl_gflops_per_gpu.json", group="HPL")
        outdir = os.path.join(args.out, pid)
        os.makedirs(outdir, exist_ok=True)
        manifest = {"id": pid, "label": label, "date": date, "source_folder": "legacy data/", "tests": []}
        for key, t in tests.items():
            with open(os.path.join(outdir, f"{key}.json"), "w") as f:
                json.dump(t, f, indent=1)
            manifest["tests"].append({"key": key, "label": t["label"], "group": t["group"],
                                      "variant": t["variant"], "nodes": len(t["rows"]), "file": f"{key}.json"})
        with open(os.path.join(outdir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=1)
        index.append({"id": pid, "label": label, "date": date, "tests": len(tests),
                      "nodes": sum(len(t["rows"]) for t in tests.values())})
        print(f"{pid:12s} {label:36s} " + ", ".join(f"{k}({len(t['rows'])})" for k, t in tests.items()))
    index.sort(key=lambda p: p["date"], reverse=True)
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump({"periods": index, "latest": index[0]["id"] if index else None}, f, indent=1)


if __name__ == "__main__":
    main()
