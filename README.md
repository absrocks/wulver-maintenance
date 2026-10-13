# Wulver Maintenance Benchmarks

Per-node benchmark results collected during scheduled maintenance of the
[Wulver](https://hpc.njit.edu) cluster at NJIT, and an interactive dashboard
for spotting nodes whose numbers are off.

**Live dashboard:** https://absrocks.github.io/wulver-maintenance/

## What the dashboard shows

Every maintenance window runs the same set of single-node benchmarks on every
node that is available. The dashboard loads one maintenance period at a time
(dropdown, defaults to the latest) and shows:

| Section | Purpose |
|---|---|
| Node health cards | One card per test: node count, median of the primary metric, and which nodes fall outside the tolerance band. |
| Per-node chart | Bar per node for the selected test and metric, with the median line and a ±tolerance band. Off nodes are red. Optionally overlay a second maintenance period for comparison. |
| Deviation heatmap | Node × test grid of percent deviation from the median, so a node that is weak in several tests stands out. |
| Values table | Sortable table of every node with all metrics for the test. |
| Node detail | Everything recorded for one node in the period, including per-GPU-pair results for the PyTorch and P2P tests. |
| History | The median (and min–max band) of the selected metric across all maintenance periods, with the selected node overlaid. |

The tolerance (default ±3 % of the per-test median) is adjustable. The view
state is kept in the URL so a specific node or comparison can be shared as a
link, for example
`?period=2026-10-13&test=nccl_a100&node=n0092&compare=2026-09-22`.

## Benchmarks

| Test | Nodes | Primary metric | Source |
|---|---|---|---|
| GROMACS | A100, L40, MIG 10g/20g/40g | ns/day | `gmx_performance_*.json` / `.csv`, GROMACS logs as fallback |
| HPCG | CPU nodes | GFLOP/s | `hpcg_gflops_results.json` |
| NCCL `all_reduce_perf` | A100, L40, MIG | average bus bandwidth (GB/s), plus 512 MiB / 1 GiB rows | `nccl_performance_*.json` or raw `nccl*.out` |
| PyTorch P2P comm (`torch-comm.py`) | A100 | min 1 GB sweep bandwidth over GPU pairs, failed pairs | raw `torch*.out` |
| nvbandwidth | A100, L40 | per-test summed bandwidth | `nvlink_results.json` |
| CUDA p2pBandwidthLatencyTest | A100 | min unidirectional P2P bandwidth, latency | raw `nvlink*.out` |
| NVIDIA HPL (legacy, Dec 2025) | A100 | GFLOP/s per GPU | `hpl_gflops_per_gpu.json` |

The benchmarks themselves are launched by `maint_main.py` as Slurm job arrays
inside the maintenance reservation.

## Repository layout

```
index.html              dashboard (Plotly, no build step)
collect_maint.py        normalizes raw results into data/
data/index.json         list of maintenance periods for the dropdown
data/<period>/          manifest.json + one JSON per test
maint_main.py           launches the benchmark job arrays on Wulver
torch-comm.py           PyTorch GPU-to-GPU communication test
```

Each `data/<period>/<test>.json` has the form

```json
{
  "test": "gromacs_a100", "label": "GROMACS (A100)", "group": "GROMACS", "variant": "a100",
  "metrics": [{"key": "ns_per_day", "label": "Performance (ns/day)", "primary": true}, ...],
  "rows": [{"node": "n0003", "ns_per_day": 135.2, "hour_per_ns": 0.177}, ...]
}
```

so a new benchmark only needs a loader in `collect_maint.py`; the page picks up
the metrics and rows generically.

## Adding a maintenance period

1. Run the benchmarks on Wulver. Results land under
   `/project/hpcadmins/am2455/maint/2026/<month>/`.
2. Add the month to the `PERIODS` list at the top of `collect_maint.py`
   (folder name, period id, label, date).
3. Mirror the small result files and regenerate the data:

   ```bash
   rsync -az --prune-empty-dirs --max-size=3m \
     --include='*/' --include='*.json' --include='*.out' --include='*.err' \
     --include='*.csv' --include='*.slurm' --exclude='*' \
     wulver:/project/hpcadmins/am2455/maint/2026/ raw/2026/
   python3 collect_maint.py --raw raw/2026 --out data
   ```

4. Commit and push to `main`. GitHub Pages redeploys in about a minute.

To preview locally, serve the folder over HTTP (the page fetches the JSON
files, which browsers block from `file://`):

```bash
python3 -m http.server 8000
# open http://localhost:8000/
```

## License

MIT, see [LICENSE](LICENSE).
