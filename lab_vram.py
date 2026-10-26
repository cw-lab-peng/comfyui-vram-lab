"""
lab_vram.py — measure peak VRAM and wall time of ComfyUI generations, one variable at a time.

Why: local diffusion bottlenecks are usually not "the model is too big" but
"how big did you open the canvas". This script submits the graphs defined in
`lab_wf_export.py` to a running ComfyUI, samples `nvidia-smi` while the job runs,
and prints a markdown table you can paste into a README or a report.

Usage
-----
    python lab_vram.py                                  # all graphs, default server
    python lab_vram.py --url http://127.0.0.1:8189      # another instance
    python lab_vram.py --only qwen                      # subset
    python lab_vram.py --interval 0.25 --repeat 2       # finer sampling, 2 runs each

Notes
-----
* Works on NVIDIA only (uses `nvidia-smi`). On AMD/Apple, replace `vram_used()`
  with the platform equivalent — the rest of the script is vendor-agnostic.
* ComfyUI caches weights between jobs, so the 2nd run of the same model shows a
  *warm* peak. That is a measurement artefact: restart the server between arms if
  you want comparable cold peaks.
"""
import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

from lab_wf_export import GRAPHS, build_api

NVSMI = "nvidia-smi"


def vram_used():
    """Currently used MiB, or None if nvidia-smi is unavailable."""
    try:
        r = subprocess.run([NVSMI, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


class VramSampler(threading.Thread):
    """Polls used-VRAM in a background thread and keeps the maximum."""

    def __init__(self, interval=0.25):
        super().__init__(daemon=True)
        self.interval = interval
        self.peak = 0
        self.base = vram_used() or 0
        self._stop = False

    def run(self):
        while not self._stop:
            v = vram_used()
            if v and v > self.peak:
                self.peak = v
            time.sleep(self.interval)

    def stop(self):
        self._stop = True


def _post(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _get(url, timeout=15):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.load(r)


def run_arm(url, api_graph, timeout=1800):
    """Submit one graph, sample VRAM until it finishes, return (peak_gb, base_gb, seconds)."""
    s = VramSampler()
    s.start()
    t0 = time.time()
    try:
        pid = _post("%s/prompt" % url, {"prompt": api_graph})["prompt_id"]
        while time.time() - t0 < timeout:
            try:
                hist = _get("%s/history/%s" % (url, pid))
            except Exception:
                hist = {}
            if pid in hist:
                break
            time.sleep(0.5)
        else:
            raise TimeoutError("timeout after %ss" % timeout)
    finally:
        elapsed = time.time() - t0
        s.stop()
        s.join(timeout=2)
    return (s.peak / 1024.0 if s.peak else None,
            s.base / 1024.0 if s.base else None,
            round(elapsed, 1))


def main():
    ap = argparse.ArgumentParser(description="Peak-VRAM / wall-time benchmark for ComfyUI graphs")
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--only", default="")
    ap.add_argument("--interval", type=float, default=0.25)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--out", default="", help="optional path to write the markdown table")
    args = ap.parse_args()

    base = vram_used()
    print("server: %s\tsampling every %ss\tidle VRAM: %s MiB" % (args.url, args.interval, base))
    rows = []
    for name, graph in GRAPHS:
        if args.only and args.only not in name:
            continue
        for rep in range(args.repeat):
            print("[%s] run %d/%d ..." % (name, rep + 1, args.repeat), flush=True)
            peak, idle, secs = run_arm(args.url, build_api(graph), args.timeout)
            print("      peak %s GB · %ss" % (peak, secs), flush=True)
            rows.append((name, peak, idle, secs))

    # aggregate: median per arm when repeated
    agg = {}
    for name, peak, idle, secs in rows:
        agg.setdefault(name, []).append((peak, secs))
    lines = ["| graph | peak VRAM (GB) | wall time (s) |", "|---|---|---|"]
    for name, vals in agg.items():
        peak = statistics.median([v[0] for v in vals if v[0]]) if any(v[0] for v in vals) else None
        secs = statistics.median([v[1] for v in vals])
        lines.append("| %s | %s | %s |" % (name, round(peak, 2) if peak else "n/a", secs))
    table = "\n".join(lines)
    print("\n" + table)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(table + "\n")
        print("\nwritten:", args.out)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
