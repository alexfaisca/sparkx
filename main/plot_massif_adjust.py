try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    raise SystemExit("Matplotlib not found. Install with: python -m pip install matplotlib")

import argparse, csv, math, re, subprocess, shutil
from pathlib import Path

# Filenames like "3_graph_0_5.massif.out" → target=3, dataset="graph_0_5"
FNAME_RE = re.compile(r"^(\d+)_(.+)\.massif\.out$")
SNAP_RE  = re.compile(r"^snapshot=(\d+)")
KV_RE    = re.compile(r"^(mem_heap_B|mem_heap_extra_B|mem_stacks_B|mem_heapB|mem_heap_extraB|mem_stackB|time)=(\d+)$")

# ------------- ms_print-based parsing (for arena/stack subtraction) -------------
# Matches: 100.00% (1,051,291,648B) (page allocation syscalls) ...
RE_TOTAL = re.compile(r'^\s*100\.00% \(([\d,]+)B\) \(page allocation syscalls\)', re.M)
# Each tree line (allowing leading pipes/colons/spaces):
# ->57.45% (603,979,776B) 0x4A04C81: alloc_new_heap (arena.c:518)
RE_NODE  = re.compile(r'^[\s\|\:]*->\s*[\d\.]+%\s*\(([\d,]+)B\)\s+0x[0-9A-Fa-f]+:\s*(.+)$')

ARENA_MARKERS = ("alloc_new_heap", "new_heap", "_int_new_arena", "arena_get2", "tcache_init")
STACK_MARKERS = ("allocate_stack", "pthread_create@@", "pthread_create", "start_thread", "clone (clone")

def run_ms_print(path: Path) -> str | None:
    exe = shutil.which("ms_print")
    if not exe:
        return None
    try:
        out = subprocess.check_output([exe, str(path)], text=True, stderr=subprocess.STDOUT)
        return out
    except subprocess.CalledProcessError:
        return None

def adjusted_peak_from_ms_print(ms_text: str, subtract_stacks: bool) -> dict | None:
    """
    Parse ms_print output, return dict with:
      peak_total, peak_adjusted, peak_snapshot, arenas_at_peak, stacks_at_peak
    If nothing parsed, return None.
    """
    if not ms_text:
        return None
    lines = ms_text.splitlines()
    i = 0
    snap_idx = -1
    found_any = False

    best = {
        "peak_total": 0,
        "peak_adjusted": 0,
        "peak_snapshot": None,
        "arenas_at_peak": 0,
        "stacks_at_peak": 0,
    }

    while i < len(lines):
        m = RE_TOTAL.match(lines[i])
        if not m:
            i += 1
            continue

        snap_idx += 1
        total = int(m.group(1).replace(',', ''))
        i += 1

        arena_bytes = 0
        stack_bytes = 0

        # Read the breakdown until a blank line or next section
        while i < len(lines):
            line = lines[i]
            if not line.strip():
                break
            n = RE_NODE.match(line)
            if n:
                b = int(n.group(1).replace(',', ''))
                rest = n.group(2)
                if any(tok in rest for tok in ARENA_MARKERS):
                    arena_bytes += b
                if subtract_stacks and any(tok in rest for tok in STACK_MARKERS):
                    stack_bytes += b
            i += 1

        adjusted = max(0, total - arena_bytes - stack_bytes)
        found_any = True

        # Choose the snapshot with the largest ADJUSTED footprint
        if adjusted > best["peak_adjusted"]:
            best.update({
                "peak_total": total,
                "peak_adjusted": adjusted,
                "peak_snapshot": snap_idx,
                "arenas_at_peak": arena_bytes,
                "stacks_at_peak": stack_bytes,
            })

        # skip trailing blanks
        while i < len(lines) and not lines[i].strip():
            i += 1

    return best if found_any else None

# ------------- original massif.out peak (raw heap+extra+stacks) -------------
def norm_key(s: str) -> str:
    return s.lower().replace("_", "").replace("stacks", "stack")

def parse_massif_peak(path: Path) -> int:
    """Return peak of heap+extra+stacks in bytes for a massif.out file."""
    cur = {}
    peak = 0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = SNAP_RE.match(line)
            if m:
                if {"mem_heap_B","mem_heap_extra_B","mem_stacks_B"} <= cur.keys():
                    total = cur["mem_heap_B"] + cur["mem_heap_extra_B"] + cur["mem_stacks_B"]
                    peak = max(peak, total)
                cur = {}
                continue
            m = KV_RE.match(line)
            if m:
                k, v = m.group(1), int(m.group(2))
                nk = norm_key(k)
                if nk == "memheapb":
                    cur["mem_heap_B"] = v
                elif nk == "memheapextrab":
                    cur["mem_heap_extra_B"] = v
                elif nk == "memstackb":
                    cur["mem_stacks_B"] = v
    if {"mem_heap_B","mem_heap_extra_B","mem_stacks_B"} <= cur.keys():
        total = cur["mem_heap_B"] + cur["mem_heap_extra_B"] + cur["mem_stacks_B"]
        peak = max(peak, total)
    return peak

# ------------- helpers -------------
def human_bytes(n: int) -> str:
    units = ["B","KiB","MiB","GiB","TiB"]
    v = float(n); i = 0
    while v >= 1024 and i < len(units)-1:
        v /= 1024.0; i += 1
    return f"{v:.2f} {units[i]}"

def load_sizes_csv(csv_path: Path):
    """
    Expect header: dataset,V,E
    dataset: string key that must match the dataset extracted from filename (the bit after N_)
    V,E: integers
    """
    sizes = {}
    with csv_path.open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ds = row["dataset"].strip()
            V = int(row["V"].replace("_","").replace(",",""))
            E = int(row["E"].replace("_","").replace(",",""))
            sizes[ds] = (V, E)
    return sizes

def compute_x(V: int, E: int, mode: str) -> float:
    if mode == "V":
        return V
    if mode == "E":
        return E
    if mode == "VE":
        return V + E
    elif mode == "VElogE":
        return V + E * math.log(max(E, 1))
    else:
        raise ValueError("unknown x-mode: " + mode)

def collect_files(inputs, pattern):
    files = set()
    pats = [p.strip() for p in pattern.split(",") if p.strip()]
    for root in inputs:
        P = Path(root)
        if P.is_file():
            files.add(P.resolve())
        elif P.is_dir():
            for pat in pats:
                for f in P.rglob(pat):
                    if f.is_file():
                        files.add(f.resolve())
        else:
            print(f"[warn] path not found: {root}")
    return sorted(files)

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser(description="Scatter plots of Massif peaks per target (N_*.massif.out).")
    ap.add_argument("inputs", nargs="+", help="massif files or directories")
    ap.add_argument("--pattern", default="*.massif.out", help="glob pattern(s) for directories (comma-separated)")
    ap.add_argument("--sizes", required=True, help="CSV with columns: dataset,V,E")
    ap.add_argument("--x", choices=["V", "E", "VE","VElogE"], default="VE", help="x-axis formula")
    ap.add_argument("-o", "--outdir", default="massif_scatter", help="output directory")
    ap.add_argument("--logx", action="store_true", help="log scale x-axis")
    ap.add_argument("--logy", action="store_true", help="log scale y-axis")
    ap.add_argument("--labels", action="store_true", help="label points with dataset names")

    ap.add_argument("--adjust", choices=["none","minus-arenas","minus-arenas-stacks"], default="none",
                    help="subtract glibc arenas (and optionally stacks) from peaks using ms_print")
    ap.add_argument("--prefer-adjusted-plot", action="store_true",
                    help="if --adjust != none, plot the adjusted peak on the Y-axis (default: still plot raw peak but include adjusted in CSV)")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    sizes = load_sizes_csv(Path(args.sizes))
    files = collect_files(args.inputs, args.pattern)

    if args.adjust != "none" and not shutil.which("ms_print"):
        print("[warn] ms_print not found in PATH; --adjust will be ignored (falling back to raw peaks).")

    subtract_arenas = args.adjust in ("minus-arenas", "minus-arenas-stacks")
    subtract_stacks = args.adjust == "minus-arenas-stacks"

    # group: target → list of dict rows
    groups = {i: [] for i in range(0, 11)}  # 0..10
    skipped = 0
    for f in files:
        m = FNAME_RE.match(f.name)
        if not m:
            skipped += 1
            continue
        tgt = int(m.group(1))
        dataset_full = m.group(2)                # e.g., "graph_0_5.lz4" or "kmer_A2a.mtx"
        dataset  = dataset_full.split('.', 1)[0] # -> "graph_0_5", "kmer_A2a", "synth_dna_..."

        raw_peak = parse_massif_peak(f)
        adj_peak = raw_peak
        arenas_at_peak = 0
        stacks_at_peak = 0
        adj_snapshot = None

        if subtract_arenas and shutil.which("ms_print"):
            ms_text = run_ms_print(f)
            res = adjusted_peak_from_ms_print(ms_text, subtract_stacks=subtract_stacks) if ms_text else None
            if res:
                adj_peak = res["peak_adjusted"]
                arenas_at_peak = res["arenas_at_peak"]
                stacks_at_peak = res["stacks_at_peak"]
                adj_snapshot = res["peak_snapshot"]
            else:
                print(f"[warn] could not parse ms_print output for {f.name}; using raw peak")

        if dataset not in sizes:
            print(f"[warn] no size entry for dataset '{dataset}', skipping {f.name}")
            continue

        V, E = sizes[dataset]
        x = compute_x(V, E, args.x)
        groups.setdefault(tgt, []).append({
            "dataset": dataset, "file": f.name, "V": V, "E": E, "x": x,
            "peak": raw_peak, "adj_peak": adj_peak,
            "arenas_at_peak": arenas_at_peak, "stacks_at_peak": stacks_at_peak,
            "adj_snapshot": adj_snapshot
        })

    if skipped:
        print(f"[info] skipped {skipped} files that did not match 'N_*.massif.out'")

    # one scatter per target
    for tgt in sorted(groups.keys()):
        rows = groups[tgt]
        if not rows:
            continue

        xs = [r["x"] for r in rows]
        if args.prefer_adjusted_plot and subtract_arenas and shutil.which("ms_print"):
            ys = [r["adj_peak"] for r in rows]
            ylab = "peak bytes (adjusted)"
            title_suffix = "adjusted by subtracting arenas" + ("+stacks" if subtract_stacks else "")
        else:
            ys = [r["peak"] for r in rows]
            ylab = "peak bytes"
            title_suffix = "raw peaks"

        labels = [r["dataset"] for r in rows]

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(xs, ys, s=18)
        xlab = "|V| + |E|" if args.x == "VE" else ("|V| + |E| · ln |E|" if args.x == "VElogE" else ("|V|" if args.x == "V" else "|E|"))
        ax.set_title(f"Massif peaks — target {tgt}  (x = {xlab}; {title_suffix})")
        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)
        if args.logx: ax.set_xscale("log")
        if args.logy: ax.set_yscale("log")
        ax.grid(True, linestyle=":", linewidth=0.5)

        if args.labels:
            for x, y, lab in zip(xs, ys, labels):
                ax.annotate(lab, xy=(x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)

        fig.tight_layout()
        png = outdir / f"target_{tgt}_scatter_{args.x}_{'adj' if (args.prefer_adjusted_plot and subtract_arenas) else 'raw'}.png"
        fig.savefig(png, dpi=140)
        plt.close(fig)

        # per-target CSV
        csv_path = outdir / f"target_{tgt}_scatter_{args.x}.csv"
        with csv_path.open("w", newline="") as cf:
            w = csv.writer(cf)
            w.writerow([
                "dataset","file","V","E","x",
                "peak_bytes","peak_h",
                "adj_peak_bytes","adj_peak_h",
                "arenas_at_adj_peak","stacks_at_adj_peak","adj_snapshot"
            ])
            for r in rows:
                w.writerow([
                    r["dataset"], r["file"], r["V"], r["E"], r["x"],
                    r["peak"], human_bytes(r["peak"]),
                    r["adj_peak"], human_bytes(r["adj_peak"]),
                    r["arenas_at_peak"], r["stacks_at_peak"], r["adj_snapshot"]
                ])
        print(f"[ok] target {tgt}: {len(rows)} points → {png}  (table: {csv_path})")

    print(f"Done. Output in: {outdir}")

if __name__ == "__main__":
    main()
