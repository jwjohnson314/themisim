#!/usr/bin/env python3
"""Emit LaTeX tables and a pgfplots figure from benchmark JSON reports.

Generated rather than hand-written so the numbers in the paper cannot drift from
the numbers that were measured. Re-run it after any new benchmark and the tables
update in place.

Usage::

    python scripts/make_benchmark_tex.py \
        --full benchmarks/full-index-1.009B.json \
        --pilot benchmarks/pilot-small-57k.json \
        --out benchmarks/benchmark-tables.tex

Requires in the document preamble: booktabs, pgfplots (with \\usepgfplotslibrary
{groupplots}), and siunitx is NOT needed -- numbers are pre-formatted here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ms(seconds: Optional[float]) -> str:
    if seconds is None:
        return "--"
    return f"{seconds * 1000:.0f}" if seconds >= 1 else f"{seconds * 1000:.1f}"


def esc(text: str) -> str:
    """Escape the LaTeX specials that occur in CPU model strings etc."""
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                 ("_", r"\_"), ("#", r"\#")):
        text = text.replace(a, b)
    return text


def gb(nbytes: float) -> str:
    return f"{nbytes / 1e9:.1f}"


def find_op(report: dict, nprobe: int, prefilter: int, diversify: int) -> Optional[dict]:
    for row in report.get("operating_points", []):
        if (row["nprobe"], row["prefilter"], row["diversify_seconds"]) == (
            nprobe, prefilter, diversify
        ):
            return row
    return None


def sweep_value(report: dict, nprobe: int, prefilter: int, key: str = "recall_at_k"):
    for g in report.get("operating_point_sweep", []):
        if g["nprobe"] == nprobe and g["prefilter"] == prefilter:
            return g.get(key)
    return None


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def table_recall(report: dict) -> str:
    sweep = report.get("operating_point_sweep", [])
    if not sweep:
        return "% no sweep in report\n"
    nprobes = sorted({g["nprobe"] for g in sweep})
    prefilters = sorted({g["prefilter"] for g in sweep})
    k = report["accuracy"]["k"]
    nq = report["accuracy"]["n_queries"]
    nlist = report["hyperparameters"]["index"]["nlist"]
    nvec = report["hyperparameters"]["index"]["n_vectors"]

    cols = "r" + "r" * len(prefilters)
    out = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{Recall@{k} of the deployed search pipeline against exhaustive "
        rf"search, over the $(\mathtt{{nprobe}}, \mathtt{{prefilter}})$ grid. "
        rf"Ground truth is an exact cosine scan of all {nvec:,} indexed vectors; "
        rf"recall is the mean over {nq} queries drawn uniformly without replacement "
        rf"(seed 0), each query being an indexed frame's own stored vector. "
        rf"Temporal diversification is disabled, without which the metric scores "
        rf"the de-duplicator rather than the retrieval. "
        rf"$\mathtt{{nlist}}={nlist:,}$.}}",
        rf"\label{{tab:recall-grid}}",
        rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
        r"& \multicolumn{" + str(len(prefilters)) + r"}{c}{\texttt{prefilter}} \\",
        r"\cmidrule(lr){2-" + str(len(prefilters) + 1) + r"}",
        r"\texttt{nprobe} (cells scanned) & " + " & ".join(str(p) for p in prefilters) + r" \\",
        r"\midrule",
    ]
    for npr in nprobes:
        frac = next(g["fraction_of_cells_probed"] for g in sweep if g["nprobe"] == npr)
        cells = []
        for pf in prefilters:
            v = sweep_value(report, npr, pf)
            cells.append("--" if v is None else f"{v:.3f}")
        out.append(rf"{npr} ({frac * 100:.3f}\%) & " + " & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(out)


def table_latency(report: dict) -> str:
    hw, st = report["hardware"], report["storage"]
    idx = st.get("index.faiss", {})
    vec = st.get("vectors.f16.dat", {})
    rows = []
    for (npr, pf) in ((64, 500), (256, 1000)):
        for div in (0, 30):
            r = find_op(report, npr, pf, div)
            if r is None:
                continue
            w, c = r["warm"], r.get("cold")
            rec = sweep_value(report, npr, pf)
            rows.append(
                rf"{npr} & {pf} & {div} & {w['n']} & {ms(w['median_s'])} & "
                rf"{ms(w['p95_s'])} & "
                + (ms(c["median_s"]) if c else "--") + " & "
                + (ms(c["p95_s"]) if c else "--") + " & "
                + ("--" if rec is None or div != 0 else f"{rec:.3f}") + r" \\"
            )
    caption = (
        rf"Query latency at the library default operating point "
        rf"($\mathtt{{nprobe}}=64$, $\mathtt{{prefilter}}=500$) and at the "
        rf"higher-recall setting $(256, 1000)$. Latency covers the full pipeline: "
        rf"IVF-PQ probe plus exact \texttt{{fp32}} rerank of the candidate pool. "
        rf"Hardware: {esc(hw['cpu_model'])} ({hw['cpu_logical']} logical cores), "
        rf"{(hw['ram_total_bytes'] or 0) / 2**30:.0f}\,GiB RAM, FAISS "
        rf"{esc(str(hw['faiss']))} with {hw['faiss_omp_threads']} OpenMP threads. "
        rf"\textbf{{Storage placement is material}}: \texttt{{index.faiss}} "
        rf"({gb(idx.get('size_bytes', 0))}\,GB) resides on {idx.get('media', '?')}, "
        rf"while \texttt{{vectors.f16.dat}} ({gb(vec.get('size_bytes', 0))}\,GB), "
        rf"read by the rerank, resides on {vec.get('media', '?')} and is far larger "
        rf"than RAM. Recall is quoted only for $\mathtt{{diversify}}=0$; see "
        rf"Table~\ref{{tab:recall-grid}}."
    )
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", r"\label{tab:latency}",
        r"\begin{tabular}{rrrrrrrrr}", r"\toprule",
        r"& & & & \multicolumn{2}{c}{warm (ms)} & \multicolumn{2}{c}{cold (ms)} & \\",
        r"\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
        r"\texttt{nprobe} & \texttt{prefilter} & \texttt{div} & $n$ & median & p95 & median & p95 & recall@10 \\",
        r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def table_cache(report: dict) -> str:
    rows = []
    for (npr, pf) in ((64, 500), (256, 1000)):
        for div in (0, 30):
            r = find_op(report, npr, pf, div)
            if r is None or "cold" not in r:
                continue
            cold, warm = r["cold"], r["warm_paired"]
            ratio = cold["median_s"] / warm["median_s"] if warm["median_s"] else float("nan")
            rows.append(
                rf"{npr} & {pf} & {div} & {cold['n']} & {ms(cold['median_s'])} & "
                rf"{ms(warm['median_s'])} & {ratio:.2f}$\times$ \\"
            )
    load = report.get("engine_load_s_cold")
    sizes = report["storage"]
    total = sum(v.get("size_bytes", 0) for v in sizes.values())
    ram = report["hardware"]["ram_total_bytes"] or 1
    caption = (
        r"Cold- versus warm-cache query latency. Cold measurements evict every "
        r"artifact from the page cache with \texttt{posix\_fadvise(DONTNEED)} "
        r"before each query; the warm column re-runs the \emph{identical} query "
        r"set, so the ratio is not confounded by query difficulty. The index is "
        r"opened with \texttt{faiss.IO\_FLAG\_MMAP} and the manifest columns with "
        r"\texttt{np.load(mmap\_mode=\textquotesingle r\textquotesingle)}, so "
        rf"engine construction costs only {load:.1f}\,s and the remaining cost is "
        rf"paid as page faults during queries. The artifacts total "
        rf"{total / 1e9:.0f}\,GB against {ram / 2**30:.0f}\,GiB of RAM "
        rf"($\approx{total / ram:.0f}\times$), so the vector store cannot be "
        rf"fully cached at any point."
    )
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", r"\label{tab:cache}",
        r"\begin{tabular}{rrrrrrr}", r"\toprule",
        r"\texttt{nprobe} & \texttt{prefilter} & \texttt{div} & $n$ & cold median (ms) & warm median (ms) & cold/warm \\",
        r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def table_baseline(report: dict) -> str:
    bf, sp = report["brute_force"], report["speedup_vs_brute_force"]
    acc = report["accuracy"]
    warm = find_op(report, 64, 500, 0)
    warm_med = warm["warm"]["median_s"] if warm else report["latency_warm"]["median_s"]
    rows = [
        rf"Vectors scanned per pass & {bf['n_vectors_scanned']:,} \\",
        rf"Bytes read per pass & {bf['bytes_read'] / 1e12:.3f}\,TB \\",
        rf"Effective scan rate & {bf['effective_scan_bytes_per_s'] / 1e6:.0f}\,MB/s \\",
        rf"Wall clock, one full pass & {bf['wall_clock_s_one_pass'] / 60:.1f}\,min "
        rf"({bf['wall_clock_s_one_pass'] / 3600:.2f}\,h) \\",
        r"\midrule",
        rf"Brute force, single query & {bf['seconds_per_query_single']:,.0f}\,s \\",
        rf"Brute force, batched over {bf['n_queries_in_pass']} queries & "
        rf"{bf['seconds_per_query_batched']:.1f}\,s/query \\",
        rf"Index, warm median (64, 500) & {warm_med * 1000:.0f}\,ms \\",
        r"\midrule",
        rf"Speedup, single query (warm) & $\approx{sp['single_query_warm']:,.0f}\times$ \\",
        rf"Speedup, batched throughput & $\approx{sp['batched_throughput']:,.0f}\times$ \\",
        rf"Recall@{acc['k']} retained & {acc['recall_at_k']:.3f} \\",
    ]
    caption = (
        r"Exact brute-force baseline. One sequential pass computes cosine "
        r"similarity against every indexed vector, casting \texttt{fp16} to "
        r"\texttt{fp32} and L2-normalising per chunk, identical to the arithmetic "
        r"the rerank performs. A single isolated query requires a whole pass; "
        r"batching amortises one pass over all queries. "
        r"\textbf{The baseline is not storage-bound and should not be read as a "
        rf"lower bound}}: it sustained {bf['effective_scan_bytes_per_s'] / 1e6:.0f}\,MB/s "
        r"against a device measured at "
        rf"{report['storage']['vectors.f16.dat'].get('cold_sequential_read_bytes_per_s', 0) / 1e6:.0f}"
        r"\,MB/s, because each chunk serialises read, conversion, normalisation and "
        r"matrix multiply with no overlap. A pipelined implementation would reduce "
        r"the speedups by roughly a factor of two."
    )
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", r"\label{tab:baseline}",
        r"\begin{tabular}{lr}", r"\toprule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def table_scaling(full: dict, pilot: dict) -> str:
    def row(label, fmt, keyfn):
        return rf"{label} & {fmt(keyfn(full))} & {fmt(keyfn(pilot))} \\"

    def warm_med(r):
        op = find_op(r, 64, 500, 0)
        return op["warm"]["median_s"] if op else r["latency_warm"]["median_s"]

    ident = lambda x: x
    rows = [
        row(r"Indexed vectors", lambda v: f"{v:,}",
            lambda r: r["hyperparameters"]["index"]["n_vectors"]),
        row(r"\texttt{nlist}", lambda v: f"{v:,}",
            lambda r: r["hyperparameters"]["index"]["nlist"]),
        row(r"Cells scanned at $\mathtt{nprobe}=64$", lambda v: f"{v * 100:.3f}\\%",
            lambda r: min(64, r["hyperparameters"]["index"]["nlist"])
            / r["hyperparameters"]["index"]["nlist"]),
        r"\midrule",
        row(r"Warm median latency (ms)", lambda v: f"{v * 1000:.1f}", warm_med),
        row(r"Throughput (queries/s)", lambda v: f"{v:.2f}",
            lambda r: r["throughput_warm"]["queries_per_s"]),
        row(r"Brute-force pass", lambda v: f"{v / 60:.2f}\\,min",
            lambda r: r["brute_force"]["wall_clock_s_one_pass"]),
        r"\midrule",
        row(r"Speedup, single warm query", lambda v: f"${v:,.0f}\\times$",
            lambda r: r["speedup_vs_brute_force"]["single_query_warm"]),
        row(r"Speedup, batched throughput", lambda v: f"${v:,.2f}\\times$",
            lambda r: r["speedup_vs_brute_force"]["batched_throughput"]),
        row(r"Recall@10 vs exhaustive", lambda v: f"{v:.3f}",
            lambda r: r["accuracy"]["recall_at_k"]),
    ]
    caption = (
        r"Scaling behaviour: the production index against the 57k-vector "
        r"\texttt{small} pilot, measured identically. Two points follow. "
        r"\textbf{(i)} At pilot scale the approximate index is \emph{slower} than "
        r"a linear scan for batch workloads, so the approach earns its keep only "
        r"at production scale. \textbf{(ii)} Pilot recall is an upper bound on "
        r"production recall rather than an estimate of it, because at equal "
        r"\texttt{nprobe} the pilot scans a far larger fraction of its database."
    )
    return "\n".join([
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", r"\label{tab:scaling}",
        r"\begin{tabular}{lrr}", r"\toprule",
        r"& Production & Pilot (\texttt{small}) \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
    ])


def figure_recall(report: dict) -> str:
    sweep = report.get("operating_point_sweep", [])
    if not sweep:
        return "% no sweep in report\n"
    nprobes = sorted({g["nprobe"] for g in sweep})
    prefilters = sorted({g["prefilter"] for g in sweep})
    k = report["accuracy"]["k"]
    nq = report["accuracy"]["n_queries"]

    left = []
    for pf in prefilters:
        pts = " ".join(
            f"({npr},{sweep_value(report, npr, pf):.4f})"
            for npr in nprobes if sweep_value(report, npr, pf) is not None
        )
        left.append(rf"\addplot+[mark=*] coordinates {{{pts}}};")
        left.append(rf"\addlegendentry{{\texttt{{prefilter}}={pf}}}")
    right = []
    for npr in nprobes:
        pts = " ".join(
            f"({pf},{sweep_value(report, npr, pf):.4f})"
            for pf in prefilters if sweep_value(report, npr, pf) is not None
        )
        right.append(rf"\addplot+[mark=*] coordinates {{{pts}}};")
        right.append(rf"\addlegendentry{{\texttt{{nprobe}}={npr}}}")

    return "\n".join([
        r"\begin{figure}[t]", r"\centering", r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"  group style={group size=2 by 1, horizontal sep=1.6cm},",
        r"  width=0.46\linewidth, height=5.2cm,",
        rf"  ylabel={{recall@{k}}}, ymin=0.55, ymax=1.02, grid=major,",
        r"  legend style={font=\scriptsize, fill opacity=0.85, draw opacity=1,",
        r"                text opacity=1},",
        r"  xmode=log, tick label style={font=\scriptsize},",
        r"  label style={font=\small}, log ticks with fixed point,",
        r"]",
        # Explicit ticks at the measured values only: a log axis would otherwise
        # label decades that were never sampled, implying data that is not there.
        r"\nextgroupplot[xlabel={\texttt{nprobe} (IVF cells probed)},",
        rf"  xtick={{{','.join(str(n) for n in nprobes)}}}, legend pos=south east]",
        *left,
        r"\nextgroupplot[xlabel={\texttt{prefilter} (candidates reranked)},",
        rf"  xtick={{{','.join(str(p) for p in prefilters)}}}, legend pos=north west]",
        *right,
        r"\end{groupplot}", r"\end{tikzpicture}",
        rf"\caption{{Recall@{k} against exhaustive search as a function of "
        rf"\texttt{{nprobe}} (left) and \texttt{{prefilter}} (right), from the same "
        rf"{nq} queries and the same exact ground truth. \texttt{{prefilter}} "
        rf"governs recall; \texttt{{nprobe}} has almost no effect once it exceeds "
        rf"8, because the candidate budget rather than the number of cells probed "
        rf"determines whether a true neighbour reaches the rerank. Temporal "
        rf"diversification disabled.}}",
        r"\label{fig:recall-panels}", r"\end{figure}", "",
    ])


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--full", type=Path, required=True)
    ap.add_argument("--pilot", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    full = json.loads(args.full.read_text())
    pilot = json.loads(args.pilot.read_text()) if args.pilot else None

    parts = [
        "% Generated by scripts/make_benchmark_tex.py -- do not edit by hand.",
        f"% Source: {args.full.name}" + (f", {args.pilot.name}" if args.pilot else ""),
        f"% Measured: {full['generated_utc']}",
        "% Preamble needs: \\usepackage{booktabs}, \\usepackage{pgfplots},",
        "%                 \\usepgfplotslibrary{groupplots}",
        "",
        table_recall(full),
        figure_recall(full),
        table_latency(full),
        table_cache(full),
        table_baseline(full),
    ]
    if pilot:
        parts.append(table_scaling(full, pilot))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(parts))
    print(f"Wrote {args.out} ({len(''.join(parts).splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
