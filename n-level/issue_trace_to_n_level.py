#!/usr/bin/env python3
"""
Infer per-SM N-level warp allocation from issue trace CSV.
"""

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict


def parse_id_list(text):
    if not text:
        return None
    ids = set()
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                start, end = end, start
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return ids


def collect_input_files(input_path):
    if os.path.isdir(input_path):
        files = []
        for name in os.listdir(input_path):
            if name.lower().endswith(".csv"):
                files.append(os.path.join(input_path, name))
        files.sort()
        if not files:
            raise ValueError(f"no .csv files in input dir: {input_path}")
        return files
    if not os.path.exists(input_path):
        raise ValueError(f"input path not found: {input_path}")
    return [input_path]


def resolve_output_paths(output_path, input_files):
    if os.path.isdir(output_path) or output_path.endswith(os.sep):
        output_dir = output_path.rstrip(os.sep)
        os.makedirs(output_dir, exist_ok=True)
        out_map = {}
        for in_path in input_files:
            base = os.path.splitext(os.path.basename(in_path))[0]
            out_csv = os.path.join(output_dir, f"{base}_warp_allocate.csv")
            out_diag = os.path.join(output_dir, f"{base}_diagnostics.txt")
            out_map[in_path] = (out_csv, out_diag)
        return out_map

    if len(input_files) > 1:
        raise ValueError("output path is a file but multiple input files found")
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(output_path))[0]
    out_diag = os.path.join(output_dir, f"{base}_diagnostics.txt")
    return {input_files[0]: (output_path, out_diag)}


def parse_issue_trace(path, sm_filter):
    last_issue = defaultdict(dict)
    max_warp_seen = defaultdict(lambda: -1)
    stats = {
        "lines": 0,
        "issue_lines": 0,
        "bad_rows": 0,
    }
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"empty file: {path}")
        idx = {name.strip(): i for i, name in enumerate(header)}
        required = ["cycle", "sm", "warp", "event"]
        for name in required:
            if name not in idx:
                raise ValueError(f"missing column '{name}' in {path}")
        c_cycle = idx["cycle"]
        c_sm = idx["sm"]
        c_warp = idx["warp"]
        c_event = idx["event"]
        for row in reader:
            if not row:
                continue
            if row[0].lstrip().startswith("#"):
                continue
            stats["lines"] += 1
            try:
                event = row[c_event].strip()
            except IndexError:
                stats["bad_rows"] += 1
                continue
            if event != "ISSUE":
                continue
            stats["issue_lines"] += 1
            try:
                sm = int(row[c_sm])
                warp = int(row[c_warp])
                cycle = int(row[c_cycle])
            except (ValueError, IndexError):
                stats["bad_rows"] += 1
                continue
            if sm_filter is not None and sm not in sm_filter:
                continue
            if warp < 0:
                stats["bad_rows"] += 1
                continue
            if warp > max_warp_seen[sm]:
                max_warp_seen[sm] = warp
            prev = last_issue[sm].get(warp)
            if prev is None or cycle > prev:
                last_issue[sm][warp] = cycle
    return last_issue, max_warp_seen, stats


def prefix_sums(values):
    prefix = [0.0]
    prefix_sq = [0.0]
    for v in values:
        prefix.append(prefix[-1] + v)
        prefix_sq.append(prefix_sq[-1] + v * v)
    return prefix, prefix_sq


def sse(prefix, prefix_sq, i, j):
    count = j - i + 1
    if count <= 0:
        return 0.0
    total = prefix[j + 1] - prefix[i]
    total_sq = prefix_sq[j + 1] - prefix_sq[i]
    return total_sq - (total * total) / count


def optimal_1d_kmeans(values, kmax):
    n = len(values)
    prefix, prefix_sq = prefix_sums(values)
    dp = [[math.inf] * (n + 1) for _ in range(kmax + 1)]
    prev = [[-1] * (n + 1) for _ in range(kmax + 1)]
    dp[0][0] = 0.0
    for k in range(1, kmax + 1):
        for i in range(1, n + 1):
            if i < k:
                continue
            best = math.inf
            best_m = -1
            for m in range(k - 1, i):
                cost = dp[k - 1][m] + sse(prefix, prefix_sq, m, i - 1)
                if cost < best:
                    best = cost
                    best_m = m
            dp[k][i] = best
            prev[k][i] = best_m
    return dp, prev


def backtrack(prev, k, n):
    boundaries = []
    i = n
    for kk in range(k, 0, -1):
        m = prev[kk][i]
        if m < 0:
            return []
        boundaries.append((m, i - 1))
        i = m
    boundaries.reverse()
    return boundaries


def compute_bic(sse_value, k, n):
    eps = 1e-9
    sse_value = max(sse_value, eps)
    return n * math.log(sse_value / n) + k * math.log(n) * 2.0


def silhouette_score(values, boundaries):
    n = len(values)
    if n <= 1 or len(boundaries) <= 1:
        return None
    labels = [-1] * n
    clusters = []
    for idx, (start, end) in enumerate(boundaries):
        members = list(range(start, end + 1))
        clusters.append(members)
        for m in members:
            labels[m] = idx
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = abs(values[i] - values[j])
            dist[i][j] = d
            dist[j][i] = d
    total = 0.0
    for i in range(n):
        cluster_id = labels[i]
        same = clusters[cluster_id]
        if len(same) > 1:
            a = sum(dist[i][j] for j in same if j != i) / (len(same) - 1)
        else:
            a = 0.0
        b = None
        for cid, members in enumerate(clusters):
            if cid == cluster_id:
                continue
            avg = sum(dist[i][j] for j in members) / len(members)
            if b is None or avg < b:
                b = avg
        if b is None:
            s = 0.0
        else:
            denom = max(a, b)
            s = 0.0 if denom == 0 else (b - a) / denom
        total += s
    return total / n


def choose_k(values, kmin, kmax, algo):
    n = len(values)
    kmax = min(kmax, n)
    kmin = min(kmin, kmax)
    dp, prev = optimal_1d_kmeans(values, kmax)
    metrics = {}
    if algo == "kmeans-bic":
        best_k = kmin
        best_score = math.inf
        bic_values = {}
        for k in range(kmin, kmax + 1):
            score = compute_bic(dp[k][n], k, n)
            bic_values[k] = score
            if score < best_score:
                best_score = score
                best_k = k
        metrics["bic"] = bic_values
        return best_k, dp, prev, metrics
    if algo == "kmeans-silhouette":
        best_k = kmin
        best_score = -math.inf
        sil_values = {}
        for k in range(max(kmin, 2), kmax + 1):
            boundaries = backtrack(prev, k, n)
            score = silhouette_score(values, boundaries)
            if score is None:
                continue
            sil_values[k] = score
            if score > best_score:
                best_score = score
                best_k = k
        if not sil_values:
            best_k = 1
        metrics["silhouette"] = sil_values
        return best_k, dp, prev, metrics
    raise ValueError(f"unknown algo: {algo}")


def center_value(values, mode):
    if mode == "mean":
        return int(round(sum(values) / len(values)))
    if mode == "median":
        return int(statistics.median(values))
    raise ValueError(f"unknown center mode: {mode}")


def gcd_list(nums):
    g = 0
    for n in nums:
        g = math.gcd(g, int(n))
    return g


def approx_gcd_slices(reps, abs_tol, rel_tol, max_slice):
    reps_int = [int(r) for r in reps]
    if not reps_int:
        return [], {
            "base": None,
            "accepted": True,
            "total_error": 0.0,
            "max_error": 0.0,
            "sum_slices": 0,
        }
    if len(reps_int) == 1:
        return [1], {
            "base": reps_int[0],
            "accepted": True,
            "total_error": 0.0,
            "max_error": 0.0,
            "sum_slices": 1,
        }

    best_ok = None
    best_ok_data = None
    best_fallback = None
    best_fallback_data = None

    for rep in reps_int:
        if rep <= 0:
            continue
        for m in range(1, max_slice + 1):
            g = rep / m
            if g <= 0:
                continue
            slices = [max(1, int(round(r / g))) for r in reps_int]
            ratios = [r / s for r, s in zip(reps_int, slices)]
            g_ref = statistics.median(ratios)
            slices = [max(1, int(round(r / g_ref))) for r in reps_int]
            ratios = [r / s for r, s in zip(reps_int, slices)]
            g_ref = statistics.median(ratios)
            errors = [abs(r - s * g_ref) for r, s in zip(reps_int, slices)]
            allowed = [max(abs_tol, rel_tol * r) for r in reps_int]
            max_slice_val = max(slices)
            if max_slice > 0 and max_slice_val > max_slice:
                continue
            sum_slices = sum(slices)
            total_error = sum(errors)
            max_error = max(errors) if errors else 0.0
            ok = all(err <= lim for err, lim in zip(errors, allowed))
            if ok:
                score = (sum_slices, max_slice_val, total_error, max_error)
                if best_ok is None or score < best_ok:
                    best_ok = score
                    best_ok_data = (
                        slices,
                        g_ref,
                        total_error,
                        max_error,
                        sum_slices,
                        True,
                    )
            else:
                score = (sum_slices, total_error, max_slice_val, max_error)
                if best_fallback is None or score < best_fallback:
                    best_fallback = score
                    best_fallback_data = (
                        slices,
                        g_ref,
                        total_error,
                        max_error,
                        sum_slices,
                        False,
                    )

    if best_ok_data is not None:
        slices, g_ref, total_error, max_error, sum_slices, accepted = best_ok_data
    elif best_fallback_data is not None:
        slices, g_ref, total_error, max_error, sum_slices, accepted = best_fallback_data
    else:
        slices = [1 for _ in reps_int]
        g_ref = min(reps_int) if reps_int else 1
        total_error = 0.0
        max_error = 0.0
        sum_slices = sum(slices)
        accepted = False

    return slices, {
        "base": g_ref,
        "accepted": accepted,
        "total_error": total_error,
        "max_error": max_error,
        "sum_slices": sum_slices,
    }


def compute_time_slices(reps, mode, abs_tol, rel_tol, max_slice):
    if not reps:
        return [], {
            "base": None,
            "accepted": True,
            "total_error": 0.0,
            "max_error": 0.0,
            "sum_slices": 0,
        }
    reps_int = [int(r) for r in reps]
    if mode == "gcd":
        g = gcd_list(reps_int)
        if g <= 0:
            g = 1
        slices = [max(1, r // g) for r in reps_int]
        return slices, {
            "base": g,
            "accepted": True,
            "total_error": 0.0,
            "max_error": 0.0,
            "sum_slices": sum(slices),
        }
    if mode == "gcd-diff":
        base = min(reps_int)
        diffs = [r - base for r in reps_int]
        g = gcd_list([d for d in diffs if d != 0])
        if g <= 0:
            g = 1
        slices = [max(1, (r - base) // g + 1) for r in reps_int]
        return slices, {
            "base": g,
            "accepted": True,
            "total_error": 0.0,
            "max_error": 0.0,
            "sum_slices": sum(slices),
        }
    if mode == "approx-gcd":
        return approx_gcd_slices(reps_int, abs_tol, rel_tol, max_slice)
    raise ValueError(f"unknown slice mode: {mode}")


def fmt_list(values):
    return "(" + ",".join(str(v) for v in values) + ")"


def merge_groups_by_slice(groups, slices, rep_times):
    merged_groups = []
    merged_slices = []
    merged_reps = []
    merges = []
    i = 0
    while i < len(groups):
        slice_val = slices[i]
        indices = [i]
        j = i + 1
        while j < len(groups) and slices[j] == slice_val:
            indices.append(j)
            j += 1
        if len(indices) == 1:
            merged_groups.append(groups[i])
            merged_slices.append(slice_val)
            merged_reps.append(rep_times[i])
        else:
            warps = []
            reps = []
            for idx in indices:
                warps.extend(groups[idx])
                reps.append(rep_times[idx])
            dup_count = len(warps) - len(set(warps))
            warps = sorted(set(warps), reverse=True)
            merged_rep = int(round(statistics.median(reps)))
            merged_groups.append(warps)
            merged_slices.append(slice_val)
            merged_reps.append(merged_rep)
            merges.append(
                {
                    "slice": slice_val,
                    "from_groups": indices,
                    "merged_rep": merged_rep,
                    "warp_count": len(warps),
                    "dup_warps": dup_count,
                }
            )
        i = j
    return merged_groups, merged_slices, merged_reps, merges


def cluster_sm(sm_id, last_issue, max_warps, args):
    normal = []
    outliers = []
    for warp_id in range(max_warps):
        cycle = last_issue.get(warp_id)
        if cycle is None:
            outliers.append(warp_id)
        else:
            normal.append((warp_id, cycle))

    normal.sort(key=lambda x: x[1])
    info = {
        "sm_id": sm_id,
        "max_warps": max_warps,
        "normal_count": len(normal),
        "outlier_count": len(outliers),
        "metrics": {},
        "clusters": [],
        "outlier_groups": [],
    }

    groups = []
    rep_times = []
    if normal:
        values = [cycle for _, cycle in normal]
        kmax = min(args.max_groups, len(values))
        kmin = min(args.min_groups, kmax)
        k, dp, prev, metrics = choose_k(values, kmin, kmax, args.algo)
        boundaries = backtrack(prev, k, len(values))
        info["metrics"] = metrics
        info["chosen_k"] = k
        info["boundaries"] = boundaries
        for group_id, (start, end) in enumerate(boundaries):
            warps = [normal[i][0] for i in range(start, end + 1)]
            cycles = [normal[i][1] for i in range(start, end + 1)]
            rep = center_value(cycles, args.center)
            info["clusters"].append(
                {
                    "group_id": group_id,
                    "size": len(warps),
                    "rep": rep,
                    "cycles": cycles,
                    "warps": warps,
                }
            )
            groups.append(sorted(warps, reverse=True))
            rep_times.append(rep)

    slices, slice_info = compute_time_slices(
        rep_times,
        args.slice_mode,
        args.slice_abs_tol,
        args.slice_rel_tol,
        args.slice_max,
    )
    if normal and len(slices) != len(groups):
        raise RuntimeError("slice/group length mismatch")
    info["slice_info"] = slice_info

    if args.merge_same_slice and groups:
        merged_groups, merged_slices, merged_reps, merges = merge_groups_by_slice(
            groups, slices, rep_times
        )
        if merges:
            info["merge_same_slice"] = True
            info["merge_groups"] = merges
        groups = merged_groups
        slices = merged_slices
        rep_times = merged_reps

    outlier_groups = []
    if outliers:
        if args.outlier_mode == "merge":
            outlier_groups = [sorted(outliers, reverse=True)]
        elif args.outlier_mode == "separate":
            outlier_groups = [[warp] for warp in sorted(outliers, reverse=True)]
        else:
            raise ValueError(f"unknown outlier mode: {args.outlier_mode}")
        info["outlier_groups"] = outlier_groups

    groups.extend(outlier_groups)
    slices.extend([args.outlier_slice] * len(outlier_groups))

    group_sizes = [len(g) for g in groups]
    return {
        "groups": groups,
        "group_sizes": group_sizes,
        "slices": slices,
        "info": info,
    }


def write_diagnostics(path, input_path, stats, per_sm_info, args):
    with open(path, "w") as f:
        f.write(f"input: {input_path}\n")
        f.write(f"algo: {args.algo}\n")
        f.write(f"center: {args.center}\n")
        f.write(f"slice_mode: {args.slice_mode}\n")
        f.write(f"slice_abs_tol: {args.slice_abs_tol}\n")
        f.write(f"slice_rel_tol: {args.slice_rel_tol}\n")
        f.write(f"slice_max: {args.slice_max}\n")
        f.write(f"min_groups: {args.min_groups}\n")
        f.write(f"max_groups: {args.max_groups}\n")
        f.write(f"max_warps: {args.max_warps}\n")
        f.write(f"outlier_mode: {args.outlier_mode}\n")
        f.write(f"outlier_slice: {args.outlier_slice}\n")
        f.write(f"lines: {stats['lines']}\n")
        f.write(f"issue_lines: {stats['issue_lines']}\n")
        f.write(f"bad_rows: {stats['bad_rows']}\n")
        f.write("\n")
        for sm_id in sorted(per_sm_info):
            info = per_sm_info[sm_id]
            f.write(f"SM {sm_id}\n")
            f.write(f"  max_warps: {info['max_warps']}\n")
            f.write(f"  normal_warps: {info['normal_count']}\n")
            f.write(f"  outliers: {info['outlier_count']}\n")
            if "chosen_k" in info:
                f.write(f"  chosen_k: {info['chosen_k']}\n")
            metrics = info.get("metrics", {})
            if "bic" in metrics:
                f.write("  bic:\n")
                for k in sorted(metrics["bic"]):
                    f.write(f"    k={k}: {metrics['bic'][k]:.4f}\n")
            if "silhouette" in metrics:
                f.write("  silhouette:\n")
                for k in sorted(metrics["silhouette"]):
                    f.write(f"    k={k}: {metrics['silhouette'][k]:.4f}\n")
            slice_info = info.get("slice_info", {})
            if slice_info:
                f.write(
                    "  slice_info: "
                    f"base={slice_info.get('base')}, "
                    f"accepted={slice_info.get('accepted')}, "
                    f"total_error={slice_info.get('total_error'):.4f}, "
                    f"max_error={slice_info.get('max_error'):.4f}, "
                    f"sum_slices={slice_info.get('sum_slices')}\n"
                )
            if info.get("merge_same_slice"):
                f.write("  merge_same_slice: enabled\n")
                for merge in info.get("merge_groups", []):
                    f.write(
                        "    merge slice="
                        f"{merge.get('slice')}: "
                        f"groups={merge.get('from_groups')}, "
                        f"merged_rep={merge.get('merged_rep')}, "
                        f"warp_count={merge.get('warp_count')}, "
                        f"dup_warps={merge.get('dup_warps')}\n"
                    )
            for cluster in info.get("clusters", []):
                f.write(
                    f"  group {cluster['group_id']}: size={cluster['size']}, rep={cluster['rep']}\n"
                )
                f.write(f"    warps: {cluster['warps']}\n")
                f.write(f"    cycles: {cluster['cycles']}\n")
            if info.get("outlier_groups"):
                f.write("  outlier_groups:\n")
                for group in info["outlier_groups"]:
                    f.write(f"    {group}\n")
            f.write("\n")


def build_output_line(sm_id, groups, group_sizes, slices):
    parts = [str(sm_id), fmt_list(group_sizes), fmt_list(slices)]
    for group in groups:
        parts.append(fmt_list(group))
    return ",".join(parts)


def main():
    parser = argparse.ArgumentParser(
        description="Infer N-level warp allocation from issue trace CSV."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Issue trace CSV file or directory containing CSV files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file or directory.",
    )
    parser.add_argument(
        "--algo",
        choices=["kmeans-bic", "kmeans-silhouette"],
        default="kmeans-bic",
        help="Clustering algorithm and model selection.",
    )
    parser.add_argument(
        "--min-groups",
        type=int,
        default=1,
        help="Minimum number of groups per SM.",
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=8,
        help="Maximum number of groups per SM.",
    )
    parser.add_argument(
        "--max-warps",
        type=int,
        default=64,
        help="Max warp slots per SM (default 64).",
    )
    parser.add_argument(
        "--center",
        choices=["median", "mean"],
        default="median",
        help="Representative time per cluster.",
    )
    parser.add_argument(
        "--slice-mode",
        choices=["approx-gcd", "gcd", "gcd-diff"],
        default="approx-gcd",
        help="Time slice mapping from representative times.",
    )
    parser.add_argument(
        "--slice-abs-tol",
        type=float,
        default=2.0,
        help="Absolute tolerance (cycles) for approximate slice fitting.",
    )
    parser.add_argument(
        "--slice-rel-tol",
        type=float,
        default=0.01,
        help="Relative tolerance for approximate slice fitting.",
    )
    parser.add_argument(
        "--slice-max",
        type=int,
        default=256,
        help="Maximum allowed slice value (also used as search bound).",
    )
    parser.add_argument(
        "--merge-same-slice",
        action="store_true",
        default=True,
        help="Merge adjacent groups that share the same slice (default).",
    )
    parser.add_argument(
        "--no-merge-same-slice",
        dest="merge_same_slice",
        action="store_false",
        help="Disable merging of same-slice groups.",
    )
    parser.add_argument(
        "--outlier-mode",
        choices=["separate", "merge"],
        default="separate",
        help="How to group warps with no ISSUE events.",
    )
    parser.add_argument(
        "--outlier-slice",
        type=int,
        default=1,
        help="Time slice length for outlier groups.",
    )
    parser.add_argument(
        "--sm-filter",
        default=None,
        help="Comma-separated SM ids or ranges (e.g., 0,1,2-4).",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print per-SM summary to stdout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed placeholder (unused for deterministic algorithms).",
    )
    args = parser.parse_args()

    sm_filter = parse_id_list(args.sm_filter)
    try:
        input_files = collect_input_files(args.input)
        out_map = resolve_output_paths(args.output, input_files)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for in_path in input_files:
        out_csv, out_diag = out_map[in_path]
        last_issue, max_warp_seen, stats = parse_issue_trace(in_path, sm_filter)
        per_sm_info = {}
        output_lines = []

        sm_ids = sorted(last_issue.keys())
        if sm_filter is not None:
            sm_ids = sorted(sm_filter)

        for sm_id in sm_ids:
            warp_data = last_issue.get(sm_id, {})
            max_warp = args.max_warps
            seen = max_warp_seen.get(sm_id, -1)
            if seen >= max_warp:
                max_warp = seen + 1
            result = cluster_sm(sm_id, warp_data, max_warp, args)
            output_lines.append(
                build_output_line(
                    sm_id,
                    result["groups"],
                    result["group_sizes"],
                    result["slices"],
                )
            )
            per_sm_info[sm_id] = result["info"]
            if args.summary:
                print(
                    f"SM {sm_id}: groups={len(result['groups'])}, "
                    f"normal={result['info']['normal_count']}, "
                    f"outliers={result['info']['outlier_count']}"
                )

        with open(out_csv, "w") as f:
            for line in output_lines:
                f.write(line + "\n")

        write_diagnostics(out_diag, in_path, stats, per_sm_info, args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
