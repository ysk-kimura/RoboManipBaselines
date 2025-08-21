#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheckMissingResults.py

Scan ./result for completed runs, generate the full product from configured
POLICIES/SEEDS/ENV_LIST/REMARK_LIST, and print unexecuted combinations.
"""

from __future__ import annotations
import argparse
import glob
import os
import re
import sys
import yaml
from collections import Counter
from pathlib import Path
from typing import Any, List, Tuple, Set
from AutoEval import AutoEval


def _ensure_list_of_str(xs: Any) -> List[str]:
    if xs is None:
        return []
    if isinstance(xs, (list, tuple)):
        return [str(x) for x in xs]
    return [str(xs)]


def _parse_pair_list(items: Any, kind: str) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if items is None:
        return out
    if not isinstance(items, list):
        raise ValueError(f"{kind} must be a YAML sequence (list).")
    for it in items:
        if isinstance(it, (list, tuple)):
            if len(it) >= 2:
                out.append((str(it[0]), str(it[1])))
            elif len(it) == 1:
                out.append((str(it[0]), ""))
            else:
                out.append(("", ""))
        elif isinstance(it, dict):
            first = None
            second = None
            for k in ("name", "env", "first", "argfile", "file"):
                if k in it:
                    first = it[k]
                    break
            for k in ("tag", "data_tag", "second", "remark", "rm"):
                if k in it:
                    second = it[k]
                    break
            if first is None and it:
                vals = list(it.values())
                first = vals[0]
                second = vals[1] if len(vals) > 1 else ""
            if first is None:
                first = ""
            if second is None:
                second = ""
            out.append((str(first), str(second)))
        else:
            out.append((str(it), ""))
    return out


def load_config_from_yaml(
    config_path: Path,
) -> Tuple[List[str], List[str], List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with config_path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception as e:
        raise ValueError(f"Failed to read/parse YAML {config_path}: {e}") from e

    policies_raw = data.get("policies") or data.get("POLICIES")
    seeds_raw = data.get("seeds") or data.get("SEEDS")
    env_list_raw = data.get("env_list") or data.get("ENV_LIST") or data.get("envs")
    remark_list_raw = (
        data.get("remark_list") or data.get("REMARK_LIST") or data.get("remarks")
    )

    # Validate presence
    missing = []
    if policies_raw is None:
        missing.append("policies")
    if seeds_raw is None:
        missing.append("seeds")
    if env_list_raw is None:
        missing.append("env_list")
    if remark_list_raw is None:
        missing.append("remark_list")
    if missing:
        raise ValueError(
            f"Missing required configuration keys in {config_path}: {', '.join(missing)}"
        )

    policies = _ensure_list_of_str(policies_raw)
    seeds = _ensure_list_of_str(seeds_raw)
    env_pairs = _parse_pair_list(env_list_raw, "env_list")
    remark_pairs = _parse_pair_list(remark_list_raw, "remark_list")

    return policies, seeds, env_pairs, remark_pairs


def find_task_success_files(result_root: str) -> List[str]:
    """Recursively find task_success_list.txt files under result_root."""
    if AutoEval is not None and hasattr(AutoEval, "_find_txt_files"):
        try:
            return list(AutoEval._find_txt_files(result_root))
        except Exception:
            pass
    pattern = os.path.join(result_root, "**", "task_success_list.txt")
    return [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]


def extract_fields_from_path(
    result_root: str, fpath: str, env_names_set: Set[str]
) -> Tuple[str, str, str, str]:
    """Extract (policy, env, seed, remark) from task_success_list path."""
    policy = ""
    env = ""
    seed = ""
    remark = ""
    raw_task_key = None

    if AutoEval is not None and hasattr(AutoEval, "_parse_tsk_suc_filepath"):
        try:
            parsed = AutoEval._parse_tsk_suc_filepath(fpath, result_root)
            if parsed and isinstance(parsed, tuple) and len(parsed) == 2:
                policy, raw_task_key = parsed
        except Exception:
            raw_task_key = None

    rel = os.path.relpath(fpath, result_root)
    parts = rel.split(os.sep)

    if not policy or raw_task_key is None:
        policy = parts[1] if len(parts) > 1 else ""
        # find env by matching known env names
        env = ""
        for p in parts[2:]:
            if p in env_names_set:
                env = p
                break
        if not env and len(parts) > 2:
            env = parts[2]
        for p in parts:
            m = re.match(r"^s(\d+)$", p)
            if m:
                seed = m.group(1)
                break
        timestamp = parts[0] if len(parts) > 0 else ""
        if timestamp and policy:
            cand = os.path.join(result_root, timestamp, policy, "setting_remark.txt")
            if os.path.isfile(cand):
                try:
                    with open(cand, "r", encoding="utf-8", errors="ignore") as fh:
                        first = fh.readline().strip()
                        if first:
                            remark = first
                except Exception:
                    remark = ""
            else:
                globpat = os.path.join(
                    result_root, timestamp, policy, "**", "setting_remark.txt"
                )
                found = glob.glob(globpat, recursive=True)
                if found:
                    try:
                        with open(
                            found[0], "r", encoding="utf-8", errors="ignore"
                        ) as fh:
                            first = fh.readline().strip()
                            if first:
                                remark = first
                    except Exception:
                        remark = ""
        return (policy, env, seed, remark)

    # raw_task_key exists
    if raw_task_key:
        parts_rt = raw_task_key.split("/", 1)
        env = parts_rt[0] if parts_rt else ""
    for p in parts:
        m = re.match(r"^s(\d+)$", p)
        if m:
            seed = m.group(1)
            break
    try:
        if AutoEval is not None and hasattr(AutoEval, "_read_remark"):
            remark_val = AutoEval._read_remark(fpath, result_root, policy, raw_task_key)
            remark = remark_val if remark_val else ""
    except Exception:
        remark = ""

    return (policy, env, seed, remark)


def collect_done_set(
    result_root: str, env_names: List[str]
) -> Set[Tuple[str, str, str, str]]:
    """Collect executed combinations by scanning task_success_list files."""
    done: Set[Tuple[str, str, str, str]] = set()
    env_set = set(env_names)
    files = find_task_success_files(result_root)
    for f in files:
        try:
            policy, env, seed, remark = extract_fields_from_path(
                result_root, f, env_set
            )
            if policy and env and seed:
                done.add((policy, env, seed, remark))
        except Exception:
            continue
    return done


def generate_all_combinations(
    policies: List[str],
    seeds: List[str],
    env_pairs: List[Tuple[str, str]],
    remark_pairs: List[Tuple[str, str]],
) -> List[Tuple[str, str, str, str, str, str]]:
    allc: List[Tuple[str, str, str, str, str, str]] = []
    for policy in policies:
        for seed in seeds:
            for env_name, env_tag in env_pairs:
                for remark_argfile, remark_val in remark_pairs:
                    allc.append(
                        (policy, seed, env_name, env_tag, remark_val, remark_argfile)
                    )
    return allc


def filter_unexecuted(all_combos, done_set):
    out = []
    for policy, seed, env_name, env_tag, remark_val, remark_argfile in all_combos:
        key = (policy, env_name, seed, remark_val)
        if key not in done_set:
            out.append((policy, seed, env_name, env_tag, remark_val, remark_argfile))
    return out


def greedy_reduce_sets(
    policies: List[str],
    seeds: List[str],
    env_names: List[str],
    remarks: List[str],
    done_set: Set[Tuple[str, str, str, str]],
    show_progress: bool = False,
):
    """Greedy non-conflicting selection."""

    # copy input lists
    all_p, all_s, all_e, all_r = (
        list(policies),
        list(seeds),
        list(env_names),
        list(remarks),
    )

    p_cnt = Counter()
    s_cnt = Counter()
    e_cnt = Counter()
    r_cnt = Counter()

    set_p = set(all_p)
    set_s = set(all_s)
    set_e = set(all_e)
    set_r = set(all_r)

    for p, e, s, r in done_set:
        if p in set_p and s in set_s and e in set_e and r in set_r:
            p_cnt[p] += 1
            s_cnt[s] += 1
            e_cnt[e] += 1
            r_cnt[r] += 1

    def stable_sort_by_count(orig_list, counter):
        return sorted(orig_list, key=lambda x: (counter.get(x, 0), orig_list.index(x)))

    all_p = stable_sort_by_count(all_p, p_cnt)
    all_s = stable_sort_by_count(all_s, s_cnt)
    all_e = stable_sort_by_count(all_e, e_cnt)
    all_r = stable_sort_by_count(all_r, r_cnt)

    if show_progress:
        print(f"[{greedy_reduce_sets.__name__}] Axis order after initial sorting:")
        print(f"  policies(sorted by conflicts): {all_p}")
        print(f"  seeds(sorted by conflicts): {all_s}")
        print(f"  envs(sorted by conflicts): {all_e}")
        print(f"  remarks(sorted by conflicts): {all_r}")

    def can_add(axis, value, cur_p, cur_s, cur_e, cur_r):
        if axis == "policy":
            for s in cur_s:
                for e in cur_e:
                    for r in cur_r:
                        if (value, e, s, r) in done_set:
                            return False
            return True
        if axis == "seed":
            for p in cur_p:
                for e in cur_e:
                    for r in cur_r:
                        if (p, e, value, r) in done_set:
                            return False
            return True
        if axis == "env":
            for p in cur_p:
                for s in cur_s:
                    for r in cur_r:
                        if (p, value, s, r) in done_set:
                            return False
            return True
        if axis == "remark":
            for p in cur_p:
                for s in cur_s:
                    for e in cur_e:
                        if (p, e, s, value) in done_set:
                            return False
            return True
        return False

    # find initial non-conflicting quad using the sorted lists
    initial_found = False
    cur_p: List[str] = []
    cur_s: List[str] = []
    cur_e: List[str] = []
    cur_r: List[str] = []
    for p in all_p:
        for s in all_s:
            for e in all_e:
                for r in all_r:
                    if (p, e, s, r) not in done_set:
                        cur_p = [p]
                        cur_s = [s]
                        cur_e = [e]
                        cur_r = [r]
                        initial_found = True
                        break
                if initial_found:
                    break
            if initial_found:
                break
        if initial_found:
            break

    if not initial_found:
        if show_progress:
            print(
                f"[{greedy_reduce_sets.__name__}] No initial non-conflicting combination found; returning empty lists"
            )
        return [], [], [], []

    if show_progress:
        print(
            f"[{greedy_reduce_sets.__name__}] Initial seed: p={cur_p[0]}, s={cur_s[0]}, e={cur_e[0]}, r={cur_r[0]}"
        )

    # iteratively add best non-conflicting candidates
    added = True
    while added:
        added = False
        best_gain = 0
        best_choice = None
        remaining = {
            "policy": [x for x in all_p if x not in cur_p],
            "seed": [x for x in all_s if x not in cur_s],
            "env": [x for x in all_e if x not in cur_e],
            "remark": [x for x in all_r if x not in cur_r],
        }
        for axis, values in remaining.items():
            for v in values:
                if can_add(axis, v, cur_p, cur_s, cur_e, cur_r):
                    gain = {
                        "policy": len(cur_s) * len(cur_e) * len(cur_r),
                        "seed": len(cur_p) * len(cur_e) * len(cur_r),
                        "env": len(cur_p) * len(cur_s) * len(cur_r),
                        "remark": len(cur_p) * len(cur_s) * len(cur_e),
                    }[axis]
                    if gain > best_gain:
                        best_gain = gain
                        best_choice = (axis, v)
        if best_choice:
            axis, value = best_choice
            if axis == "policy":
                cur_p.append(value)
            elif axis == "seed":
                cur_s.append(value)
            elif axis == "env":
                cur_e.append(value)
            elif axis == "remark":
                cur_r.append(value)
            added = True
            if show_progress:
                print(
                    f"[{greedy_reduce_sets.__name__}] Added {axis}='{value}', new sizes: p={len(cur_p)} s={len(cur_s)} e={len(cur_e)} r={len(cur_r)} (gain={best_gain})"
                )

    return cur_p, cur_s, cur_e, cur_r


def main(argv=None) -> int:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    default_config_path = (
        Path(__file__).resolve().parent / "configs" / "CheckMissingResults.yaml"
    )
    p.add_argument(
        "--config",
        default=default_config_path,
        type=Path,
        help="optional path to YAML config file",
    )
    p.add_argument(
        "--result_root",
        default="./result",
        help="root directory where results are stored",
    )
    args = p.parse_args(argv)
    config_path: Path = args.config
    try:
        policies, seeds, env_pairs, remark_pairs = load_config_from_yaml(config_path)
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}", file=sys.stderr)
        return 2

    # ensure seeds are strings
    seeds = [str(s) for s in seeds]

    env_names = [e for e, _ in env_pairs]
    remark_values = [r for _, r in remark_pairs]

    done_set = collect_done_set(str(args.result_root), env_names)
    all_combos = generate_all_combinations(policies, seeds, env_pairs, remark_pairs)
    missing = filter_unexecuted(all_combos, done_set)

    print("# Summary")
    print(f"# config file: {config_path}")
    print(f"# configured policies: {len(policies)} -> {policies}")
    print(f"# configured seeds: {len(seeds)} -> {seeds}")
    print(f"# configured envs: {len(env_pairs)} -> {env_names}")
    print(f"# configured remarks: {len(remark_pairs)} -> {remark_values}")
    print(f"# discovered executed combinations: {len(done_set)}")
    print(f"# full product size: {len(all_combos)}")
    print(f"# missing combinations: {len(missing)}")
    print()

    reduced_p, reduced_s, reduced_e, reduced_r = greedy_reduce_sets(
        policies, seeds, env_names, remark_values, done_set
    )
    print("# Greedy reduction result:")
    print('  POLICIES_NEW="' + " ".join(reduced_p) + '"')
    print('  SEEDS_NEW="' + " ".join(reduced_s) + '"')
    print()
    print("  ## ENV_LIST_NEW (each line: ENV_NAME DATA_TAG)")
    for en in reduced_e:
        dt = next((dt for (n, dt) in env_pairs if n == en), "")
        print(f"  {en} {dt}")
    print()
    print("  ## REMARK_LIST_NEW (each line: ARGFILE REMARK)")
    for rv in reduced_r:
        argfile = next((af for (af, rm) in remark_pairs if rm == rv), "")
        print(f"  {(argfile, rv)}")
    print()
    reduced_size = len(reduced_p) * len(reduced_s) * len(reduced_e) * len(reduced_r)
    print(f"  ## reduced product size: {reduced_size}")
    reduced_all = generate_all_combinations(
        reduced_p,
        reduced_s,
        [(e, next(dt for (n, dt) in env_pairs if n == e)) for e in reduced_e],
        [(next(af for (af, rm) in remark_pairs if rm == r), r) for r in reduced_r],
    )
    reduced_missing = filter_unexecuted(reduced_all, done_set)
    print(
        f"  ## missing in reduced product: {len(reduced_missing)} (should equal reduced product size)\n"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
