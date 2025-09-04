#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CheckMissingResults.py

Scan result_root for completed runs, generate the full product from configured
POLICIES/SEEDS/ENV_LIST/REMARK_LIST, and print/save unexecuted combinations.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import yaml
from AutoEval import AutoEval


class CheckMissingResults:
    def __init__(
        self,
        result_root: str,
        config_paths: List[Path],
        output_file: Optional[Path] = None,
        n_show_first: int = 10,
    ):
        self.result_root = result_root
        self.config_paths = config_paths
        self.output_file = output_file
        self.n_show_first = n_show_first

        self.merged_missing: List[Tuple[str, str, str, str, str, str]] = []
        self.merged_done_set: Set[Tuple[str, str, str, str, str]] = set()
        self.merged_all_combos: List[Tuple[str, str, str, str, str, str]] = []

    @staticmethod
    def _ensure_list_of_str(xs: Any) -> List[str]:
        if xs is None:
            return []
        if isinstance(xs, (list, tuple)):
            return [str(x) for x in xs]
        return [str(xs)]

    @staticmethod
    def parse_yaml_pair_list(items: Any, kind: str) -> List[Tuple[str, str]]:
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

    def load_config_from_yaml(self, config_path: Path):
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

        policies = self._ensure_list_of_str(policies_raw)
        seeds = self._ensure_list_of_str(seeds_raw)
        env_pairs = self.parse_yaml_pair_list(env_list_raw, "env_list")
        remark_pairs = self.parse_yaml_pair_list(remark_list_raw, "remark_list")

        return policies, seeds, env_pairs, remark_pairs

    def find_task_success_files(self) -> List[str]:
        if AutoEval is not None and hasattr(AutoEval, "_find_txt_files"):
            try:
                return list(AutoEval._find_txt_files(self.result_root))
            except Exception:
                pass
        pattern = os.path.join(self.result_root, "**", "task_success_list.txt")
        return [p for p in glob.glob(pattern, recursive=True) if os.path.isfile(p)]

    def extract_fields_from_path(
        self, fpath: str, env_names_set: Set[str]
    ) -> Tuple[str, str, str, str]:
        policy = ""
        env = ""
        seed = ""
        remark = ""
        raw_task_key = None

        if AutoEval is not None and hasattr(AutoEval, "_parse_tsk_suc_filepath"):
            try:
                parsed = AutoEval._parse_tsk_suc_filepath(fpath, self.result_root)
                if parsed and isinstance(parsed, tuple) and len(parsed) == 2:
                    policy, raw_task_key = parsed
            except Exception:
                raw_task_key = None

        rel = os.path.relpath(fpath, self.result_root)
        parts = rel.split(os.sep)

        if not policy or raw_task_key is None:
            policy = parts[1] if len(parts) > 1 else ""
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
                cand = os.path.join(
                    self.result_root, timestamp, policy, "setting_remark.txt"
                )
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
                        self.result_root, timestamp, policy, "**", "setting_remark.txt"
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
                remark_val = AutoEval._read_remark(
                    fpath, self.result_root, policy, raw_task_key
                )
                remark = remark_val if remark_val else ""
        except Exception:
            remark = ""

        return (policy, env, seed, remark)

    def collect_done_set(
        self, env_names: List[str]
    ) -> Set[Tuple[str, str, str, str, str]]:
        done: Set[Tuple[str, str, str, str, str]] = set()
        env_set = set(env_names)
        files = self.find_task_success_files()
        for f in files:
            try:
                policy, env, seed, remark = self.extract_fields_from_path(f, env_set)
                data_tag = ""
                try:
                    rel = os.path.relpath(f, self.result_root)
                    parts = rel.split(os.sep)
                    if env in parts:
                        idx = parts.index(env)
                        if idx + 1 < len(parts):
                            data_tag = parts[idx + 1]
                except Exception:
                    data_tag = ""
                if policy and env and seed:
                    done.add((policy, env, data_tag, seed, remark))
            except Exception:
                continue
        return done

    @staticmethod
    def generate_all_combinations(
        policies: List[str],
        seeds: List[str],
        env_pairs: List[Tuple[str, str]],
        remark_pairs: List[Tuple[str, str]],
    ) -> List[Tuple[str, str, str, str, str, str]]:
        allc: List[Tuple[str, str, str, str, str, str]] = []
        for policy in policies:
            for seed in seeds:
                for env_name, data_tag in env_pairs:
                    for remark_argfile, remark_val in remark_pairs:
                        allc.append(
                            (
                                policy,
                                seed,
                                env_name,
                                data_tag,
                                remark_val,
                                remark_argfile,
                            )
                        )
        return allc

    @staticmethod
    def filter_unexecuted(all_combos, done_set):
        out = []
        for policy, seed, env_name, data_loc, remark_val, remark_argfile in all_combos:
            data_tag = AutoEval.extract_dataset_tag(data_loc)
            key = (policy, env_name, data_tag, seed, remark_val)
            if key not in done_set:
                out.append(
                    (policy, seed, env_name, data_loc, remark_val, remark_argfile)
                )
        return out

    @staticmethod
    def write_missing_to_csv(missing, output_path: Path) -> None:
        """Write the list of missing combinations to a CSV file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = [
            "policy",
            "seed",
            "env_name",
            "data_tag",
            "remark_val",
            "remark_argfile",
        ]
        with output_path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for row in sorted(missing):
                writer.writerow(row)

    @staticmethod
    def format_missing_row(row: Tuple[str, str, str, str, str, str]) -> str:
        policy, seed, env_name, data_tag, remark_val, remark_argfile = row
        return f"policy={policy} seed={seed} env={env_name} (tag={data_tag}) remark='{remark_val}' argfile='{remark_argfile}'"

    def run(self) -> int:
        print(f"[{self.__class__.__name__}] Starting run process")
        for config_path in self.config_paths:
            try:
                print(f"[{self.__class__.__name__}] Loading config {config_path}")
                policies, seeds, env_pairs, remark_pairs = self.load_config_from_yaml(
                    config_path
                )
            except Exception as e:
                print(
                    f"[{self.__class__.__name__}] ERROR: Failed to load config {config_path}: {e}"
                )
                continue

            seeds = [str(s) for s in seeds]
            env_names = [e for e, _ in env_pairs]
            remark_values = [r for _, r in remark_pairs]

            done_set = self.collect_done_set(env_names)
            self.merged_done_set.update(done_set)

            all_combos = self.generate_all_combinations(
                policies, seeds, env_pairs, remark_pairs
            )
            self.merged_all_combos.extend(all_combos)

            missing = self.filter_unexecuted(all_combos, done_set)
            self.merged_missing.extend(missing)

            print(f"[{self.__class__.__name__}] Summary for {config_path}:")
            print(
                f"[{self.__class__.__name__}]   Configured policies: {len(policies)} -> {policies}"
            )
            print(
                f"[{self.__class__.__name__}]   Configured seeds: {len(seeds)} -> {seeds}"
            )
            print(
                f"[{self.__class__.__name__}]   Configured envs: {len(env_pairs)} -> {env_names}"
            )
            print(
                f"[{self.__class__.__name__}]   Configured remarks: {len(remark_pairs)} -> {remark_values}"
            )
            print("")

        self.merged_missing = list({tuple(row) for row in self.merged_missing})

        if self.output_file:
            out_path = self.output_file
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            out_path = Path(__file__).resolve().parent / "auto_eval_jobs.csv"

        if out_path.exists():
            base, ext = os.path.splitext(out_path.name)
            backup_path = out_path.parent / f"{base}_old{ext}"
            counter = 1
            while backup_path.exists():
                backup_path = out_path.parent / f"{base}_old_{counter}{ext}"
                counter += 1
            out_path.rename(backup_path)
            print(
                f"[{self.__class__.__name__}] Existing file moved: {out_path} -> {backup_path}"
            )

        self.write_missing_to_csv(self.merged_missing, out_path)

        num_show = max(0, self.n_show_first)
        print(
            f"[{self.__class__.__name__}] Showing first {num_show} missing combinations:"
        )
        for idx, row in enumerate(self.merged_missing[:num_show], start=1):
            print(
                f"[{self.__class__.__name__}] {idx:3d}. {self.format_missing_row(row)}"
            )
        if len(self.merged_missing) == 0:
            print(f"[{self.__class__.__name__}]   (none)")

        print(
            f"[{self.__class__.__name__}] Executed combinations: {len(self.merged_done_set)}"
        )
        print(
            f"[{self.__class__.__name__}] Full product size: {len(self.merged_all_combos)}"
        )
        print(
            f"[{self.__class__.__name__}] Missing combinations: {len(self.merged_missing)}"
        )
        print(f"[{self.__class__.__name__}] Saved to: {out_path}")

        print(f"[{self.__class__.__name__}] Run finished")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    default_config_dir = Path(__file__).resolve().parent / "configs"
    default_config_paths = list(default_config_dir.glob("*.yaml"))
    parser.add_argument(
        "--config",
        default=default_config_paths,
        type=Path,
        nargs="+",
        help="paths to YAML config files (can specify multiple)",
    )
    parser.add_argument(
        "--result_root",
        default="./result",
        help="root directory where results are stored",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        type=Path,
        help="Path to save missing combinations as CSV",
    )
    parser.add_argument(
        "--n_show_first",
        default=10,
        type=int,
        help="Number of first missing combinations to print to stdout",
    )
    args = parser.parse_args()

    checker = CheckMissingResults(
        args.result_root, args.config, args.output_file, args.n_show_first
    )
    sys.exit(checker.run())
