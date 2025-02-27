import argparse
import importlib
import re
import sys

import yaml


def camel_to_snake(name):
    """Converts camelCase or PascalCase to snake_case (also converts the first letter to lowercase)"""
    name = re.sub(
        r"([a-z0-9])([A-Z])", r"\1_\2", name
    )  # Insert '_' between a lowercase/number and an uppercase letter
    name = re.sub(
        r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name
    )  # Insert '_' between consecutive uppercase letters followed by a lowercase letter
    name = name[0].lower() + name[1:]  # Convert the first letter to lowercase
    return name.lower()


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="This is a meta argument parser for the rollout switching between different policies and environments. The actual arguments are handled by another internal argument parser.",
        add_help=False,
    )
    parser.add_argument(
        "rollout_policy",
        type=str,
        choices=["Mlp", "Sarnn", "Act", "DiffusionPolicy"],
        help="rollout policy",
    )
    parser.add_argument("rollout_env", type=str, help="rollout environment")
    parser.add_argument("--rollout_config", type=str, help="rollout configuration file")
    parser.add_argument(
        "-h", "--help", action="store_true", help="Show this help message and continue"
    )

    args, remaining_argv = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_argv
    if args.help:
        parser.print_help()
        sys.argv += ["--help"]

    if "Isaac" in args.rollout_env:
        from isaacgym import (
            gymapi,  # noqa: F401
            gymtorch,  # noqa: F401
            gymutil,  # noqa: F401
        )

    policy_module = importlib.import_module(
        f"robo_manip_baselines.{camel_to_snake(args.rollout_policy)}"
    )
    RolloutPolicyClass = getattr(policy_module, f"Rollout{args.rollout_policy}")

    rollout_module = importlib.import_module("robo_manip_baselines.rollout")
    RolloutEnvClass = getattr(rollout_module, f"Rollout{args.rollout_env}")

    class Rollout(RolloutPolicyClass, RolloutEnvClass):
        pass

    if args.rollout_config is None:
        rollout_config = {}
    else:
        with open(args.rollout_config, "r") as f:
            rollout_config = yaml.safe_load(f)

    rollout = Rollout(**rollout_config)
    rollout.run()


if __name__ == "__main__":
    main()
