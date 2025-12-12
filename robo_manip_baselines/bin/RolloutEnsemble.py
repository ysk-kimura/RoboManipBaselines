import argparse
import importlib
import importlib.util
import os
import sys

import yaml

from robo_manip_baselines.common.ensemble.RolloutEnsembleBase import RolloutEnsembleBase


class RolloutEnsembleMain:
    operation_parent_module_str = "robo_manip_baselines.envs.operation"
    policy_parent_module_str = "robo_manip_baselines.policy"
    policy_choices = [
        "Mlp",
        "Sarnn",
        "Act",
        "MtAct",
        "DiffusionPolicy",
        "DiffusionPolicy3d",
        "FlowPolicy",
        "ManiFlowPolicy",
    ]

    def __init__(self):
        self.setup_args()

    def setup_args(self):
        env_utils_spec = importlib.util.spec_from_file_location(
            "EnvUtils",
            os.path.join(os.path.dirname(__file__), "..", "common/utils/EnvUtils.py"),
        )
        env_utils_module = importlib.util.module_from_spec(env_utils_spec)
        env_utils_spec.loader.exec_module(env_utils_module)

        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            description="This is a meta argument parser for the rollout ensemble, switching between different policies and environments. The actual arguments are handled by another internal argument parser.",
            fromfile_prefix_chars="@",
            add_help=False,
        )
        parser.add_argument(
            "policy",
            type=str,
            nargs="+",
            default=None,
            choices=self.policy_choices,
            help="policy(s)",
        )
        parser.add_argument(
            "env",
            type=str,
            help="environment",
            choices=env_utils_module.get_env_names(
                operation_parent_module_str=self.operation_parent_module_str
            ),
        )
        parser.add_argument("--config", type=str, help="configuration file")
        parser.add_argument(
            "-h",
            "--help",
            action="store_true",
            help="Show this help message and continue",
        )

        self.args, remaining_argv = parser.parse_known_args()
        sys.argv = [sys.argv[0]] + remaining_argv
        if self.args.policy is None or self.args.env is None:
            parser.print_help()
            sys.exit(1)
        elif self.args.help:
            parser.print_help()
            print("\n================================\n")
            sys.argv += ["--help"]

    def run(self):
        if "Isaac" in self.args.env:
            from isaacgym import (
                gymapi,  # noqa: F401
                gymtorch,  # noqa: F401
                gymutil,  # noqa: F401
            )

        # This includes pytorch import, so it must be later than isaac import
        from robo_manip_baselines.common import camel_to_snake, remove_prefix

        operation_module = importlib.import_module(
            f"{self.operation_parent_module_str}.Operation{self.args.env}"
        )
        OperationEnvClass = getattr(operation_module, f"Operation{self.args.env}")

        if self.args.config is None:
            config = {}
        else:
            with open(self.args.config, "r") as f:
                config = yaml.safe_load(f)

        rollouts = []
        main_env = None
        for pol_idx, policy_name in enumerate(self.args.policy):
            policy_module = importlib.import_module(
                f"{self.policy_parent_module_str}.{camel_to_snake(policy_name)}"
            )
            RolloutPolicyClass = getattr(policy_module, f"Rollout{policy_name}")

            class Rollout(OperationEnvClass, RolloutPolicyClass):
                _rpc = RolloutPolicyClass

                def __init__(self, pol_idx=None, **kwargs):
                    super().__init__(pol_idx=pol_idx, **kwargs)

                @property
                def policy_name(self):
                    return remove_prefix(self._rpc.__name__, "Rollout")

            try:
                if main_env is None:
                    rollout_inst = Rollout(pol_idx=pol_idx, **config)
                    main_env = rollout_inst.env
                else:
                    rollout_inst = Rollout(pol_idx=pol_idx, env=main_env, **config)
            except TypeError as e:
                raise TypeError(f"Init fail '{policy_name}': {e}") from e
            rollouts.append(rollout_inst)
        if len(rollouts) == 0:
            raise RuntimeError("No rollout instances. Check policies.")
        if len(rollouts) == 0:
            raise RuntimeError("No rollout instances. Check policies.")
        ensemble = RolloutEnsembleBase(rollout_instances=rollouts)
        ensemble.run()


if __name__ == "__main__":
    main = RolloutEnsembleMain()
    main.run()
