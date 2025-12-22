import argparse
import importlib
import importlib.util
import inspect
import os
import sys

import yaml

from robo_manip_baselines.common.base.RolloutBase import RolloutBase
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
            help="policy",
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
        parser.add_argument(
            "--checkpoint",
            type=str,
            nargs="+",
            required=True,
            help="checkpoint file(s)",
        )

        self.args, remaining_argv = parser.parse_known_args()
        self._remaining_argv = remaining_argv
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

        checkpoint_list = self.args.checkpoint
        if checkpoint_list is None:
            raise ValueError("`checkpoint` must be specified in config file.")
        if not isinstance(checkpoint_list, (list, tuple)):
            raise TypeError("`checkpoint` must be a list of paths (one per policy).")
        if len(checkpoint_list) != len(self.args.policy):
            raise ValueError(
                f"Number of policies ({len(self.args.policy)}) does not match "
                f"number of checkpoints ({len(checkpoint_list)})."
            )

        rollout_ensemble = RolloutEnsembleBase()
        # Pass OperationEnvClass so the ensemble can instantiate and initialize the environment
        _ = rollout_ensemble.setup_env(OperationEnvClass, **config)

        rollout_inst_list = []
        for pol_idx, policy_name in enumerate(self.args.policy):
            policy_module = importlib.import_module(
                f"{self.policy_parent_module_str}.{camel_to_snake(policy_name)}"
            )
            RolloutPolicyClass = getattr(policy_module, f"Rollout{policy_name}")

            class Rollout(OperationEnvClass, RolloutPolicyClass):
                _rpc = RolloutPolicyClass

                def __init__(self, **kwargs):
                    super().__init__(**kwargs)

                @property
                def policy_name(self):
                    return remove_prefix(self._rpc.__name__, "Rollout")

            pconfig = dict(config)

            sys.argv = [
                sys.argv[0],
                "--checkpoint",
                checkpoint_list[pol_idx],
            ] + getattr(self, "_remaining_argv", [])
            rollout_inst = object.__new__(Rollout)
            op_template = getattr(rollout_ensemble, "_operation_template", None)
            sig = inspect.signature(OperationEnvClass.__init__)
            op_arg_names = [
                name
                for name, param in sig.parameters.items()
                if name != "self"
                and param.kind
                in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                )
            ]
            copied = []
            if op_template is not None:
                for name in op_arg_names:
                    if hasattr(op_template, name) and not hasattr(rollout_inst, name):
                        value = getattr(op_template, name)
                        if value is not None:
                            setattr(rollout_inst, name, value)
                            copied.append(name)
            for name in op_arg_names:
                if name in pconfig and not hasattr(rollout_inst, name):
                    setattr(rollout_inst, name, pconfig[name])
                    if name not in copied:
                        copied.append(name)
            RolloutPolicyClass.__init__(
                rollout_inst, env=rollout_ensemble.env, **pconfig
            )
            RolloutBase.__init__(rollout_inst, env=rollout_ensemble.env, argv=sys.argv)
            rollout_inst._operation_template = op_template
            rollout_inst.env = rollout_ensemble.env
            rollout_inst.reset_flag = getattr(rollout_inst, "reset_flag", True)
            rollout_inst.quit_flag = getattr(rollout_inst, "quit_flag", False)
            rollout_inst.inference_duration_list = getattr(
                rollout_inst, "inference_duration_list", []
            )
            rollout_inst_list.append(rollout_inst)
        if not rollout_inst_list:
            raise RuntimeError("No rollout instances. Check policies.")
        rollout_ensemble.set_rollout_inst_list(rollout_inst_list)
        rollout_ensemble.run()


if __name__ == "__main__":
    main = RolloutEnsembleMain()
    main.run()
