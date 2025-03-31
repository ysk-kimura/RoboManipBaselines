# [RoboManipBaselines](https://isri-aist.github.io/RoboManipBaselines-ProjectPage)
Software that integrates various imitation learning methods and benchmark task environments to provide baselines for robot manipulation

[![CI-install](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/install.yml/badge.svg)](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/install.yml)
[![CI-pre-commit](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/isri-aist/RoboManipBaselines/actions/workflows/pre-commit.yml)
[![LICENSE](https://img.shields.io/github/license/isri-aist/RoboManipBaselines)](https://github.com/isri-aist/RoboManipBaselines/blob/master/LICENSE)

https://github.com/user-attachments/assets/ba4a772f-0de5-47da-a4ec-bdcbf13d7d58

## Quick start
See [the quick start](./doc/quick_start.md) to collect data in the MuJoCo simulation and train and rollout the ACT policy.

## Install
See [the installation guide](./doc/install.md).

## Policies
### MLP
See [mlp](./robo_manip_baselines/policy/mlp).

### SARNN
See [sarnn](./robo_manip_baselines/policy/sarnn).

### ACT
See [act](./robo_manip_baselines/policy/act).

### DiffusionPolicy
See [diffusion_policy](./robo_manip_baselines/policy/diffusion_policy).

## Data
See [the dataset list](./doc/dataset_list.md) for demonstration datasets.

See [the learned parameters](./doc/learned_parameters.md) for policies learned from these datasets.

## Teleoperation
See [teleop](./robo_manip_baselines/teleop).

## Environments
See [the environment catalog](doc/environment_catalog.md) for a full list of environments.

See [envs](./robo_manip_baselines/envs) for installation procedures for each environment.

## Misc
See [misc](./robo_manip_baselines/misc) for miscellaneous standalone scripts.

## Evaluation results
See [the evaluation results](doc/evaluation_results.md).

## Contribute
If you would like to contribute to this repository, please check out [the contribution guide](./CONTRIBUTING.md).

## License
Files that originate from this repository are subject to the BSD 2-Clause License. If a file explicitly states a different license, or if there are different license files in a directory, those licenses will take precedence. For files in third-party directories, please follow the respective licenses.

## Citation
You can cite this work with:
```bib
@software{RoboManipBaselines_GitHub2024,
author = {Murooka, Masaki and Motoda, Tomohiro and Nakajo, Ryoichi},
title = {{RoboManipBaselines}},
url = {https://github.com/isri-aist/RoboManipBaselines},
version = {1.0.0},
year = {2024}
month = dec,
}
```
