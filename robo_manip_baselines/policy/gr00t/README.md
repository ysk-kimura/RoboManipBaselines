# Isaac GR00T

## Install

You need three virtual environments for data preparation, training, and rollout.

**Env 1: Environment for data preparation**

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../../doc/install.md#common-installation).

Install [LeRobot](https://github.com/huggingface/lerobot) by the following commands.
```console
# Go to the top directory of this repository
$ cd third_party/lerobot
$ pip install -e .
```

**Env 2: Environment for training**

Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) by the following commands.
```console
$ # Go to any directory
$ git clone https://github.com/yu-kitagawa/Isaac-GR00T
$ cd Isaac-GR00T
$ pip install -e .
$ pip install wheel packaging ninja
$ pip install --no-build-isolation flash-attn==2.7.4.post1
```

**Env 3: Environment for rollout**

Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) as in Env 2.

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../../doc/install.md#common-installation).

If `numpy>=2.0.0`, install `numpy<2.0.0`.
```console
$ pip install "numpy<2.0.0"
```

## Dataset Preparation

Convert your RMB dataset into the LeRobot dataset format.

```console
# Use Env 1
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python misc/ConvertRmbDataToLerobot.py <RMB_data_dir> --repo_id <user_name>/<dataset_name>
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.

Here is an example command for UR5e.

```console
# Use Env 2
# Go to Isaac-GR00T directory
$ python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.6-3B --dataset-path <data_dir> --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/UR5e/ur5e_config.py --num-gpus <num_gpus> --output-dir ./log/<log_name> --save-total-limit 5 --save-steps 60000 --max-steps 60000 --global-batch-size 64 --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader-num-workers 4
```

## Policy rollout

Run a trained policy in the simulator.

```console
# Use Env 3
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Gr00t <task_name> --checkpoint <checkpoint_dir> --world_idx 0 --no_plot --task_desc <task_description_text>
```
