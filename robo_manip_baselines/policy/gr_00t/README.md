# Isaac GR00T

## Install

1. **Env to prepare dataset**
Install [lerobot](https://github.com/huggingface/lerobot) by the following commands.
```console
$ # Go to any directory
$ git clone https://github.com/huggingface/lerobot.git
$ cd lerobot
$ pip install -e .
```

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../README.md#Install).

2. **Env to train**
Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) by the following commands.
```console
$ # Go to any directory
$ git clone https://github.com/yu-kitagawa/Isaac-GR00T
$ cd Isaac-GR00T
$ pip install -e .
$ pip install wheel packaging ninja
$ pip install --no-build-isolation flash-attn==2.7.4.post1
```

3. **Env to rollout**
Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) as in 2.

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../README.md#Install).

If `numpy>=2.0.0`, install `numpy<2.0.0`.
```console
$ pip install "numpy<2.0.0"

## Dataset preparation

Convert your RMB data to lerobot dataset.

```console
$ Use Env 1
$ Go to robo_manip_baselines directry
$ python misc/ConvertRmbDataToLerobot.py --path <your RMB data dir> --repo_id <your name>/<data name>
```

## Model Training

Train the model. The trained weights are saved in the `log` folder.

Here is an example command for UR5e.

```console
$ Use Env 2
$ Go to Isaac GR00T directory
$ python gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.6-3B --dataset-path <your data dir> --embodiment-tag NEW_EMBODIMENT --modality-config-path examples/UR5e/ur5e_config.py --num-gpus <num gpus> --output-dir ./log/<any name> --save-total-limit 5 --save-steps 60000 --max-steps 60000 --global-batch-size 64 --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 --dataloader-num-workers 4
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ Use Env 3
$ Go to robo_manip_baselines directry
$ python ./bin/Rollout.py Gr00t <task name> --checkpoint <checkpoint dir> --world_idx 0　--no_plot --task_desc <task description text>
```
