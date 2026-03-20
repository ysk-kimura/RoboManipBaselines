# Isaac GR00T

## Install

You need four virtual environments for data preparation, LeRobot data conversion, training, and rollout.

### Env 1: Environment for data preparation

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../../doc/install.md#common-installation).

Install [LeRobot](https://github.com/huggingface/lerobot) by the following commands.
```console
# Go to the top directory of this repository
$ cd third_party/lerobot
$ pip install -e .
```

### Env 2: Environment for LeRobot data conversion

Install [lerobot_conversion](https://github.com/yu-kitagawa/Isaac-GR00T/tree/main/scripts/lerobot_conversion) by the following commands.
```console
# Go to any directory
$ git clone https://github.com/yu-kitagawa/Isaac-GR00T
$ cd Isaac-GR00T/scripts/lerobot_conversion
$ pip install -e .
```

### Env 3: Environment for training

Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) by the following commands.
```console
# Go to any directory
$ git clone https://github.com/yu-kitagawa/Isaac-GR00T
$ cd Isaac-GR00T
$ pip install -e .
$ pip install wheel packaging ninja
$ pip install --no-build-isolation flash-attn==2.7.4.post1
```

> [!NOTE]
> When running on ABCI, the following command:
> ```console
> $ pip install --no-build-isolation flash-attn==2.7.4.post1
> ```
> must be executed on a GPU node.
>
> Before running this command on a GPU node, you need to load the CUDA module:
> ```console
> $ source /etc/profile.d/modules.sh
> $ module load cuda/12.6/12.6.1
> ```
>
> and set a temporary directory:
> ```console
> $ export TMPDIR=$HOME/tmp
> $ mkdir -p $TMPDIR
> ```

### Env 4: Environment for rollout

Install [GR00T](https://github.com/yu-kitagawa/Isaac-GR00T) as in Env 3.

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../../doc/install.md#common-installation).

If `numpy>=2.0.0`, install `numpy<2.0.0`.
```console
$ pip install "numpy<2.0.0"
```

## Data Preparation

Convert RMB dataset into the LeRobot dataset format.

```console
# Use Env 1
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python misc/ConvertRmbDataToLerobot.py <rmb_dataset_dir> --output_dir <dataset_root>/<dataset_repo_id>
```

## LeRobot Data Conversion

Convert from LeRobot v3 to v2.

```console
# Use Env 2
# Go to Isaac-GR00T directory
$ cd scripts/lerobot_conversion
$ python convert_v3_to_v2.py --root <dataset_root> --repo-id <dataset_repo_id>
$ cp <dataset_root>/<dataset_repo_id>_v3.0/meta/modality.json <dataset_root>/<dataset_repo_id>/meta/
```

## Model Training

Train a model.
Here is an example command for UR5e.

```console
# Use Env 3
# Go to Isaac-GR00T directory
$ export NUM_GPUS=1
$ CUDA_VISIBLE_DEVICES=0 python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.6-3B \
  --dataset-path <dataset_root>/<dataset_repo_id> \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/UR5e/ur5e_config.py \
  --num-gpus $NUM_GPUS \
  --output-dir <checkpoint_dir> \
  --save-total-limit 5 \
  --save-steps 60000 \
  --max-steps 60000 \
  --global-batch-size 64 \
  --color-jitter-params brightness 0.3 contrast 0.4 saturation 0.5 hue 0.08 \
  --dataloader-num-workers 4
```

> [!NOTE]
> When running on ABCI, you need to load the CUDA module:
> ```console
> $ source /etc/profile.d/modules.sh
> $ module load cuda/12.6/12.6.1
> ```

## Policy rollout

Run a trained policy in the simulator.

```console
# Use Env 4
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py Gr00t <task_name> --checkpoint <checkpoint_dir> --world_idx 0 --no_plot --task_desc <task_description_text>
```
