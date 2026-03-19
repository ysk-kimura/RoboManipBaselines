# lerobot pi0

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
Install [lerobot](https://github.com/yu-kitagawa/lerobot) by the following commands.
```console
$ # Go to any directory
$ git clone https://github.com/yu-kitagawa/lerobot.git
$ cd lerobot
$ pip install -e ".[pi]"
```

3. **Env to rollout**
Install [lerobot](https://github.com/yu-kitagawa/lerobot) as in 2.

Install [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines) according to [here](../../README.md#Install).

## advance preparation
Create a huggingface account and search google/paligemma-3b-pt-224 model on the website.

There is an item marked "You need to agree to share your contact information to access this model".
Click on "Agree and access repository" in that item and follow the steps to complete the procedure.

## Dataset preparation

Convert your RMB data to lerobot dataset.

```console
$ Use Env 1
$ Go to robo_manip_baselines directry
$ python misc/ConvertRmbDataToLerobot.py --path <your RMB data dir> --repo_id <your name>/<data name>
```

## Model Training

Train the model. The trained weights are saved in the `outputs` folder.

Here is an example command for UR5e.

Once you've run this command and confirmed that the pi0_base model has been downloaded, please stop.
Next, please delete all `observation.images` keys from the `input_features` section in the `config.json` file for the pi0_base model(`~/.cache/huggingface/hub/models--lerobot--pi0_base/snapshots/<omitted>/config.json`).

```console
$ Use Env 2
$ Go to lerobot directory
$ python src/lerobot/scripts/lerobot_train.py --dataset.repo_id=<your name>/<data name> --dataset.root=<your data dir> --policy.type=pi0 --job_name=pi0_finetune --policy.pretrained_path=lerobot/pi0_base --policy.repo_id=local_repo --policy.compile_model=true --policy.gradient_checkpointing=false --policy.dtype=bfloat16 --policy.freeze_vision_encoder=true --policy.train_expert_only=true --policy.push_to_hub=false --policy.input_features=null --policy.input_features='{"observation.images.front_rgb": {"shape":[3,224,224], "type":"VISUAL"},"observation.images.hand_rgb": {"shape":[3,224,224], "type":"VISUAL"},"observation.images.left_rgb": {"shape":[3,224,224], "type":"VISUAL"},"observation.images.right_rgb": {"shape":[3,224,224], "type":"VISUAL"},"observation.state": {"shape":[7], "type":"STATE"}}' --policy.output_features='{"action": {"shape":[7], "type":"ACTION"}}' --policy.n_action_steps=8 --policy.chunk_size=16 --batch_size=16
```

## Policy rollout

Run a trained policy in the simulator.

```console
$ Use Env 3
$ Go to robo_manip_baselines directry
$ python ./bin/Rollout.py Pi0 <task name> --checkpoint <checkpoint dir> --world_idx 0 --no_plot --task_desc <task description text>
```
