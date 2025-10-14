# Flow Policy

## Install
See [here](../../../doc/install.md#flow-policy) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Data preprocessing
Decide the parameters of the bounding box for cropping point clouds:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./misc/VisualizePointCloud.py ./dataset/<dataset_name>/<rmb_file_name>
```
After adjusting the bounding box position and size, as well as the point cloud orientation, using the keyboard or a 3D mouse, press the "P" key to print the bounding box parameters.

Generate and store point clouds from RGB and depth images:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./misc/AddPointCloudToRmbData.py ./dataset/<dataset_name> --min_bound <x, y, z> --max_bound <x, y, z> --rpy_angle <roll, pitch, yaw>
```
For bounding box parameters such as `min_bound`, you can use the results adjusted in `VisualizePointCloud.py`.

> [!TIP]
> You can download the dataset with point cloud added to [MujocoUR5eCable_Dataset30](https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/dataset_list.md#ur5e--demo-30) from [here](https://www.dropbox.com/scl/fo/kkj0nj1guc95j24fb8zux/AC7hJOTcLPcu7bG668d-TKQ?rlkey=p14dattkal9upafsezssymslk&dl=1).

## Model training
Train a model:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py FlowPolicy --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/FlowPolicy/<checkpoint_name>
```

> [!NOTE]
> If you encounter the following error,
> ```console
> ImportError: cannot import name 'cached_download' from 'huggingface_hub'
> ```
> downgrade `huggingface_hub` by the following command.
> ```console
> $ pip install huggingface_hub==0.21.4
> ```

## Policy rollout
Run a trained policy:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Rollout.py FlowPolicy MujocoUR5eCable --checkpoint ./checkpoint/FlowPolicy/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@inproceedings{FlowPolicy_AAAI2025,
  author = {Qinglun Zhang and Zhen Liu and Haoqiang Fan and Guanghui Liu and Bing Zeng and Shuaicheng Liu},
  title = {FlowPolicy: Enabling Fast and Robust 3D Flow-based Policy via Consistency Flow Matching for Robot Manipulation},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {39},
  number = {14},
  pages = {14754--14762},
  year = {2025}
}
```
