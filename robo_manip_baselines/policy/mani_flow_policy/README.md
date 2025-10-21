# ManiFlow Policy

## Install
See [here](../../../doc/install.md#maniflow-policy) for installation.

## Dataset preparation
Collect demonstration data by [teleoperation](../../teleop).

## Data preprocessing (for pointcloud policy)
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
Train a model with camera images:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py ManiFlowPolicy image --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/ManiFlowPolicy/<checkpoint_name>
```

Train a model with pointcloud:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Train.py ManiFlowPolicy pointcloud --dataset_dir ./dataset/<dataset_name> --checkpoint_dir ./checkpoint/ManiFlowPolicy/<checkpoint_name>
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
$ python ./bin/Rollout.py ManiFlowPolicy MujocoUR5eCable --checkpoint ./checkpoint/ManiFlowPolicy/<checkpoint_name>/policy_last.ckpt
```

## Technical Details
For more information on the technical details, please see the following paper:
```bib
@inproceedings{ManiFlowPolicy_CoRL2025,
  author={Yan, Ge and Zhu, Jiyue and Deng, Yuquan and Yang, Shiqi and Qiu, Ri-Zhao and Cheng, Xuxin and Memmel, Marius and Krishna, Ranjay and Goyal, Ankit and Wang, Xiaolong and Fox, Dieter},
  title={{ManiFlow}: A General Robot Manipulation Policy via Consistency Flow Training},
  booktitle={Conference on Robot Learning},
  year={2025}
}
```
