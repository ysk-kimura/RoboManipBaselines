# Training tips

### Change the input/output of policy
The `--state_keys`, `--action_keys`, and `--camera_names` options allow changing the policy input and output.

Examples of `--state_keys`:
```console
# Set measured joint position as observed state (default)
--state_keys measured_joint_pos

# Leave observed state empty
--state_keys # no arguments
```

Examples of `--action_keys`:
```console
# Set command joint position as action (default)
--action_keys command_joint_pos

# Set command joint position relative to previous step as action
--action_keys command_joint_pos_rel

# Set command end-effector pose as action
--action_keys command_eef_pose command_gripper_joint_pos

# Set command end-effector pose relative to previous step as action
--action_keys command_eef_pose_rel command_gripper_joint_pos
```
