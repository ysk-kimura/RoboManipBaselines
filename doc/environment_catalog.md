# Environment catalog

You can teleoperate a robot in each environment by giving `<env_name>` (e.g. `MujocoUR5eCable`) as an argument to `Teleop.py` as follows:
```console
# Go to the top directory of this repository
$ cd robo_manip_baselines
$ python ./bin/Teleop.py <env_name>
```

## MuJoCo environments
### UR5e
#### MujocoUR5eCable
Task to pass the cable between two poles.

https://github.com/user-attachments/assets/e01a411f-00eb-4579-b0e8-ab301f10e02b

#### MujocoUR5eRing
Task to pick a ring and put it around the pole.

https://github.com/user-attachments/assets/fb922a0f-e292-493a-9856-e73bbc119fc8

#### MujocoUR5eParticle
Task to scoop up particles.

https://github.com/user-attachments/assets/3dd17b4f-8781-4945-bf86-8c433c822b87

#### MujocoUR5eCloth
Task to roll up the cloth.

https://github.com/user-attachments/assets/4c78f4ff-8b77-4f2d-be7b-755a09227f5f

#### MujocoUR5eInsert
Task to insert a pin into a hole.

https://github.com/user-attachments/assets/5ca1eafe-f818-4efb-a012-6db92990ad14

### xArm7
#### MujocoXarm7Cable
Task to pass the cable between two poles.

https://github.com/user-attachments/assets/450d04e8-7154-4c01-849c-d1f9b50be0ae

#### MujocoXarm7Ring
Task to pick a ring and put it around the pole.

https://github.com/user-attachments/assets/da83db94-7fcb-4690-9b73-41af0c1394a8

### ALOHA
#### MujocoAlohaCable
Task to pass the cable between two poles.

https://github.com/user-attachments/assets/3bbeeeb8-7034-428c-a17e-f3968a2890b9

## Isaac environments
### UR5e
#### IsaacUR5eChain
Task to pick a chain and hang it on a hook

https://github.com/user-attachments/assets/803759a5-be70-4530-b9ba-b5d381b1bb7a

#### IsaacUR5eCabinet
Task to open the top lid of the box

https://github.com/user-attachments/assets/b5f0cdc3-f9f6-4b4d-9d03-089b84d44f34

## Real-world environments
### UR5e
#### RealUR5eDemo
Various manipulation tasks with UR5e in the real world

https://github.com/user-attachments/assets/1b894ea0-6b5a-4bf8-abaa-8cd3ec2c4f85

See [here](./real_ur5e.md) for instructions on how to operate real robot.

### xArm7
#### RealXarm7Demo
Various manipulation tasks with xArm7 in the real world

https://github.com/user-attachments/assets/ab0c9830-5b33-48e8-9dac-272460a51a39
