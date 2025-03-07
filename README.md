<!--
 * @Author: Wenhao Ding
 * @Email: wenhaod@andrew.cmu.edu
 * @Date: 2021-07-18 21:43:53
 * @LastEditTime: 2021-08-25 19:45:48
 * @Description: 
-->
# SafeBench: A Benchmarking Platform for Safety Evaluation of Autonomous Vehicles

This is an evaluation platform based on Carla Simulator. The structure of this platform is shown in this figure:
![pipeline](./misc/pipeline_2.png)

## Instructions

Firstly, make sure you already install the NVIDIA driver on your mechine. All environment settings are store in a docker image, please follow the instructions below to install all things.

### Step 1. Download docker image

1. Install Docker by following [this link](https://docs.docker.com/engine/install/ubuntu/).
2. Install NVIDIA-Docker2 by following [this link](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker).
3. Pull the [Docker image](https://hub.docker.com/r/garyxcj/carla): `docker pull garyxcj/carla`

### Step 2. Download source code

Download the source code from this repo: 
`git clone https://github.com/trust-ai/SafeBench.git`

### Step 3. Run the Docker container

The command of running the container is in `run_docker.sh`, you just need to run this script. After running it, a window of the Carla Simulator will show up.

### Step 4. Access container bash and launch the entire platform

1. Open a new terminal window and run the script `run_bash.sh` to access the bash of container.
2. Change directory: `cd Evaluation`
3. Complile all files with ROS tools: `catkin_make`
4. Set up environment: `. ./devel/setup.bash`
5. launch platform: `roslaunch manager manager.launch`

Finally, you should be able to see that the Carla window changes the map and spawns an ego vehicle. Another window of pygame will also show up for controlling the ego vehicle.

## File Structure

```
pkgs
 └── scenario_runner
       ├── srunner
       └── scenario_runner.py
src
 ├── agent
 │     └── gym_node
 │
 ├── evaluation
 │     ├── scenario_node
 │     └── scenario_runner_server
 │
 ├── manager
 ├── run.sh
 ├── requirement.txt
 └── CMakeLists.txt
```


## Reference

This repo is modified from [GRAIC](https://github.com/PoPGRI/Race).
