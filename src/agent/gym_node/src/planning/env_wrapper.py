import traceback
from math import cos
import gym
import time
import rospy
import numpy as np
from easydict import EasyDict as edict
from carla_ros_scenario_runner_types.msg import CarlaScenarioStatus


CFG = edict(
    ACC_MAX=3,
    STEERING_MAX=0.3,
    OBS_TYPE=0,
    MAX_EPISODE_LEN=300,
    FRAME_SKIP=4,
)

class EnvWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super().__init__(env)
        self.cfg = cfg
        self._env = env
        self.spec.id = "GymCarla"
        self.spec.max_episode_steps = cfg.MAX_EPISODE_LEN
        env._max_episode_steps = cfg.MAX_EPISODE_LEN
        self._max_episode_steps = env._max_episode_steps
        self.frame_skip = cfg.FRAME_SKIP

        self._build_obs_space(cfg.OBS_TYPE)

        # build action space, assume the obs range from -1 to 1
        act_dim = 2
        act_lim = np.ones((act_dim), dtype=np.float32)
        self.action_space = gym.spaces.Box(-act_lim, act_lim, dtype=np.float32)

        self.scenario_status = CarlaScenarioStatus.STOPPED
        self.status_subscriber = rospy.Subscriber("/scenario/status",
                                    CarlaScenarioStatus,
                                    self.scenario_runner_status_callback)

    def scenario_runner_status_callback(self, status):
        self.scenario_status = status.status

    def wait_for_reset(self):
        while self.scenario_status != CarlaScenarioStatus.RUNNING:
            time.sleep(1)
        rospy.loginfo('gym_node: reset done')
        obs = self._env.initialize()
        return self._preprocess_obs(obs)

    def wait_for_terminate(self, wait=True):
        # while self.scenario_status not in [CarlaScenarioStatus.STOPPED, CarlaScenarioStatus.ERROR]:
        if wait:
            rospy.loginfo('gym_node: waiting to terminate')
            while self.scenario_status == CarlaScenarioStatus.RUNNING:
                time.sleep(1)
        rospy.loginfo('gym_node: terminate')

    def step(self, actions):
        """

        Args:
            actions: a list of actions

        Returns: lists of o, reward, done, info

        """
        actions = self._postprocess_action(actions)

        reward = [0 for i in range(len(actions))]
        cost = [0 for i in range(len(actions))]
        done = False

        o = None
        info = None
        for i in range(self.frame_skip):
            # here o, r, info are all lists
            # d should not be list
            o, r, d, info = self._env.step(actions, self.scenario_status)
            # what about this d?
            if d:
                done = True
            r, info = self._preprocess_reward(r, info)
            o = self._preprocess_obs(o)

            for i in range(len(reward)):
                reward[i] += r[i]

            for i in range(len(info)):
                if "cost" in info[i]:
                    cost[i] += info[i]["cost"]
            if done:
                break
        # if "cost" in info:
        #     info["cost"] = cost
        if done:
            self.wait_for_terminate(wait=False)
        return o, reward, done, info

    def _build_obs_space(self, obs_type):
        self.obs_type = obs_type
        if self.obs_type == 0:
            # 4 state space
            obs_dim = 4
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim,
                                                    obs_lim,
                                                    dtype=np.dtype)
        elif self.obs_type == 1:
            # 11 state space
            obs_dim = 11
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim,
                                                    obs_lim,
                                                    dtype=np.dtype)
        elif self.obs_type == 2 or self.obs_type == 3:
            # 4 state space + bev
            obs_dim = 256
            # assume the obs range from -1 to 1
            obs_lim = np.ones((obs_dim), dtype=np.float32)
            self.observation_space = gym.spaces.Box(-obs_lim,
                                                    obs_lim,
                                                    dtype=np.dtype)
        else:
            raise NotImplementedError

    def _preprocess_obs(self, obs):
        self.obs_type = 1
        """

        Args:
            obs: a list of obs

        Returns: a list of processed obs

        """
        # print("====================")
        # print("obs in env_wrapper")
        # print(obs)
        processed_obs = []
        for ob in obs:
            # TODO: need to change
            if self.obs_type == 0:
                processed_obs.append(ob['state'][:4].astype(np.float64))
            elif self.obs_type == 1:
                new_ob = np.array([0, 0, 0, 0,
                                    0, 0, 0,
                                    0, 0,
                                    0, 0])
                processed_obs.append(new_ob)
            elif self.obs_type == 2:
                processed_obs.append({"img": ob['birdeye'], "states": ob['state'][:4].astype(np.float64)})
            elif self.obs_type == 3:
                processed_obs.append({"img": ob['camera'], "states": ob['state'][:4].astype(np.float64)})
            else:
                raise NotImplementedError
            # processed_obs.append(ob)

        return processed_obs

    def _preprocess_reward(self, reward, info):
        return reward, info

    def _postprocess_action(self, actions):
        # normalize and clip the action
        processed_actions = []
        for action in actions:
            action = action * np.array([self.cfg.ACC_MAX, self.cfg.STEERING_MAX])
            action[0] = max(min(self.cfg.ACC_MAX, action[0]), -self.cfg.ACC_MAX)
            action[1] = max(min(self.cfg.STEERING_MAX, action[1]),
                            -self.cfg.STEERING_MAX)
            processed_actions.append(action)
        return processed_actions

params = {
    'number_of_vehicles': 100,
    'number_of_walkers': 0,
    'display_size': 256,  # screen size of bird-eye render
    'max_past_step': 1,  # the number of past steps to draw
    'dt': 0.1,  # time interval between two frames
    'discrete': False,  # whether to use discrete control space
    'discrete_acc': [-3.0, 0.0, 3.0],  # discrete value of accelerations
    'discrete_steer': [-0.2, 0.0, 0.2],  # discrete value of steering angles
    'continuous_accel_range': [-3.0, 3.0],  # continuous acceleration range
    'continuous_steer_range': [-0.3, 0.3],  # continuous steering angle range
    'ego_vehicle_filter':
    'vehicle.lincoln*',  # filter for defining ego vehicle
    'port': 2000,  # connection port
    'town': 'Town03',  # which town to simulate
    'task_mode':
    'random',  # mode of the task, [random, roundabout (only for Town03)]
    'max_time_episode': 1000,  # maximum timesteps per episode
    'max_waypt': 12,  # maximum number of waypoints
    'obs_range': 32,  # observation range (meter)
    'lidar_bin': 0.125,  # bin size of lidar sensor (meter)
    'd_behind': 12,  # distance behind the ego vehicle (meter)
    'out_lane_thres': 4,  # threshold for out of lane (meter)
    'desired_speed': 9,  # desired speed (m/s)
    'max_ego_spawn_times': 200,  # maximum times to spawn ego vehicle
    'display_route': True,  # whether to render the desired route
    'pixor_size': 64,  # size of the pixor labels
    'pixor': False,  # whether to output PIXOR observation
}


def carla_env(env_args, obs_type):
    CFG.OBS_TYPE = obs_type
    params.update(env_args)
    return EnvWrapper(gym.make('carla-v0', params=params), cfg=CFG)