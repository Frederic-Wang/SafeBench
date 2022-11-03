#!/usr/bin/env python

# Copyright (c) 2022: Shuai Wang (shuaiwa2@andrew.cmu.edu)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

from __future__ import division

import os
import cv2
import copy
import numpy as np
import pygame
import time
import shutil
from skimage.transform import resize

import rospy
from ros_compatibility import get_service_response
from carla_ros_scenario_runner_types.msg import CarlaScenarioRunnerStatus
from carla_ros_scenario_runner_types.srv import GetEgoVehicleRoute
from carla_ros_scenario_runner_types.srv import UpdateRenderMap
from carla_ros_scenario_runner_types.msg import CarlaScenarioStatus
import carla_common.transforms as trans


import gym
from gym import spaces
from gym.utils import seeding
import carla

from gym_carla.render import BirdeyeRender
from gym_carla.route_planner import RoutePlanner
from gym_carla.misc import *

class CarlaEnv(gym.Env):
    def __init__(self, params):
        # parameters
        self.dt = params['dt']
        self.max_past_step = params['max_past_step']
        self.max_time_episode = params['max_time_episode']
        self.max_waypt = params['max_waypt']

        self.display_size = params['display_size']  # rendering screen size
        self.obs_range = params['obs_range']
        self.d_behind = params['d_behind']

        # Connect to carla server and get world object, Don't use load_world()
        rospy.loginfo('Connecting to Carla server...')
        client = carla.Client('localhost', params['port'])
        client.set_timeout(10.0)
        self.world = client.get_world()
        rospy.loginfo('Carla server connected!')

        # collision sensor
        self.collision_hist = []   # collision history
        self.collision_hist_l = 1  # collision history length

        self.collision_sensor = None
        self.lidar_sensor = None
        self.camera_sensor = None

        self.terminate = False
        self.time_step = 0
        self._tick = 0

        self.render = True

        # for multiple egos
        self.ego_routeplanner_dict = {}

        if self.render:
            self._init_renderer()
            update_render_map_service = rospy.Service('/gym_node/update_render_map', UpdateRenderMap,
                                                      self._update_render)
            rospy.loginfo('Finish initializing renderer.')



    def initialize(self):
        if self.collision_sensor is not None:
            self._stop_sensor()

        self.settings = self.world.get_settings()
        self.settings.fixed_delta_seconds = self.dt
        self._set_synchronous_mode(False)

        # get actors polygon list
        self.vehicle_polygons = []
        vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
        self.vehicle_polygons.append(vehicle_poly_dict)
        self.walker_polygons = []
        walker_poly_dict = self._get_actor_polygons('walker.*')
        self.walker_polygons.append(walker_poly_dict)

        # get actors info list
        self.vehicle_trajectories = []
        self.vehicle_accelerations = []
        self.vehicle_angular_velocities = []
        self.vehicle_velocities = []
        vehicle_info_dict_list = self._get_actor_info('vehicle.*')
        self.vehicle_trajectories.append(vehicle_info_dict_list[0])
        self.vehicle_accelerations.append(vehicle_info_dict_list[1])
        self.vehicle_angular_velocities.append(vehicle_info_dict_list[2])
        self.vehicle_velocities.append(vehicle_info_dict_list[3])

        # get ego vehicles
        # comment: Fred, here is the difference
        vehicles = self.world.get_actors().filter('vehicle.*')
        ego_rolename_set = set()
        self.ego_vehicles = []
        for vehicle in vehicles:
            # if vehicle.attributes['role_name'] == self.role_name:
            #     self.ego = vehicle
            cur_role_name = vehicle.attributes['role_name']
            if cur_role_name.startswith('ego_vehicle') and cur_role_name not in ego_rolename_set:
                self.ego_vehicles.append(vehicle)
                ego_rolename_set.add(cur_role_name)

        # find collision sensor
        # TODO: now get sensor_list[0], might need to get multiple sensors on multiple egos
        self.collision_sensor_list = self.world.get_actors().filter('sensor.other.collision')
        if len(self.collision_sensor_list) == 0:
            raise RuntimeError('No collision sensor in the simulator')
        else:
            # if len(self.collision_sensor) > 1:
            self.collision_sensor = self.collision_sensor_list[0]
            # self._stop_sensors_list(self.collision_sensor_list)

            def get_collision_hist(event):
                impulse = event.normal_impulse
                intensity = np.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
                self.collision_hist.append(intensity)
                if len(self.collision_hist) > self.collision_hist_l:
                    self.collision_hist.pop(0)

            self.collision_hist = []
            self.collision_sensor.listen(lambda event: get_collision_hist(event))

        # find camera sensor
        # TODO: now get camera_list[0], might need to get multiple cameras on multiple egos
        self.camera_sensor_list = self.world.get_actors().filter('sensor.camera.rgb')
        if len(self.camera_sensor_list) == 0:
            raise RuntimeError('No camera sensor in the simulator')
        else:
            self.camera_sensor = self.camera_sensor_list[0]

            # self._stop_sensors_list(self.camera_sensor_list)
            def get_camera_img(data):
                array = np.frombuffer(data.raw_data, dtype=np.dtype("uint8"))
                array = np.reshape(array, (data.height, data.width, 4))
                array = array[:, :, :3]
                array = array[:, :, ::-1]
                self.camera_img = array

            self.camera_sensor.listen(lambda data: get_camera_img(data))

        # find lidar
        # TODO: now get lidar_list[0], might need to get multiple cameras on multiple egos
        self.lidar_height = 2.1
        self.lidar_sensor_list = self.world.get_actors().filter('sensor.lidar.ray_cast')
        if len(self.lidar_sensor_list) == 0:
            raise RuntimeError('No LiDAR sensor in the simulator')
        else:
            self.lidar_sensor = self.lidar_sensor_list[0]

            # self._stop_sensors_list(self.lidar_sensor_list)
            def get_lidar_data(data):
                self.lidar_data = data

            self.lidar_sensor.listen(lambda data: get_lidar_data(data))

        # Enable sync mode
        self.settings.synchronous_mode = True
        self.world.apply_settings(self.settings)

        map = self.world.get_map()

        # comment: here is what need to change a lot
        for cur_ego in self.ego_vehicles:
            response = None
            rospy.wait_for_service('/carla_data_provider/get_ego_vehicle_route')
            try:
                requester = rospy.ServiceProxy('/carla_data_provider/get_ego_vehicle_route', GetEgoVehicleRoute)
                response = requester(cur_ego.id)
            except rospy.ServiceException as e:
                rospy.loginfo('Run scenario service call failed: {}'.format(e))

            init_waypoints = []
            if response is not None:
                for pose in response.route_info.ego_vehicle_route.poses:
                    carla_transform = trans.ros_pose_to_carla_transform(pose.pose)
                    current_waypoint = map.get_waypoint(carla_transform.location)
                    init_waypoints.append(current_waypoint)

            # for each ego, it has a route planner
            # TODO: self.waypoints? self.target_road_option? ...
            routeplanner = RoutePlanner(cur_ego, self.max_waypt, init_waypoints)
            self.ego_routeplanner_dict[cur_ego] = routeplanner

        return self._get_obs()

    """
    actions and scenario_status should be lists
    """
    def step(self, actions, scenario_status):
        done = False
        info_list = []
        for i in range(actions):
            action = actions[i]
            if len(action) == 2:
                if self.discrete:
                    acc = self.discrete_act[0][action // self.n_steer]
                    steer = self.discrete_act[1][action % self.n_steer]
                else:
                    acc = action[0]
                    steer = action[1]

                # Convert acceleration to throttle and brake
                if acc > 0:
                    throttle = np.clip(acc / 3, 0, 1)
                    brake = 0
                else:
                    throttle = 0
                    brake = np.clip(-acc / 8, 0, 1)

            # When action contains steer, throttle and brake
            elif len(action) == 3:
                steer = np.clip(action[0], -1, 1)
                throttle = np.clip(action[1], 0, 1)
                brake = np.clip(action[2], 0, 1)

            else:
                raise RuntimeError('Wrong number of actions.')

            cur_ego = self.ego_vehicles[i]

            # Apply control
            act = carla.VehicleControl(throttle=float(throttle),
                                       steer=float(-steer),
                                       brake=float(brake))
            cur_ego.apply_control(act)
            self.world.tick()
            self._tick += 1

            # # Append actors polygon list
            # vehicle_poly_dict = self._get_actor_polygons('vehicle.*')
            # self.vehicle_polygons.append(vehicle_poly_dict)
            # while len(self.vehicle_polygons) > self.max_past_step:
            #     self.vehicle_polygons.pop(0)
            # walker_poly_dict = self._get_actor_polygons('walker.*')
            # self.walker_polygons.append(walker_poly_dict)
            # while len(self.walker_polygons) > self.max_past_step:
            #     self.walker_polygons.pop(0)

            # # Append actors info list
            # vehicle_info_dict_list = self._get_actor_info('vehicle.*')
            # self.vehicle_trajectories.append(vehicle_info_dict_list[0])
            # while len(self.vehicle_trajectories) > self.max_past_step:
            #     self.vehicle_trajectories.pop(0)
            # self.vehicle_accelerations.append(vehicle_info_dict_list[1])
            # while len(self.vehicle_accelerations) > self.max_past_step:
            #     self.vehicle_accelerations.pop(0)
            # self.vehicle_angular_velocities.append(vehicle_info_dict_list[2])
            # while len(self.vehicle_angular_velocities) > self.max_past_step:
            #     self.vehicle_angular_velocities.pop(0)
            # self.vehicle_velocities.append(vehicle_info_dict_list[3])
            # while len(self.vehicle_velocities) > self.max_past_step:
            #     self.vehicle_velocities.pop(0)

            waypoints, target_road_option, current_waypoint, target_waypoint, _, vehicle_front, waypoint_location_list = self.ego_routeplanner_dict[cur_ego].run_step()
            cur_info = {
                'waypoints': waypoints,
                'vehicle_front': vehicle_front,
                'cost': self._get_cost()
            }

            info_list.append(cur_info)

            terminal = False

            if terminal or scenario_status in [CarlaScenarioStatus.STARTING, CarlaScenarioStatus.STOPPED,
                                               CarlaScenarioStatus.SHUTTINGDOWN, CarlaScenarioStatus.ERROR]:
                done = True
                if not terminal:
                    rospy.loginfo('Terminate due scenario runner')
                # images = sorted(os.listdir(self.record_images_dir))
                # writer = cv2.VideoWriter(os.path.join(self.record_dir,
                #                                       'scenario_{}_route_{}_{}_{}.mp4'.format(self.scenario_id,
                #                                                                               self.route_id,
                #                                                                               self.method,
                #                                                                               self.data_id)),
                #                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                #                          (self.display_size * 3, self.display_size))
                # for img_name in images:
                #     img_path = os.path.join(self.record_images_dir, img_name)
                #     img = cv2.imread(img_path)
                #     writer.write(img)
                # writer.release()
                # shutil.rmtree(self.record_images_dir)
            elif scenario_status in [CarlaScenarioStatus.RUNNING]:
                done = False
            else:
                raise Exception('Unknown running status')

        self.time_step += 1
        self._timestamp = self.world.get_snapshot().timestamp.elapsed_seconds

        return self._get_obs(), self._get_reward(), done, copy.deepcopy(info_list)

    def _get_actor_polygons(self, filt):
        pass

    def _get_actor_info(self, filt):
        pass

    """
    this should return a list of all obs for all egos
    """
    def _get_obs(self):
        """get the observations"""
        # TODO: implement this function
        obs = []
        for cur_ego in self.ego_vehicles:
            ob = {
                'tick': int(self._tick),
            }
            obs.append(ob)
        return obs

    def _get_reward(self):
        """
        calculate the step reward
        should return a list
        """
        #TODO: implement this function
        r_list = []

        for cur_ego in self.ego_vehicles:
            r_list.append(1)

        return r_list

    def _init_renderer(self):
        """Initialize the birdeye view renderer.
        """
        pygame.init()
        self.display = pygame.display.set_mode(
            (self.display_size * 3, self.display_size),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        pixels_per_meter = self.display_size / self.obs_range
        pixels_ahead_vehicle = (self.obs_range / 2 -
                                self.d_behind) * pixels_per_meter
        birdeye_params = {
            'screen_size': [self.display_size, self.display_size],
            'pixels_per_meter': pixels_per_meter,
            'pixels_ahead_vehicle': pixels_ahead_vehicle
        }
        self.birdeye_render = BirdeyeRender(self.world, birdeye_params)

    def _update_render(self, req, response=None):
        rospy.loginfo("New map requested...")
        result = True
        try:
            self.birdeye_render = BirdeyeRender(self.world, self.birdeye_render.params)
        except:
            result = False
        if req.town != self.birdeye_render.town_map.name:
            result = False
        response = get_service_response(UpdateRenderMap)
        response.result = result
        if result:
            rospy.loginfo("Map updated")
        return response

    def _get_cost(self):
        return 0

    def _terminal(self):
        pass

    def _stop_sensor(self):
        pass

    def _stop_sensors_list(self, sensors):
        pass

    def _clear_all_actors(self, actor_filters):
        pass