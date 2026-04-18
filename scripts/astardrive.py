#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

import json
import math
import time
import os


TIME_PER_ACTION_S = 1.0   # must match astar_phase2.py TIME_PER_ACTION_S
CM_TO_M           = 0.01  # convert cm → meters for ROS2 (ROS uses meters)

class AStarDriver(Node):
    def __init__(self):
        super().__init__("astar_driver")
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        path_data = None
        # load the json format path
        for path in ['path_output.json',
            os.path.join(os.path.dirname(__file__), 'path_output.json')]:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    path_data = json.load(f)        
        
        if path_data is None:
            self.get_logger().error('path_output.json not found.')
            return
        
        self.rpm1 = path_data["rpm1"]
        self.rpm2 = path_data["rpm2"]
        # convert wheel constants cm → meter for velocity calculation
        self.wheel_radius = path_data['wheel_radius_cm'] * CM_TO_M
        self.wheel_dist   = path_data['wheel_dist_cm']   * CM_TO_M
        # each waypoint is where rob is after 1-second action
        self._waypoints    = path_data['path']   # list of {x, y, theta_deg} in cm
        
        self.get_logger().info(f"num waypoints = {len(self._waypoints)}")
        self.get_logger().info(f"rpm1={self.rpm1}  rpm2={self.rpm2}")
        self.get_logger().info(f'Wheel r={self.wheel_radius:.4f}m  Wheel dist={self.wheel_dist:.4f}m')
        
        #Drive the turtlebot
        self.drive_path()

    def rpm_to_rads(self, rpm):
        return rpm * 2 * 3.14 / 60
    
    def rpms_to_twist(self, rpm_left, rpm_right):
        ul = self.rpm_to_rads(rpm_left)
        ur = self.rpm_to_rads(rpm_right)
        r  = self.wheel_radius
        L  = self.wheel_dist
        linear  = (r/2) * (ul + ur)
        angular = (r/L) * (ur - ul)
        return linear, angular
    
    def publish_cmd_velocity(self, linear_vel, angular_vel):
        msg = Twist()
        msg.linear.x = linear_vel #m/s going forward
        msg.angular.z = angular_vel #rad/s turning
        self.cmd_vel_pub.publish(msg)
        
    def stop_robot(self):
        self.publish_cmd_velocity(0.0, 0.0) #publish 0 to stop the robot
        self.get_logger().info("Robot is stop")
        
    def get_action_space(self):
        return [
            (0, self.rpm1),  #left_wheel_rpm, right_wheel_rpm
            (self.rpm1, 0),    
            (self.rpm1, self.rpm1), 
            (0, self.rpm2), 
            (self.rpm2, 0),    
            (self.rpm2, self.rpm2), 
            (self.rpm1, self.rpm2), 
            (self.rpm2, self.rpm1), 
        ]
        
    def find_best_action(self,cx, cy, ct_deg, nx, ny):
        """
        find which of 8 RPM action will move rob from wp1 to wp2
        """
            
        r = self.wheel_radius #m 
        L = self.wheel_dist #m     
        dt = 0.1 # seconds per integration step
        steps = int(TIME_PER_ACTION_S / dt)
        
        cx_m = cx * CM_TO_M #convert from cm to m for sim
        cy_m = cy * CM_TO_M
        nx_m = nx * CM_TO_M
        ny_m = ny * CM_TO_M
        
        best_action = None
        best_dist = float('inf')
        
        for act in self.get_action_space(): 
            rpm_left, rpm_right = act[0], act[1]
            ul = self.rpm_to_rads(rpm_left)
            ur = self.rpm_to_rads(rpm_right)
            theta = math.radians(ct_deg)
            sim_x = cx_m
            sim_y = cy_m
            
            for _ in range(steps): # simulate this action for time_per_action_s seconds
                dx = (r/2) * (ul + ur) * math.cos(theta) * dt
                dy = (r/2) * (ul + ur) * math.sin(theta) * dt
                dtheta = (r/L) * (ur - ul) * dt
        
                sim_x += dx
                sim_y += dy
                theta += dtheta
            
            #dist from simulated end to actual next waypoint
            dist = math.sqrt((sim_x - nx_m)**2 + (sim_y - ny_m)**2)

            if dist < best_dist:
                best_dist = dist
                best_action = (rpm_left, rpm_right)
        
        return best_action


    def drive_path(self):
        """
        Drive through each waypoint in the path.
        For each step: find best RPM action → convert to velocity → publish.
        """

        self.get_logger().info('Starting to drive. Waiting 2s for Gazebo...')
        time.sleep(2.0)
        
        for i in range(len(self._waypoints) - 1):
            curr_waypoint = self._waypoints[i]
            next_waypoint = self._waypoints[i + 1]
            
            cx, cy, ct = curr_waypoint['x'], curr_waypoint['y'], curr_waypoint['theta_deg']
            nx, ny = next_waypoint['x'], next_waypoint['y']
            
            self.get_logger().info(
                f'Step {i+1}/{len(self._waypoints) - 1} | '
                f'({cx:.1f},{cy:.1f},{ct:.0f}deg) → ({nx:.1f},{ny:.1f})'
            )
            
            #find which rpm action match this movement result
            rpm_left, rpm_right = self.find_best_action(cx, cy, ct, nx, ny)
            #convert rpm to linear, angular vel for ROS
            linear_vel, angular_vel = self.rpms_to_twist(rpm_left, rpm_right)

            self.get_logger().info(
                f'  RPMs=({rpm_left},{rpm_right}) | '
                f'linear={linear_vel:.3f}m/s  angular={angular_vel:.3f}rad/s'
            )
            
            # publish velocity for 1 second (open-loop)
            self.publish_cmd_velocity(linear_vel, angular_vel)
            # wait for 1 second before move to next waypoint
            time.sleep(TIME_PER_ACTION_S)
        
        self.stop_robot()
        self.get_logger().info('Path complete!')


def main(args=None):
    rclpy.init(args=args)
    node = AStarDriver()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
