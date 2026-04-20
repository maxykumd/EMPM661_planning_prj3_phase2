#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

import json
import math
import time
import os


TIME_PER_ACTION_S = 1.0   
CM_TO_M           = 0.01  # convert cm to meter for ROS2

class AStarDriver(Node):
    def __init__(self):
        super().__init__("astar_driver")
        # create odom sub
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.robot_x = 0.0
        self.robot_y = 0.0
        
        # create velocity pub
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
        
        self.path_data = path_data

        
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

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
    def rpm_to_rads(self, rpm):
        return rpm * 2 * math.pi / 60
    
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
        self.get_logger().info('Starting to drive. Waiting 2s for Gazebo...')
        time.sleep(2.0)

        # read actions directly from JSON — no need to guess which action was used
        actions = self.path_data['actions']
        last_logged_x = self.robot_x   # ← track last logged position
        last_logged_y = self.robot_y

        for i, action in enumerate(actions):
            rpm_left  = action['rpm_left']
            rpm_right = action['rpm_right']

            linear_vel, angular_vel = self.rpms_to_twist(rpm_left, rpm_right)

            self.get_logger().info(
                f'Step {i+1}/{len(actions)} | '
                f'RPMs=({rpm_left},{rpm_right}) | '
                f'linear={linear_vel:.3f}m/s  angular={angular_vel:.3f}rad/s'
            )

            # CORRECT — spin while sleeping so odom updates
            self.publish_cmd_velocity(linear_vel, angular_vel)
            start = time.time()

            while time.time() - start < TIME_PER_ACTION_S:
                rclpy.spin_once(self, timeout_sec=0.1)

                # only log if moved more than 0.3m = 30cm from last log
                dist_moved = math.sqrt(
                    (self.robot_x - last_logged_x)**2 +
                    (self.robot_y - last_logged_y)**2
                )
                if dist_moved >= 0.3:
                    self.get_logger().info(
                        f'  Pos: ({self.robot_x:.2f}, {self.robot_y:.2f})'
                    )
                    last_logged_x = self.robot_x
                    last_logged_y = self.robot_y
        self.stop_robot()
        # log final position always
        self.get_logger().info(
            f'Final pos: ({self.robot_x:.2f}, {self.robot_y:.2f})'
        )
        self.get_logger().info('Path complete!')

def main(args=None):
    rclpy.init(args=args)
    node = AStarDriver()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
