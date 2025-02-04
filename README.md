# AV Project 11

- The packages and libraries required for running the code are present in the 'requirements.txt' file run them all using:

pip install -r requirements.txt

NOTE : you need to run this command in the directory which contains the 'requirements.txt' file.

- To Run the code use the following commands:

(1) Run a humble shell using the following commands:

humble_shell

source ~/.rosinit

(2) Move to the ROS2 workspace:

cd av/ros_ws

(3) Build the lane_detect package:

colcon build --symlink-install --packages-select lane_detect

source install/setup.bash

(4) Play the nuScenes ROS bag (check with 'ros2 topic list' if required):

ros2 bag  play ~/av/data/nuScenes/655 --loop --rate=0.1

(5) Run the lane_detect package:

ros2 run lane_detect laneDetect /cam_front/raw --nms 0.4
"# Lane_change_detection_system_autonomous_vehicles" 
