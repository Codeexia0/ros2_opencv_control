# ArUco Marker Robot Control with ROS2

This project demonstrates a ROS2-based robot control system using a TurtleBot simulated in Gazebo. The robot's movement is determined by the position of an ArUco marker detected in the camera feed. The system integrates a USB camera driver, custom nodes for marker detection, and logic to control the robot's movement.

## Features

- **Camera Integration**: Uses the `usb_cam` package to capture video from a built-in or external camera.
- **ArUco Marker Detection**: Detects markers in the video feed and determines their position relative to the camera's field of view.
- **Robot Control**: Moves the robot:
  - **Forward**: When the marker is in the top half of the image.
  - **Backward**: When the marker is in the bottom half of the image.
- **Gazebo Simulation**: Simulates the robot and environment for testing.

## Requirements

- ROS2 (Humble or later recommended)
- Gazebo
- `usb_cam` ROS2 package
- OpenCV (for ArUco marker detection)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Codeexia0/ros2_opencv_control
   cd ros2_opencv_control
   ```

2. Build the ROS2 workspace:
   ```bash
   source /opt/ros/humble/setup.bash
   colcon build
   source install/setup.bash
   ```

3. Install dependencies:
   ```bash
   sudo apt install ros-<ros2-distro>-usb-cam
   sudo apt install ros-<ros2-distro>-gazebo-ros
   sudo apt install python3-opencv
   ```

## Usage

1. **Launch the Camera Driver**
   Start the camera driver to capture the video feed:
   ```bash
   ros2 run usb_cam usb_cam_node_exe
   ```

2. **Run the Custom Node**
   Run the node to process the video feed and detect ArUco markers:
   ```bash
   ros2 run camera_subscriber camera_node
   ```

3. **Launch the Robot in Gazebo**
   Start the Gazebo simulation environment:
   ```bash
   export TURTLEBOT3_MODEL=burger
   ros2 launch turtlebot3_gazebo empty_world.launch.py
   ```

## Demonstration

[![Watch the video](https://img.youtube.com/vi/AUqGh9ucM-A/0.jpg)](https://www.youtube.com/watch?v=AUqGh9ucM-A)

Click the image above to watch the demonstration video: **"ArUco Marker Robot Control with ROS2"**.

## Authors
- [Emin Shirinov](https://www.github.com/Codeexia0)
- [George Punnoose](https://www.github.com/George-P-1)
