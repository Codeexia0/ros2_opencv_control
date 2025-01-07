#!/usr/bin/env python3

import rclpy  # ROS 2 Python Client Library
from rclpy.node import Node  # Base class for ROS nodes
from sensor_msgs.msg import Image  # Image message type
from geometry_msgs.msg import Twist  # Twist message for robot control
from cv_bridge import CvBridge  # Convert between ROS Image and OpenCV
import cv2  # OpenCV library
import cv2.aruco as aruco  # ArUco marker detection


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        # Initialize the CvBridge
        self.bridge = CvBridge()

        # Subscription to the image_raw topic
        self.subscription = self.create_subscription(
            Image,
            'image_raw',
            self.listener_callback,
            10
        )

        # Publisher for robot commands
        self.cmd_publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        self.get_logger().info("Camera node started and subscribing to 'image_raw'.")

        # ArUco dictionary and parameters
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        self.parameters = aruco.DetectorParameters()

    def create_twist_command(self, linear_x, angular_z):
        """
        Helper function to create a Twist message.
        """
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.angular.z = angular_z
        return cmd

    def listener_callback(self, image_data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(image_data, "bgr8")

            # Get the image dimensions (height, width)
            height, width, _ = cv_image.shape

            # Convert the image to grayscale for marker detection
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            # Detect ArUco markers
            corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            # If markers are detected
            if ids is not None:
                # Draw detected markers on the image
                aruco.drawDetectedMarkers(cv_image, corners, ids)
                self.get_logger().info(f"Detected marker IDs: {ids.flatten().tolist()}")

                for marker_id, corner in zip(ids.flatten(), corners):
                    # Get the center of the marker by averaging the corner coordinates
                    marker_center = corner[0].mean(axis=0)

                    # Calculate the vertical position of the marker's center (y-coordinate)
                    marker_center_y = marker_center[1]
                    # Calculate the horizontal position of the marker's center (x-coordinate)
                    marker_center_x = marker_center[0]

                    # Calculate the vertical and horizontal centers of the image
                    image_center_y = height / 2
                    image_center_x = width / 2

                    # Move the robot based on the marker's position
                    linear_velocity = 0.0
                    angular_velocity = 0.0

                    # Vertical movement (forward or backward)
                    if marker_center_y < image_center_y - 10:  # Marker is above the center
                        linear_velocity = 0.5  # Move forward
                        self.get_logger().info(f"Marker at {marker_id} is above center. Moving forward.")
                    elif marker_center_y > image_center_y + 10:  # Marker is below the center
                        linear_velocity = -0.5  # Move backward
                        self.get_logger().info(f"Marker at {marker_id} is below center. Moving backward.")
                    else:
                        self.get_logger().info(f"Marker at {marker_id} is near vertical center. No vertical movement.")

                    # Horizontal movement (rotation)
                    if marker_center_x < image_center_x - 10:  # Marker is to the left of center
                        angular_velocity = 0.5  # Rotate clockwise
                        self.get_logger().info(f"Marker at {marker_id} is left of center. Rotating clockwise.")
                    elif marker_center_x > image_center_x + 10:  # Marker is to the right of center
                        angular_velocity = -0.5  # Rotate counterclockwise
                        self.get_logger().info(f"Marker at {marker_id} is right of center. Rotating counterclockwise.")
                    else:
                        self.get_logger().info(f"Marker at {marker_id} is near horizontal center. No rotation.")

                    # Publish the twist command with both linear and angular velocities
                    command = self.create_twist_command(linear_velocity, angular_velocity)
                    self.cmd_publisher.publish(command)

            # Display the image with detected markers
            cv2.imshow("ArUco Marker Detection", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error in listener_callback: {e}")


def main(args=None):
    rclpy.init(args=args)

    # Create and run the CameraNode
    camera_node = CameraNode()

    try:
        # Spin the node
        rclpy.spin(camera_node)
    except KeyboardInterrupt:
        pass
    finally:
        # Shutdown the node
        camera_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
