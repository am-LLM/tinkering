"""# Field 424: 15-State Error-State Kalman Filter (ESKF) for Visual-Inertial Odometry

Implements high-speed 6-DOF IMU quaternion integration coupled with downward optical flow
velocity updates. Tracks 15 error states (position, velocity, orientation on SO(3), accelerometer bias,
and gyroscope bias) providing drift-resistant dead-reckoning hover in 100% GPS-denied/jammed environments.
"""
