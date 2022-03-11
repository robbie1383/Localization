import math
import random
import numpy as np


class Robot:

    def __init__(self, wall, size):
        self.radius = int(size / 2)
        self.x, self.y = self.initPosition(wall[0])
        self.frontX = self.x + self.radius
        self.frontY = self.y
        self.v = 0
        self.w = 0
        self.theta = 0
        self.speed = 1
        self.sensor_limit = 200
        self.real_track = []

    def initPosition(self, wall):
        x = random.randint(wall[0][0] + self.radius, wall[2][0] - self.radius)
        y = random.randint(wall[0][1] + self.radius, wall[2][1] - self.radius)
        return x, y

    def move(self, movement, delta_t):
        # Check keys for movement
        # movement = [w, s, a, d, x]
        if movement[0] == 1:
            self.v += self.speed
        if movement[1] == 1:
            self.v -= self.speed
        if movement[2] == 1:
            self.w += 0.01
        if movement[3] == 1:
            self.w -= 0.01
        if movement[4] == 1:
            self.v = 0
            self.w = 0

        # If it's moving
        if self.v != 0 or self.w != 0:
            result = [self.x, self.y, self.theta] + np.matmul(
                [[delta_t * np.cos(self.theta), 0], [delta_t * np.sin(self.theta), 2], [0, delta_t]], [self.v, self.w])
            next_x, next_y, new_theta = result[0], result[1], result[2]
            # Transfer results from the ICC computation
            self.x = next_x
            self.y = next_y
            self.theta = new_theta
            self.frontX, self.frontY = self.rotate(self.theta, self.radius)
        self.real_track.append((self.x, self.y))
        return self.v, self.w, np.round(np.degrees(self.theta), 2)

    def rotate(self, angle, r):
        # Rotate the robot at a certain angle from the x-axis
        front_x = self.x + np.cos(angle) * r
        front_y = self.y + np.sin(angle) * r
        return front_x, front_y

    def get_close_landmarks(self, landmarks):
        sensors = []
        close_landmarks = []
        for landmark in landmarks:
            x, y = landmark
            sensors.append((math.sqrt((self.x - x) ** 2 + (self.y - y) ** 2)))
        sensors_copy = sensors.copy()
        sensors_copy.sort()
        for i in range(3):
            if sensors_copy[i] < self.sensor_limit:
                close_landmarks.append(landmarks[sensors.index(sensors_copy[i])])
        return close_landmarks
