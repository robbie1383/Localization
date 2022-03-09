import random
import numpy as np

class Robot:

    def __init__(self, wall, size):
        self.radius = int(size / 2)
        self.x, self.y = self.initPosition(wall)
        self.frontX = self.x + self.radius
        self.frontY = self.y
        self.v = 0
        self.w = 0
        self.theta = 0
        self.speed = 1

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
            result = [self.x, self.y, self.theta] + np.matmul([[delta_t * np.cos(self.theta), 0], [delta_t * np.sin(self.theta), 2], [0, delta_t]], [self.v, self.w])
            next_x, next_y, new_theta = result[0], result[1], result[2]

            # Transfer results from the ICC computation
            self.x = next_x
            self.y = next_y
            self.theta = new_theta
            self.frontX, self.frontY = self.rotate(self.theta, self.radius)

        return self.v, self.w, np.round(np.degrees(self.theta), 2)

    def rotate(self, angle, r):
        # Rotate the robot at a certain angle from the x-axis
        front_x = self.x + np.cos(angle) * r
        front_y = self.y + np.sin(angle) * r
        return front_x, front_y

"""
    def right_intersection(point, wall_point1, wall_point2, sensor_start, sensor_end):
        x, y = point
        right = (x - wall_point1[0]) * (x - wall_point2[0]) <= 0 and \
                (y - wall_point1[1]) * (y - wall_point2[1]) <= 0 and \
                (x - sensor_start[0]) * (sensor_end[0] - sensor_start[0]) >= 0 and \
                (y - sensor_start[1]) * (sensor_end[1] - sensor_start[1]) >= 0
        return right
        
    def distanceToSensors(self, outer_wall):
        dist = []
        walls = []
        angle = copy(self.theta)
        for i in range(12):
            min_dist_out_wall, wall_out = self.distance(outer_wall, angle)
            dist.append(min_dist_out_wall)
            walls.append(wall_out)
            angle += math.pi / 6
        return dist, walls

    def distance(self, wall, angle):
        dist = []
        select_wall = []
        for i in range(len(wall) - 1):
            # Out wall line
            point1 = wall[i]
            point2 = wall[i + 1]
            a1 = point2[1] - point1[1]
            b1 = point1[0] - point2[0]
            c1 = a1 * point1[0] + b1 * point1[1]

            # Sensor line
            point3 = [self.x, self.y]
            point4 = Vector2(self.rotate(angle, self.radius))
            a2 = point4[1] - point3[1]
            b2 = point3[0] - point4[0]

            c2 = a2 * point3[0] + b2 * point3[1]
            determinant = a1 * b2 - a2 * b1

            if abs(determinant) > 0.00000001:  # if there is an intersection
                new_x = -(b1 * c2 - b2 * c1) / determinant  # intersection coordinate
                new_y = (a1 * c2 - a2 * c1) / determinant
                if abs(round(new_x) - new_x) < 0.00000001:
                    new_x = round(new_x)
                if abs(round(new_y) - new_y) < 0.00000001:
                    new_y = round(new_y)
                # make sure intersection is in front of the sensor
                if right_intersection((new_x, new_y), point1, point2, point3, point4):
                    dist.append(math.sqrt((new_x - point4[0]) ** 2 + (new_y - point4[1]) ** 2))
                    select_wall.append([point1, point2])
        wall = []
        if len(dist):
            min_dist_out_wall = min(dist)  # CLoser wall to sensor
            wall_index = dist.index(min_dist_out_wall)
            wall = select_wall[wall_index]
        else:
            min_dist_out_wall = 1500
        return min_dist_out_wall, wall
"""