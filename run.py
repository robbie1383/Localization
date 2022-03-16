import numpy as np
import pygame

from Robot import Robot
from Localization import Localization

SEV = 35  # SCREEN_EDGE_VACANCY
WIDTH = 670
HEIGHT = 870
# A list of RGB values for the colours used in the game.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
DARKGRAY = (125, 130, 138)
PURPLE = (114, 85, 163)
LIGHTPURPLE = (185, 167, 217)

landmarks = [(SEV, SEV), (WIDTH - SEV, SEV),
             (SEV, SEV + 200), (WIDTH - SEV - 200, SEV + 200),
             (SEV + 200, SEV + 400), (WIDTH - SEV, SEV + 400),
             (SEV + 200, HEIGHT - SEV - 150), (WIDTH - SEV - 200, HEIGHT - SEV - 250),
             (WIDTH - SEV - 200, HEIGHT - SEV), (SEV, HEIGHT - SEV), (WIDTH - SEV, HEIGHT - SEV),
             ]

line0 = [(SEV, SEV), (SEV, HEIGHT - SEV), (WIDTH - SEV, HEIGHT - SEV), (WIDTH - SEV, SEV), (SEV, SEV)]
line1 = [(SEV, SEV + 200), (WIDTH - SEV - 200, SEV + 200)]
line2 = [(SEV + 200, HEIGHT - SEV - 150), (SEV + 200, HEIGHT - SEV - 400)]
line3 = [(SEV + 200, HEIGHT - SEV - 400), (WIDTH - SEV, HEIGHT - SEV - 400)]
line4 = [(WIDTH - SEV - 200, HEIGHT - SEV - 250), (WIDTH - SEV - 200, HEIGHT - SEV)]
walls = [line0, line1, line2, line3, line4]


def getColour(pressed):
    if pressed:
        return PURPLE
    else:
        return LIGHTPURPLE


class Simulation:

    def __init__(self):
        pygame.init()
        self.delta_t = 0.1
        self.walls = walls
        self.robot = Robot(self.walls, 70)
        self.screen = pygame.display.set_mode((WIDTH + 150, HEIGHT))
        pygame.display.set_caption("Modular Robot Simulator")
        self.font = pygame.font.SysFont("Pokemon GB.ttf", 50)
        self.running = True
        self.localization = Localization(self.delta_t, [self.robot.x, self.robot.y, self.robot.theta])
        self.clock = pygame.time.Clock()
        self.history_ellipse = []

    def show(self, velocities):
        self.screen.fill(WHITE)

        # Fill in robot environment
        # Display walls
        for wall in self.walls:
            pygame.draw.aalines(self.screen, DARKGRAY, True, wall, 50)
        pygame.draw.circle(self.screen, PURPLE, (self.robot.x, self.robot.y), self.robot.radius)
        pygame.draw.line(self.screen, BLACK, (self.robot.x, self.robot.y), (self.robot.frontX, self.robot.frontY), 1)
        # Display all landmarks
        for point in landmarks:
            pygame.draw.circle(self.screen, PURPLE, point, 4)
        close_landmarks = self.robot.get_close_landmarks(landmarks)
        # Display line to close landmarks (at most 3), distance limit can be changed in robot.sensor_limit
        for close_landmark in close_landmarks:
            pygame.draw.line(self.screen, GREEN, close_landmark, (self.robot.x, self.robot.y), 1)
        if len(self.robot.real_track) > 1:
            pygame.draw.aalines(self.screen, DARKGRAY, False, self.robot.real_track, 50)
        # Display velocities
        left = "v = " + str(np.round(velocities[0], 2))
        self.screen.blit(self.font.render(left, 105, BLACK), (WIDTH + 10, HEIGHT / 2 - 305))
        right = "w = " + str(np.round(velocities[1], 2))
        self.screen.blit(self.font.render(right, 105, BLACK), (WIDTH + 10, HEIGHT / 2 - 255))
        theta = "θ = " + str(np.round(velocities[2] % 360, 2)) + "°"
        self.screen.blit(self.font.render(theta, 105, BLACK), (WIDTH + 10, HEIGHT / 2 - 205))
        # Visualization ellipse and predicted tracks
        self.visualization_predict()

        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(10)
            velocities = self.update()
            self.show(velocities)

    def update(self):
        # Quit the simulation
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()
                pygame.quit()

        # Get pressed keys and update robot position
        keys = pygame.key.get_pressed()
        movement = [keys[pygame.K_w], keys[pygame.K_s], keys[pygame.K_a], keys[pygame.K_d], keys[pygame.K_x]]
        velocities = self.robot.move(movement, self.delta_t)

        # Update localization
        close = self.robot.get_close_landmarks(landmarks)
        range = self.robot.getSensorRange(close)
        bearing = self.robot.getBearing(close)
        z = self.localization.getObservationPose(close, range, bearing)
        # z = [self.robot.x, self.robot.y, self.robot.theta]
        # print("Real values", self.robot.x, self.robot.y, self.robot.theta)
        # print("Estimated values", z)
        self.localization.kalmanFilter(velocities[0:2], z)
        return velocities

    def visualization_predict(self):
        # https://cookierobotics.com/007/
        ellipse, location = self.localization.get_ellipse()
        predict_track = self.localization.predict_track
        width, height, angle = ellipse
        x, y = location
        # init an transparent surface
        surface = pygame.Surface((600, 600), pygame.SRCALPHA)
        # draw ellipse on transparent surface,make sure the ellipse locate in the center of the surface
        pygame.draw.ellipse(surface, GREEN, (300 - width / 2, 300 - height / 2, width * 10, height * 10), 3)
        # rotate the  ellipse by rotating the whole surface
        surface2 = pygame.transform.rotate(surface, angle)
        self.history_ellipse.append(surface2)
        # get the ellipse location on surface
        for sur in self.history_ellipse:
            rcx, rcy = sur.get_rect().center
            # show ellipse and adjust the location to the track
            # self.screen.blit(sur, (self.robot.x - rcx, self.robot.y - rcy))
            self.screen.blit(surface2, (x - rcx, y - rcy))
        if len(predict_track) > 1:
            pygame.draw.aalines(self.screen, DARKGRAY, False, predict_track, 50)

    def stop(self):
        self.running = False
        exit()


def main():
    simulation = Simulation()
    simulation.run()


if __name__ == "__main__":
    main()
