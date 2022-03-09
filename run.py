import numpy as np
import pygame

from Robot import Robot
from Localization import Localization

SEV = 35  # SCREEN_EDGE_VACANCY
WIDTH = 835
HEIGHT = 835
# A list of RGB values for the colours used in the game.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARKGRAY = (125, 130, 138)
PURPLE = (114, 85, 163)
LIGHTPURPLE = (185, 167, 217)

def getColour(pressed):
    if pressed:
        return PURPLE
    else:
        return LIGHTPURPLE

class Simulation:

    def __init__(self):
        pygame.init()
        self.delta_t = 0.1
        self.outer_wall = [(SEV, SEV), (SEV, WIDTH - SEV), (WIDTH - SEV, HEIGHT - SEV), (HEIGHT - SEV, SEV), (SEV, SEV)]
        self.robot = Robot(self.outer_wall, 70)
        self.screen = pygame.display.set_mode((WIDTH + 350, HEIGHT))
        pygame.display.set_caption("Modular Robot Simulator")
        self.font = pygame.font.SysFont("Pokemon GB.ttf", 50)
        self.running = True
        self.localization = Localization(self.delta_t, [self.robot.x, self.robot.y, self.robot.theta])
        self.clock = pygame.time.Clock()

    def show(self, velocities):
        self.screen.fill(WHITE)

        # Fill in robot environment
        pygame.draw.aalines(self.screen, DARKGRAY, True, self.outer_wall, 50)
        pygame.draw.circle(self.screen, PURPLE, (self.robot.x, self.robot.y), self.robot.radius)
        pygame.draw.line(self.screen, BLACK, (self.robot.x, self.robot.y), (self.robot.frontX, self.robot.frontY), 1)

        # Display velocities
        left = "v = " + str(np.round(velocities[0],2))
        self.screen.blit(self.font.render(left, 105, BLACK), (WIDTH + 50, HEIGHT / 2 - 305))
        right = "w = " + str(np.round(velocities[1],2))
        self.screen.blit(self.font.render(right, 105, BLACK), (WIDTH + 50, HEIGHT / 2 - 255))
        theta = "θ = " + str(np.round(velocities[2] % 360, 2)) + "°"
        self.screen.blit(self.font.render(theta, 105, BLACK), (WIDTH + 50, HEIGHT / 2 - 205))

        # Display localization approximation

        pygame.display.flip()

    def run(self):
        while self.running:
            self.clock.tick(50)
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
        z = self.localization.getObservationPose(self.robot.x, self.robot.y, self.robot.theta)
        print(self.robot.x, self.robot.y)
        self.localization.kalmanFilter(velocities[0:2], z)

        return velocities

    def stop(self):
        self.running = False
        exit()


def main():
    simulation = Simulation()
    simulation.run()


if __name__ == "__main__":
    main()
