import numpy as np
import math
from sympy.solvers import solve
from sympy import Symbol


class Localization:

    def __init__(self, delta_t, state):
        self.state = state
        self.covariance = np.dot(np.identity(3), 0.1)
        self.delta_t = delta_t
        self.A = np.identity(3)
        self.C = np.identity(3)
        self.R = np.dot(np.identity(3), 0.1)
        self.Q = np.dot(np.identity(3), 0.1)
        self.B = [[self.delta_t * np.cos(self.state[2]), 0], [self.delta_t * np.sin(self.state[2]), 2],
                  [0, self.delta_t]]
        self.predict_track = [(state[0], state[1])]

    def kalmanFilter(self, action, observation):
        # Prediction
        # B = [[self.delta_t * np.cos(self.state[2]), 0], [self.delta_t * np.sin(self.state[2]), 2], [0, self.delta_t]]
        statePrediction = np.matmul(self.A, self.state) + np.matmul(self.B, action)
        covariancePrediction = np.matmul(np.matmul(self.A, self.covariance), np.transpose(self.A)) + self.R
        # Correction
        aux = np.matmul(np.matmul(self.C, statePrediction), np.transpose(self.C)) + self.Q
        K = np.matmul(np.matmul(covariancePrediction, np.transpose(self.C)), np.linalg.pinv(aux.astype(float)))
        newState = statePrediction + np.matmul(K, (observation - np.matmul(self.C, statePrediction)))
        newCovariance = np.matmul(np.identity(3) - np.matmul(K, self.C), covariancePrediction)
        # Update
        x = newState[0]
        y = newState[1]
        self.predict_track.append((x, y))

        self.state = newState
        self.covariance = newCovariance

    def getObservationPose(self, closelandmarks, ranges, bearings):
        estX = 0
        estY = 0
        estAngle = 0
        size = len(closelandmarks)

        if size == 3:
            x1, y1 = closelandmarks[0]
            x2, y2 = closelandmarks[1]
            x3, y3 = closelandmarks[2]
            xx = Symbol('xx')
            yy = Symbol('yy')
            eq1 = (xx - x1) ** 2 + (yy - y1) ** 2 - (ranges[0]) ** 2 - (xx - x3) ** 2 - (yy - y3) ** 2 + (
                ranges[2]) ** 2
            eq2 = (xx - x2) ** 2 + (yy - y2) ** 2 - (ranges[1]) ** 2 - (xx - x3) ** 2 - (yy - y3) ** 2 + (
                ranges[2]) ** 2
            solutions = solve((eq1, eq2), (xx, yy))
            estX = solutions[xx]
            estY = solutions[yy]

            estAngle = (math.atan2((y1 - estY), (x1 - estX)) - bearings[0])
        return [estX, estY, estAngle]
        # slide 20 in "19 ARS - Localization with Kalman Filter.pdf"

    def get_ellipse(self):
        a = self.covariance[0][0]
        b = self.covariance[0][1]
        c = self.covariance[1][0]
        lambda_1 = (a + c) / 2 + math.sqrt(((a - c) / 2) ** 2 + b ** 2)
        lambda_2 = (a + c) / 2 - math.sqrt(((a - c) / 2) ** 2 + b ** 2)
        width = 2 * math.sqrt(lambda_1)
        height = 2 * math.sqrt(abs(lambda_2))
        theta = 0
        if b == 0 and a < c:
            theta = math.pi / 2
        else:
            theta = math.atan2(lambda_1 - a, b)
        x = self.state[0]
        y = self.state[1]
        ellipse = (width, height, theta)
        location = (x, y)
        return ellipse, location
