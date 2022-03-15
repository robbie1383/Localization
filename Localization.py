import numpy as np
import math


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
        self.predict_track = []

    def kalmanFilter(self, action, observation):
        # Prediction
        B = [[self.delta_t * np.cos(self.state[2]), 0], [self.delta_t * np.sin(self.state[2]), 2], [0, self.delta_t]]
        statePrediction = np.matmul(self.A, self.state) + np.matmul(B, action)
        covariancePrediction = np.matmul(np.matmul(self.A, self.covariance), np.transpose(self.A)) + self.R
        # Correction
        aux = np.matmul(np.matmul(self.C, statePrediction), np.transpose(self.C)) + self.Q
        K = np.matmul(np.matmul(covariancePrediction, np.transpose(self.C)), np.linalg.pinv(aux))
        newState = statePrediction + np.matmul(K, (observation - np.matmul(self.C, statePrediction)))
        newCovariance = np.matmul(np.identity(3) - np.matmul(K, self.C), covariancePrediction)
        # Update
        x = newState[0]
        y = newState[1]
        self.predict_track.append((x, y))
        self.state = newState
        self.covariance = newCovariance

    def getObservationPose(self, x, y, theta, actionn):
        xt = np.matmul(self.A, [x, y, theta]) + np.matmul(self.B, actionn) + self.R
        z = np.matmul(self.C, xt) + self.Q
        return z
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
