class Localization:

    def __init__(self, delta_t):
        self.delta_t = delta_t
        self.A = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.B = []
        self.C = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    def kalmanFilter(self, state, covariance, action, observation):
        pass