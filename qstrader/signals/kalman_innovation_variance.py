import numpy as np
import pandas as pd
from qstrader.signals.signal import Signal


class InnovationVarianceSignal(Signal):
    def __init__(self, start_dt, universe, delta, ve):
        self.state_mean = np.zeros(2)  # Beta estimate (initially zero)
        self.state_cov = np.zeros((2, 2))  # State covariance (P)
        self.process_noise = delta / (1 - delta) * np.eye(2)  # Vw
        self.measurement_noise = ve  # Ve (measurement noise variance)
        self.errors = []  # Store errors for debugging

        super().__init__(start_dt, universe, [1])

    def predict_innovation_variance(self, assets):

        X_price = pd.Series(self.buffers.prices[f"{assets[0]}_1"])[0]

        Y_price = pd.Series(self.buffers.prices[f"{assets[1]}_1"])[0]

        # Augment x with 1 for intercept
        x = np.array([X_price, 1.0])  # Shape: (2,)

        # === PREDICTION STEP ===
        self.state_mean = self.state_mean  # Beta remains same before update
        R = self.state_cov + self.process_noise  # Updated state covariance

        # Predict measurement (yhat)
        yhat = np.dot(x, self.state_mean)  # Predicted y (EWC price)

        # Predict measurement variance (innovation variance Q)
        Q = np.dot(x, np.dot(R, x)) + self.measurement_noise

        # === UPDATE STEP ===
        error = Y_price - yhat  # Innovation (actual - predicted)
        self.errors.append(error)

        # Kalman gain
        K = np.dot(R, x) / Q

        # Update state mean (beta)
        self.state_mean = self.state_mean + K * error
        hedge_ratio = self.state_mean[0]

        # Update state covariance (P)
        self.state_cov = R - np.outer(K, np.dot(x, R))

        return hedge_ratio, Q, error

    def __call__(self, assets):
        return self.predict_innovation_variance(assets)
