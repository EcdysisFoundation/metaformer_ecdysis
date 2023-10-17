

class EarlyStopper:
    """ Early stopper class to signal when the metric is not improving. """

    def __init__(self, patience=10, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.metric_max = 0

    def early_stop(self, metric):
        """
        Update internal state of the early stopper and return True if it's time to stop. Assumes that the metric is
        the higher, the better. Pass the negative of the metric if the lower, the better.
        Args:
            metric: Single float value to monitor.

        Returns: True when the metric is not improving for more than patience epochs. False otherwise.
        """
        if metric > self.metric_max:
            self.metric_max = metric
            self.counter = 0
        elif metric < (self.metric_max + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
