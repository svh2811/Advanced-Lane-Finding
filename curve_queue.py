import numpy as np

class QueueCurve:
    def __init__(self, n):
        self.counter = 0
        self.queue = np.zeros((n, 3))
        self.n = n

    def enqueue(self, curve):
        self.queue[self.counter, :] = curve
        self.counter = (self.counter + 1) % self.n

    def mean(self):
        return self.queue.mean(axis = 0)
