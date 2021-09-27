import numpy as np

class MSELoss:
    def calc_forward(self, p, y):
        self.p = p
        self.y = y

        diffs = ( np.square(p - y) )

        return diffs.mean(axis=1)

    def calc_backward(self, _):
        return self.p - self.y

    def update(self, _):
        # Nothing to update
        pass

class CrossEntropyLoss:
    def calc_forward(self, p, y):
        self.p = p
        self.y = y
        p_of_y = p[np.arange(len(y)), y]
        log_prob = np.log(p_of_y)
        return -log_prob.mean()

    def calc_backward(self, _):
        dlog_softmax = np.zeros_like(self.p)
        dlog_softmax[np.arange(len(self.y)), self.y] -= 1.0/len(self.y)
        return dlog_softmax / self.p

    def update(self, _):
        # Nothing to update
        pass
