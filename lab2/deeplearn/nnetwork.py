import numpy as np

class NeuralNetwork:
    def __init__(self):
        self._epoch = 5
        self._learning_rate = 0.1
        self._batch_size = 5
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def configure(self, *, epoch=5, learning_rate=0.1, batch_size=5):
        if epoch < 1:
            raise ValueError("Epoch cannot be lower than 1")
        if batch_size < 1:
            raise ValueError("Batch size cannot be lower than 1")
        self._epoch         = epoch
        self._learning_rate = learning_rate
        self._batch_size    = batch_size

    def train(self, train_data, train_targets, *, loss_func):
        for _ in range(self._epoch):
            for i in range(0, len(train_data), self._batch_size):
                frame_end = i + self._batch_size
                data    = train_data[i:frame_end]
                targets = train_targets[i:frame_end]

                y_probs = self._forward_calc(data)
                loss = loss_func.calc_forward(y_probs, targets)

                self._backward_calc(loss, loss_func)
                self._update_all()

    def predict(self, input_data):
        return self._forward_calc(input_data)

    def _forward_calc(self, train_data):
        current_data = train_data
        
        for l in self._layers:
            current_data = l.calc_forward(current_data)

        return current_data

    def _backward_calc(self, loss, loss_func):
        curr_derivative = loss_func.calc_backward(loss)
        
        for l in self._layers[::-1]:
            curr_derivative = l.calc_backward(curr_derivative)

    def _update_all(self):
        for l in self._layers:
            l.update(self._learning_rate)

class LinearNeuralNetwork:
    def __init__(self):
        self._epoch = 5
        self._window = 5
        self._learning_rate = 0.1
        self._layers = []

    def add_layer(self, layer):
        self._layers.append(layer)

    def configure(self, *, epoch=5, window=5, learning_rate=0.1):
        if epoch < 1:
            raise ValueError("Epoch cannot be lower than 1")
        if window < 1:
            raise ValueError("Window cannot be lower than 1")
        self._epoch         = epoch
        self._window        = window
        self._learning_rate = learning_rate

    def train(self, train_data, train_labels, *, loss_func):
        td_size = len(train_data)

        for _ in range(self._epoch):
            for i in range(0, td_size - self._window):
                frame_end = i + self._window
                # Most of framework is made for 2D (sample_size, feats) matrices
                # So, generalize
                data   = np.expand_dims(train_data[i:frame_end], 0)
                target = np.expand_dims(train_labels[frame_end], 0)

                y_prob = self._forward_calc(data)
                loss   = loss_func.calc_forward(y_prob, target)

                self._backward_calc(loss, loss_func)
                self._update_all()

    def predict(self, input_data, *, predict_size=None):
        window = self._window
        id_size = len(input_data)

        if id_size < window:
            raise ValueError(f"Not enough data to predict (have {id_size}, need at least {window}")

        if predict_size is None:
            predict_size = id_size - window

        can_predict = id_size - window + 1
        can_predict = can_predict if can_predict < window else predict_size
        if can_predict < window and predict_size > can_predict:
            raise ValueError(f"Can only predict {can_predict} when expected {predict_size}")

        res = np.zeros((predict_size, ))

        # Predict size can be less then we can infere from input data or more
        # If it is less, infere what is needed
        # If it is more, additionally infere from our predictions
        split_point = min(predict_size, id_size - window)

        for i in range(0, split_point):
            frame_end = i + window
            data = np.expand_dims(input_data[i:frame_end], 0)
            res[i] = self._forward_calc(data)

        for i in range(split_point, predict_size):
            frame_start = i - window
            data = np.expand_dims(res[frame_start:i], 0)
            res[i] = self._forward_calc(data)

        return res

    def _forward_calc(self, train_data):
        current_data = train_data
        
        for l in self._layers:
            current_data = l.calc_forward(current_data)

        return current_data

    def _backward_calc(self, loss, loss_func):
        curr_derivative = loss_func.calc_backward(loss)
        
        for l in self._layers[::-1]:
            curr_derivative = l.calc_backward(curr_derivative)

    def _update_all(self):
        for l in self._layers:
            l.update(self._learning_rate)



