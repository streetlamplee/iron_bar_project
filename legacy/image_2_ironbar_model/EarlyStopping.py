

class EarlyStopping():
    def __init__(self, patience=5, mode = 'min', delta = 0.0):
        self.counter = 0
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.early_stop = False

    def step(self, current_score):

        if self.best_score is None:
            self.best_score = current_score
            return True, False

        improvement = (current_score < self.best_score - self.delta if self.mode == 'min'
                       else current_score > self.best_score + self.delta)

        if improvement:
            self.best_score = current_score
            self.counter = 0
            return improvement, self.early_stop
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return improvement, self.early_stop