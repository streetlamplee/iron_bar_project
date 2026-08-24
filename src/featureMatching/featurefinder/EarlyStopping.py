"""
검증 점수가 나아지지 않으면 학습을 중단시키는 도구.
(iron_bar_segmentation/EarlyStopping.py 와 같은 내용의 사본이다.)
"""



class EarlyStopping():
    def __init__(self, patience=5, mode = 'min', delta = 0.0):
        """
        :param patience: 개선이 없어도 기다릴 epoch 수
        :param mode: 'min'이면 작을수록 좋은 지표(loss), 'max'면 클수록 좋은 지표
        :param delta: 이 값보다 크게 좋아져야 개선으로 인정한다
        """
        self.counter = 0
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.early_stop = False

    def step(self, current_score):
        """:return: (개선되었는가, 멈춰야 하는가)"""

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