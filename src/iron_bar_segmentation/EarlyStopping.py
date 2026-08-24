"""
검증 점수가 더 이상 좋아지지 않으면 학습을 멈추는 도구.

계속 학습하면 학습 데이터에만 맞춰지는(overfitting) 시점이 오기 때문에,
일정 횟수 동안 개선이 없으면 중단한다.
"""


class EarlyStopping():
    def __init__(self, patience=5, mode = 'min', delta = 0.0):
        """
        :param patience: 개선이 없어도 참고 기다릴 epoch 수
        :param mode: 'min'이면 점수가 작아질수록 좋음(loss), 'max'면 클수록 좋음(정확도 등)
        :param delta: 이 값보다 크게 좋아져야 "개선"으로 인정한다
        """
        self.counter = 0
        self.patience = patience
        self.mode = mode
        self.delta = delta
        self.best_score = None
        self.early_stop = False

    def step(self, current_score):
        """
        매 epoch마다 검증 점수를 넣어 호출한다.

        :return: (이번에 개선되었는가, 학습을 멈춰야 하는가)
                 첫 번째 값은 checkpoint 저장 여부를 판단하는 데 쓴다.
        """
        # 첫 호출은 비교 대상이 없으므로 무조건 개선으로 본다.
        if self.best_score is None:
            self.best_score = current_score
            return True, False

        improvement = (current_score < self.best_score - self.delta if self.mode == 'min'
                       else current_score > self.best_score + self.delta)

        if improvement:
            # 기록을 갱신했으므로 참을성 카운터를 초기화한다.
            self.best_score = current_score
            self.counter = 0
            return improvement, self.early_stop
        else:
            # 개선이 없으면 카운터를 올리고, patience를 넘기면 중단 신호를 켠다.
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return improvement, self.early_stop