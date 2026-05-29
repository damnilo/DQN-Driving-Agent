class ActionMapper:
    def __init__(self):
        self.action = {
            # Pravo — ekspert ovo koristi 91% vremena
            0:  [ 0.00,  0.60],
            1:  [ 0.00,  0.30],
            2:  [ 0.00, -0.30],  # kočenje

            # Fino lijevo — gdje ekspert stvarno skreće
            3:  [-0.05,  0.50],
            4:  [-0.10,  0.45],
            5:  [-0.15,  0.40],
            6:  [-0.20,  0.35],
            7:  [-0.25,  0.30],
            8:  [-0.30,  0.25],

            # Fino desno — simetrično
            9: [ 0.05,  0.50],
            10: [ 0.10,  0.45],
            11: [ 0.15,  0.40],
            12: [ 0.20,  0.35],
            13: [ 0.25,  0.30],
            14: [ 0.30,  0.25],

            15: [ -0.30, -0.15],
            16: [ -0.10, -0.25],
            17: [ 0.30, -0.15],
            18: [ 0.10, -0.25]
        }

    def map(self, discrete_action):
        return self.action[int(discrete_action)]

    def num_actions(self):
        return len(self.action)