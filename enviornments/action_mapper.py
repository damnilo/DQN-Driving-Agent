class ActionMapper:
    def __init__(self):
        self.action = {
            # Pravo — ekspert ovo koristi 91% vremena
            0:  [ 0.00,  0.60],
            1:  [ 0.00,  0.35],
            2:  [ 0.00, -0.30],  # kočenje

            # Fino lijevo — gdje ekspert stvarno skreće
            3:  [-0.05,  0.55],
            4:  [-0.10,  0.50],
            5:  [-0.15,  0.45],
            6:  [-0.20,  0.40],
            7:  [-0.25,  0.35],
            8:  [-0.30,  0.28],
            9:  [-0.35, 0.25],
            10: [-0.40, 0.22],
            11: [-0.45, 0.18],
            12: [-0.50, 0.15],
            13: [-0.55, 0.12],

            # Fino desno — simetrično
            14: [ 0.05,  0.55],
            15: [ 0.10,  0.50],
            16: [ 0.15,  0.45],
            17: [ 0.20,  0.40],
            18: [ 0.25,  0.35],
            19: [ 0.30,  0.28],
            20: [ 0.35,  0.25],
            21: [ 0.40,  0.22],
            22: [ 0.45,  0.18],
            23: [ 0.50,  0.15],
            24: [ 0.55,  0.12],
            

            25: [ -0.30, -0.15],
            26: [ -0.10, -0.25],
            27: [ 0.30, -0.15],
            28: [ 0.10, -0.25],

        }

    def map(self, discrete_action):
        return self.action[int(discrete_action)]

    def num_actions(self):
        return len(self.action)