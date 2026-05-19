class ActionMapper:

    def __init__(self):

        self.action = {
            0: [-0.3, 0.4],

            1: [-0.1, 0.5],

            2: [0.0, 0.7],

            3: [0.1, 0.5],

            4: [0.3, 0.4],

            5: [0.0, -0.2],

            6: [-0.3, -0.2],

            7: [0.3, -0.2],

            8: [0.0, 0.4]
        }

    def map(self, discrete_action):

        return self.action[discrete_action]
    
    def num_actions(self):

        return len(self.action)