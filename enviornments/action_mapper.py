class ActionMapper:

    def __init__(self):

        self.action = {
            0: [-0.75, 0.2],

            1: [-0.5, 0.3],

            2: [-0.25, 0.4],

            3: [-0.1, 0.5],

            4: [0.0, 0.6],

            5: [0.0, 0.3],

            6: [0.0, 0.2],

            7: [0.1, 0.5],

            8: [0.25, 0.4],

            9: [0.5, 0.3],

            10: [0.75, 0.2],
        }

    def map(self, discrete_action):

        return self.action[discrete_action]
    
    def num_actions(self):

        return len(self.action)