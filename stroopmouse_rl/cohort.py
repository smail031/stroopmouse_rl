import numpy as np


class Cohort:
    '''
    A group of mouse agents that will be trained simultaneously.
    '''

    def __init__(self, n_mice: int, rule: object, policy: object,
                 q_table: object, environment: object):
        '''
        Arguments:
        -----------
        n_mice: int
            The number of mice in the cohort.

        rule: object
            Instance of a Rule class. Maps the relationship between states,
            actions and outcomes. Also determines when rule switches occurr.

        policy: object
            Instance of a Policy class. Determines how Q values are converted
            to action selection probabilities.

        q_table: object
            Instance of a QTable class. Stores a Q value for each state-action
            pair, and determines how these Q values are updated.

        environment: object
            Instance of an Environment class. Generates an environmental state
            for each trial.
        '''
        self.mouse_objects = np.empty(n_mice, dtype=object)
        # Initialize Mouse objects.
        for ms in range(n_mice):
            self.mouse_objects[ms] = Mouse(rule, policy, q_table, environment)

    def simulate(self, nested: bool = True):
        '''
        Simulate each mouse agent and store the data. 

        Arguments:
        ----------
        nested: bool
            If false, returns a 1D np array of dicts for each mouse. If true,
            returns a single dict in which each key contains nested 1D np arrays
            for each mouse.
        '''
        self.res = np.empty(len(self.mouse_objects), dtype=dict)

        for mouse in range(len(self.mouse_objects)):
            self.res[mouse] = self.mouse_objects[mouse].simulate()

        if nested:
            self.nested_arrays()

    def nested_arrays(self):
        '''
        Reorganizes simulation results into a single dict, with each key
        containing nested arrays for that measurement for all mice.
        '''
        new_dict = {}
        keys = list(self.res[0].keys())

        for key in keys:
            new_dict[key] = np.empty(len(self.mouse_objects), dtype=np.ndarray)

            for ms in range(len(self.mouse_objects)):
                new_dict[key][ms] = self.res[ms][key]

        self.res = new_dict


class Mouse:

    def __init__(self, rule: object, policy: object,
                 q_table: object, environment: object):
        '''
        Arguments:
        ----------
        rule: object
            Instance of a Rule class. Maps the relationship between states,
            actions and outcomes. Also determines when rule switches occurr.

        policy: object
            Instance of a Policy class. Determines how Q values are converted
            to action selection probabilities.

        q_table: object
            Instance of a QTable class. Stores a Q value for each state-action
            pair, and determines how these Q values are updated.

        environment: object
            Instance of an Environment class. Generates an environmental state
            for each trial.
        '''

        self.rule = rule
        self.policy = policy
        self.q_table = q_table
        self.environment = environment

    def simulate(self):
        '''Run the simulated experiment.'''
        for state in self.environment.state_sequence:  # Iterate through trials
            action_values = self.q_table.evaluate(int(state))
            action = self.policy.choose(action_values)
            performance, reward = self.rule.outcome(int(state), int(action))
            self.q_table.update(int(state), int(action), reward)
            self.rule.trial_end()

        data = {}  # Will store all session data.
        for i in [self.rule, self.policy, self.q_table, self.environment]:
            i.store_data(data)

        return data
