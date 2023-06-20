import numpy as np


class QTable:
    '''
    A class to store and update Q values, representing the value of actions
    performed within a given state.
    '''
    def evaluate(self, state):
        '''Get the Q-value of a given action in a given state,
        or all actions in a given state.

        Don't need to reimplement this for other Q-tables.

        '''
        return self.table[int(state), :]

    def update(self, state, action, reward):
        '''Update the Q-values.

        Run an update on the Q-values after performing `action` starting from
        `state` and receiving `reward`.
        '''
        raise NotImplementedError

    def store_data(self, data_dict: dict):
        '''
        Store the q_table for each trial into a dictionary, then reset values
        for the next simulation.

        Arguments:
        ----------
        data_dict: dictionary
            The dictionary into which the q tables will be stored.
        '''
        data_dict['q_tables'] = np.array(self.q_tables)


class RescorlaWagner(QTable):
    '''
    An update rule that follows the Rescorla-Wagner (or delta) learning model.
    '''
    def __init__(self, initial_q_values: np.ndarray, learning_rate: float):
        '''
        Arguments:
        ----------
        initial_q_values: np.ndarray, shape: #states x #actions
            The initial values for each state-action pair.

        learning_rate: float
            Alpha parameter in RW learning, determines the extent to which
            reward prediction errors update q values.

        '''
        self.initial_q_values = initial_q_values
        self.table = initial_q_values
        self.learning_rate = learning_rate
        self.q_tables = []
        self.rpe = []

    def update(self, state: object, action: object, reward: int):
        '''
        Based on the state-action pair and reward outcome, updates the
        corresponding Q value.

        Arguments:
        ----------
        state: object (Tone class)
            The current state of the environment.

        actions: object (Action class)
            The action chosen on the current trial.

        reward: int
            The reward received on the current trial.
        '''
        current_value = self.table[int(state), int(action)]
        rpe = reward - current_value
        self.table[int(state), int(action)] += (self.learning_rate * rpe)

        self.q_tables.append(np.copy(self.table))
        self.rpe.append(rpe)

    def store_data(self, data_dict):
        '''
        Stores Q tables and RPEs on a trial-by-trial basis, then resets data
        for the next simulation.
        '''
        super().store_data(data_dict)
        data_dict['rpe'] = np.array(self.rpe, dtype=np.float16)
        # Reset class instance.
        if type(self) == RescorlaWagner:
            self.__init__(self.initial_q_values, self.learning_rate)


class RescorlaWagnerLapse(RescorlaWagner):
    '''
    A delta learning rule inspired from the Rescorla-Wagner learning model,
    but incorporating lapse trials. On these trials, feedback does not update
    value estimates, or updates it using a different learning rate.
    '''
    def __init__(self, initial_q_values: np.ndarray, learning_rate: float,
                 engaged: object, lapse_lr: float = 0):
        '''
        '''
        self.initial_q_values = initial_q_values
        self.table = initial_q_values
        self.learning_rate = learning_rate
        self.engaged = engaged
        self.lapse_learning_rate = lapse_lr
        self.q_tables = []
        self.rpe = []
        self.lr = []

    def update(self, state: object, action: object, reward: int):
        '''
        '''
        current_value = self.table[int(state), int(action)]
        rpe = reward - current_value
        # Determine which learning rate to use, based on engagement.
        if self.engaged.trials[len(self.rpe)] == 1:
            lr = self.learning_rate
        else:
            lr = self.lapse_learning_rate
        # Update Q value.
        self.table[int(state), int(action)] += (lr * rpe)

        self.q_tables.append(np.copy(self.table))
        self.rpe.append(rpe)
        self.lr.append(lr)

    def store_data(self, data_dict):
        super().store_data(data_dict)
        data_dict['learning_rate'] = np.array(self.lr, dtype=np.float16)
        # Reset class instance.
        if type(self) == RescorlaWagnerLapse:
            self.__init__(self.initial_q_values, self.learning_rate,
                          self.engaged, self.lapse_learning_rate)
