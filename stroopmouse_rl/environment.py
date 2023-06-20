import numpy as np


class Environment:
    '''
    A class to manage states of the environment(i.e. tones).
    '''
    def __init__(self):
        raise NotImplementedError

    def store_data(self, data_dict: dict):
        '''
        Stores the state sequence in a dictionary.
        '''
        data_dict['state_sequence'] = self.state_sequence.astype(str)


class RandomStates(Environment):
    '''
    An environment that randomly alternates between states, with a bias term.
    '''
    def __init__(self, state_space: list, n_trials: int, p: list = None):
        '''
        '''
        self.n_trials = n_trials
        self.state_space = state_space
        self.p = p

        self.state_sequence = self.set_random_states()

    def set_random_states(self):
        '''
        Generate a random sequence of states, with given probabilities.
        '''
        if self.p is None:
            self.p = np.ones(len(self.state_space))/len(self.state_space)

        else:  # Normalize probabilities.
            self.p = np.array(self.p)/np.sum(self.p)

        return np.random.choice(self.state_space, size=self.n_trials, p=self.p)

    def store_data(self, data_dict: dict):
        '''
        Stores the state sequence in a dictionary and reset the state sequence.
        '''
        data_dict['state_sequence'] = self.state_sequence.astype(str)
        # Reset class instance.
        self.__init__(self.state_space, self.n_trials, self.p)


class PreDefStates(Environment):
    '''
    An environment that provides a predefined sequence of states to the agent.
    State sequence can be str or Tone objects.
    '''
    def __init__(self, state_space, state_sequence):
        self.state_space = state_space
        self.n_trials = len(state_sequence)

        if type(state_sequence[0]) != object:
            self.state_sequence = self.get_state_objects(state_space,
                                                         state_sequence)

        else:
            self.state_sequence = state_sequence

    def get_state_objects(self, states, state_sequence):
        '''
        Given a sequence of str or int corresponding to states, generates a
        corresponding sequence of state (i.e.Tone) objects.
        '''
        obj_sequence = np.empty(len(state_sequence), dtype=object)
        state_str = np.array([str(i) for i in states]).astype(str)

        for trial in range(self.n_trials):
            index = np.where(state_str == str(state_sequence[trial]))[0][0]
            obj_sequence[trial] = states[index]

        return obj_sequence

    def store_data(self, data_dict: dict):
        '''
        Stores the state sequence in a dictionary and reset the state sequence.
        '''
        data_dict['state_sequence'] = self.state_sequence.astype(str)
        # Reset class instance.
        self.__init__(self.state_space, self.state_sequence)


class SingleState(Environment):
    '''
    An Environment object for a task with a single environmental state
    (e.g. two-armed bandit task).
    '''
    def __init__(self, state_space: list, n_trials: int):
        '''
        '''
        if len(state_space) > 1:
            raise ValueError('SingleState object must have state space of 1.')

        self.n_trials = n_trials
        self.state_space = state_space

        self.state_sequence = np.full(self.n_trials, self.state_space[0])

    def store_data(self, data_dict: dict):
        '''
        Stores the state sequence in a dictionary and reset the state sequence.
        '''
        data_dict['state_sequence'] = self.state_sequence.astype(str)
        # Reset class instance.
        self.__init__(self.state_space, self.state_sequence)


class EngagedTrials():
    '''
    A class to determine, on each trial, whether the agent is engaged or not.
    '''
    def __init__(self, lapse_rate, n_trials):
        '''
        '''
        self.lapse_rate = lapse_rate
        self.n_trials = n_trials
        self.get_trials()

    def get_trials(self):
        '''
        Generate (or re-generate) the sequence of engaged vs lapsed trials.
        '''
        self.trials = np.random.choice([1, 0], self.n_trials,
                                       p=[1-self.lapse_rate, self.lapse_rate])
