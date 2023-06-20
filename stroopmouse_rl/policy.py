import numpy as np
from scipy.special import softmax
from hmmlearn.hmm import MultinomialHMM


class Policy:
    '''A behavioural policy is a probability distribution over actions.'''

    def choose(self, action_values):
        '''
        Choose an action according to the policy, given the action values.

        This is equivalent to sampling from a probability distribution over
        actions.
        '''
        raise NotImplementedError

    def choice_history_objects(self) -> object:
        '''
        If there is a predefined choice history, will return the choice for the
        current trial.
        '''
        choices = np.empty(len(self.choice_history), dtype=object)
        action_names = np.array([i.name for i in self.actions]).astype(str)

        for trial in range(len(self.choice_history)):
            choice_index = np.where(action_names
                                    == str(self.choice_history[trial]))[0][0]
            choices[trial] = self.actions[choice_index]
        return choices

    def store_data(self, data_dict: dict):
        '''
        Store choice data in a dictionary, then resets choices for the
        next simulation.
        '''
        data_dict['choices'] = np.array(self.choices)


class EGreedyPolicy(Policy):
    '''
    Epsilon-greedy behavioural policy. Each trial, the agent will either
    exploit (choose action with highest Q value) or explore (choose a random
    action). The rate of exploration is set by self.explore_rate.
    '''
    def __init__(self, actions: list, explore_rate: float,
                 choice_history: np.ndarray = None):
        '''
        Arguments:
        ----------
        actions: list
            A list of Action objects representing the action space.

        explore_rate: float
            The fraction of trials in which to explore (choose random action).

        choice_history: np.ndarray
            For each trial, provides a predefined choice that will be taken.
            Used for fitting model to real behavioral data.
        '''
        self.actions = actions
        self.explore_rate = explore_rate
        self.choices = []

        self.choice_history = choice_history

        if self.choice_history is not None:
            if self.choice_history.dtype != object:
                self.choice_history = self.choice_history_objects()

    def choose(self, action_values: np.ndarray) -> object:
        '''
        If it's an explore trial, choose randomly between all actions.
        Otherwise, choose action with highest action value.
        '''
        if np.random.rand() < self.explore_rate:
            choice = np.random.choice(self.actions)
        else:
            choice = self.actions[np.argmax(action_values)]

        self.choices.append(str(choice))
        return choice

    def store_data(self, data_dict: dict):
        '''
        '''
        super().store_data(data_dict)
        # Reset class instance.
        self.__init__(self.actions, self.explore_rate, self.choice_history)


class SoftMaxPolicy(Policy):
    '''
    Behavioural policy that transforms action values into choice
    probabilities using the softmax function.
    '''
    def __init__(self, actions: list, softness: float,
                 choice_history: np.ndarray = None):
        '''
        Arguments:
        ----------
        actions: list
            A list of all possible actions.

        softness: float
            Inverse temperature of softmax function, i.e. scaling factor for
            q values. Higher softness -> more exploitation.

        choice_history: np.ndarray
            For each trial, provides a predefined choice that will be taken.
            Used for fitting model to real behavioral data.
        '''
        self.actions = actions
        self.softness = softness
        self.choice_history = choice_history
        self.choices = []  # To store choices.

        if self.choice_history is not None:
            if self.choice_history.dtype != object:
                self.choice_history = self.choice_history_objects()

    def choose(self, action_values: np.ndarray) -> object:

        if self.choice_history is None:
            action_probabilities = softmax(self.softness * action_values)
            choice = np.random.choice(self.actions, p=action_probabilities)

        else:  # If there is a predefined sequence of choices.
            choice = self.choice_history[len(self.choices)]

        self.choices.append(str(choice))
        return choice

    def store_data(self, data_dict: dict):
        '''
        '''
        super().store_data(data_dict)
        # Reset class instance.
        if type(self) == SoftMaxPolicy:
            self.__init__(self.actions, self.softness, self.choice_history)


class SoftMaxLapse(SoftMaxPolicy):
    '''
    Behavioral policy that uses a softmax policy to make choices when the agent
    is engaged. However, there are "lapse trials" in which the agent makes
    a random choice.
    '''
    def __init__(self, actions: list, softness: float,
                 engaged: object,
                 choice_history: np.ndarray = None):
        '''
        Arguments:
        ----------
        actions: list
            A list of all possible actions.

        softness: float
            Inverse temperature of softmax function, i.e. scaling factor for
            q values. Higher softness -> more exploitation.

        lapse_rate: float
            A probability, between 0 and 1, that the agent will make a random
            choice.

        choice_history: np.ndarray, length n_trials
            For each trial, provides a predefined choice that will be taken.
            Used for fitting model to real behavioral data.

        engaged_history: np.ndarray, length n_trials
            For each trial, defines whether the agent is engaged (1) or not (0)
        '''
        self.actions = actions
        self.softness = softness
        self.choice_history = choice_history
        self.engaged = engaged
        self.choices = []  # To store choices.

        if self.choice_history is not None:
            if self.choice_history.dtype != object:
                self.choice_history = self.choice_history_objects()

    def choose(self, action_values: np.ndarray) -> object:
        '''
        Determine whether the current trial is a lapse trial. If so, choose
        randomly. If not, use softmax to choose.
        '''
        if self.engaged.trials[len(self.choices)] == 1:
            # Engaged; choose using softmax.
            return super().choose(action_values)

        else:  # Lapse; choose randomly.
            choice = np.random.choice(self.actions)
            self.choices.append(str(choice))
            return choice

    def store_data(self, data_dict: dict):
        '''
        '''
        super().store_data(data_dict)
        data_dict['engaged'] = np.array(self.engaged.trials)
        # Reset class instance.
        if type(self) == SoftMaxLapse:
            self.__init__(self.actions, self.softness, self.engaged,
                          self.choice_history)
            self.engaged.get_trials()


class PreDefActions(Policy):
    '''
    A behavioral policy that will use a predefined set of actions,
    regardless of underlying Q values.
    '''
    def __init__(self, actions: list, chosen_actions: np.ndarray):
        self.actions = actions
        self.action_names = np.array([i.name for i in actions])
        self.chosen_actions = chosen_actions
        self.choices = []

    def choose(self, action_values: np.ndarray) -> object:
        action_index = np.where(
            self.action_names == self.chosen_actions[0])[0][0]
        self.chosen_actions = self.chosen_actions[1:]  # remove the first trial
        choice = self.actions[action_index]

        self.choices.append(str(choice))
        return choice


class RandomChoice(Policy):
    '''
    A policy that randomly chooses between actions, with a bias term.
    '''
    def __init__(self, actions: list, p: list = None):
        '''
        Arguments:
        ----------
        actions: list of Action objects
            A list of all possible actions.

        p: list, default = None
            Probability of each action (must be same length as actions).
            If None, generates uniform distribution across actions.
        '''
        self.actions = actions
        self.p = p
        self.choices = []

        if self.p is None:
            self.p = np.ones(len(actions))/len(actions)
        else:
            self.p = np.array(self.p)/np.sum(self.p)

    def choose(self, action_values: np.ndarray) -> object:
        choice = np.random.choice(self.actions, p=self.p)
        self.choices.append(choice)
        return choice

    def store_data(self, data_dict: dict):
        '''
        '''
        super().store_data(data_dict)
        # Reset class instance.
        self.__init__(self.actions, self.p)


def hmm_states(start, transition, n_trials):
    '''
    Use a Hidden Markov Model to generate discrete state sequences.

    Arguments:
    ----------
    start: np.ndarray, length n_states
        The starting probabilities across states. Must sum to 1.

    transition: np.ndarray shape n_states x n_states
        The transition probability matrix for the HMM. Rows must sum to 1.

    n_trials: int
        The number of trials for which to generate state sequences.
    '''
    n_states = len(start)
    model = MultinomialHMM(n_components=n_states)
    model.startprob_ = start
    model.transmat_ = transition
    model.emissionprob_ = np.array([[1, 0],
                                    [0, 1]])

    state_sequence = model.sample(n_trials)

    return state_sequence[1]
