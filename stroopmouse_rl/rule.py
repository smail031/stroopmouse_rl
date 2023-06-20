import numpy as np


class Rule:
    '''
    The rule maps each possible action to its associated reward probabilities.
    '''
    def __init__(self):
        raise NotImplementedError

    def map_outcomes(self):
        '''
        Establishes a map between states, actions and outcomes.
        '''
        raise NotImplementedError

    def outcome(self):
        '''
        Given a state and action, determines whether the agent chose correctly
        and samples from a reward probability distribution.
        '''
        raise NotImplementedError

    def trial_end(self):
        '''
        Indicates what should happen at the end of the trial: checking
        a performance criterion, switching rules, switching probabilities, etc.
        '''
        raise NotImplementedError

    def store_data(self, data_dict: dict):
        '''
        Stores trial-by-trial rule, performance and reward data into a dict,
        then resets stored data for the next simulation.
        '''
        data_dict['rule'] = np.array(self.rule_trial)
        data_dict['performance'] = np.array(self.performance)
        data_dict['reward'] = np.array(self.reward)


class ProbRev(Rule):
    '''
    Rule for a probabilistic reversal learning task. Once a performance
    criterion has been met, the state-action map is immediately inverted.

    Attributes:
    -----------
    self.p_reward: float
        Represents the probability of reward given correct choice. Probability
        of reward given an incorrect choice is 1 - self.p_reward.
    self.tones: list
        A list of Tone objects representing each of the possible auditory cues.
    self.actions: list
        A list of Action objects representing each of the possible actions.
    self.correct: numpy array
        An array of shape (tones x actions) that represents, for each
        tone-action pair, if that response is considered correct(1) or not(0).
    self.reward: numpy array
        An array of shape (tones x actions) storing the reward probabilities
        for each tone-action pair.

    Methods:
    --------
    outcomes: Using the current rule, generates corresponding 'correct'
        and 'reward' arrays.
    reversal: Reverses the rule and generates new correct and reward arrays.
    '''
    def __init__(self, p_reward: float, tones: list, actions: list,
                 criterion: list, reward_history: np.array = None,
                 initial: int = None):
        self.p_reward = p_reward
        self.tones = tones
        self.actions = actions
        self.criterion = criterion
        self.reward_history = reward_history
        self.initial = initial
        self.performance_history = []

        self.rule_trial = []
        self.performance = []
        self.reward = []
        self.reversals = []

        if self.initial is None:
            self.rule = int(np.random.choice([0, 1]))
        else:
            self.rule = self.initial

        self.map_outcomes()

    def map_outcomes(self):
        '''
        Using the rule, outputs an array that indicates, for each tone-action
        pair, whether the trial would be considered correct. Also outputs an
        array with associated reward probabilities for each tone-action pair.

        If properly formatted, the array indices represent the following
        tone-action pairs: [lowF-L, lowF-R], [hiF-L, hiF-R].
        '''
        correct_p = self.p_reward
        incorrect_p = 1-self.p_reward
        null_p = 0

        if self.rule == 0:
            self.correct_array = np.array([[1, 0, 0],
                                           [0, 1, 0]])
            self.reward_probs = np.array([[correct_p, incorrect_p, null_p],
                                          [incorrect_p, correct_p, null_p]])

        elif self.rule == 1:
            self.correct_array = np.array([[0, 1, 0],
                                           [1, 0, 0]])
            self.reward_probs = np.array([[incorrect_p, correct_p, null_p],
                                          [correct_p, incorrect_p, null_p]])

    def outcome(self, state, action):
        '''
        Determine whether the mouse will be rewarded for its choice.
        '''
        correct = int(self.correct_array[state, action])

        if self.reward_history is not None:
            # Take the first value of the vector, then remove it.
            reward = self.reward_history[0]
            self.reward_history = self.reward_history[1:]
        else:
            reward = int(np.random.rand() < self.reward_probs[state, action])

        self.performance_history.append(correct)
        self.performance.append(correct)
        self.reward.append(reward)
        return correct, reward

    def trial_end(self):
        '''
        Determines what will occurr at the end of the trial.
        '''
        if self.check_criterion():
            self.reversal()
            self.performance_history = []

        self.rule_trial.append(self.rule)

    def reversal(self):
        '''
        Reverses the rule (0->1 or 1->0) and generates new 'correct'
        and 'reward_probs' arrays accordingly.
        '''
        self.rule = int(1-self.rule)
        self.map_outcomes()
        self.reversals.append(len(self.performance))  # Store reversal trial.

    def check_criterion(self) -> bool:
        '''
        Uses the performance history to check whether a given criterion has
        been met. If so, triggers a rule switch.
        '''
        if len(self.performance_history) < self.criterion[1]:
            return False

        else:
            past_perform = self.performance_history[-self.criterion[1]:-1]
        return (np.sum(past_perform) >= self.criterion[0])

    def store_data(self, data_dict: dict):
        super().store_data(data_dict)
        data_dict['reversals'] = self.reversals
        # Reset class instance.

        if type(self) == ProbRev:
            self.__init__(p_reward=self.p_reward, tones=self.tones,
                          actions=self.actions, criterion=self.criterion,
                          reward_history=self.reward_history,
                          initial=self.initial)


class ProbRevCount(ProbRev):
    '''
    A Rule for a probabilistic reversal learning task. Similar to ProbRev rule,
    but there is a trial countdown between the agent reaching criterion and
    the reversal.
    '''
    def __init__(self, p_reward: float, tones: list, actions: list,
                 countdown_start: int, criterion: list,
                 reward_history: np.array = None, initial: int = None):

        super().__init__(p_reward, tones, actions, criterion,
                         reward_history, initial)
        self.countdown_start = countdown_start
        self.countdown = np.nan
        self.countdown_trials = []  # To store countdown values.

    def trial_end(self):
        '''
        Determines what will occurr at the end of the trial. If there isn't
        an ongoing trial countdown, will check the criterion. If there is a
        countdown, will subtract 1 and trigger a reversal if == 0.
        '''
        if np.isnan(self.countdown):
            if self.check_criterion():
                # Criterion met, countdown starts.
                self.countdown = self.countdown_start

        else:
            if self.countdown == 0:
                # Rule reversal.
                self.reversal()
                self.countdown = np.nan
                self.performance_history = []
            else:
                self.countdown -= 1

        self.countdown_trials.append(self.countdown)
        self.rule_trial.append(self.rule)

    def store_data(self, data_dict: dict):
        super().store_data(data_dict)
        data_dict['countdown'] = np.array(self.countdown_trials)

        # Reset class instance.
        if type(self) == ProbRevCount:
            self.__init__(self.p_reward, self.tones, self.actions,
                          self.countdown_start, self.criterion,
                          self.reward_history, self.initial)


class ProbTrialRev(ProbRevCount):
    '''
    A rule for a probabilistic reversal learning task similar to ProbRevCount
    rule. However, reversals are triggered at pre-defined trials rather than
    after reaching a performance criterion.
    '''

    def __init__(self, p_reward: float, tones: list, actions: list,
                 countdown_start: int, rev_trials: list,
                 reward_history: np.array = None, initial: int = None):

        criterion = None
        super().__init__(p_reward, tones, actions, countdown_start, criterion,
                         reward_history, initial)
        self.rev_trials = rev_trials
        self.criterion_trials = np.array(self.rev_trials) - (countdown_start+2)
        self.trial_counter = 0

    def trial_end(self):
        if np.isnan(self.countdown):
            # print(self.trial_counter)
            # print(self.criterion_trials)
            if self.trial_counter in self.criterion_trials:
                # Mouse becomes an "expert"
                self.countdown = self.countdown_start

        else:
            if self.countdown == 0:
                # Rule reversal
                self.reversal()
                self.countdown = np.nan

            else:
                self.countdown -= 1

        self.trial_counter += 1
        self.countdown_trials.append(self.countdown)
        self.rule_trial.append(self.rule)

    def store_data(self, data_dict: dict):
        super().store_data(data_dict)

        # Reset class instance.
        if type(self) == ProbTrialRev:
            self.__init__(self.p_reward, self.tones, self.actions,
                          self.countdown_start, self.rev_trials,
                          self.reward_history, self.initial)


class RevVarProb(ProbRevCount):
    '''
    A rule for a probabilistic reversal learning task similar to ProbRevCount
    rule. However, once the mouse reaches criterion, it goes through a series
    of changes in reward probability.
    '''

    def __init__(self, p_rew_list: list, tones: list, actions: list,
                 countdown_start: int, criterion: list,
                 reward_history: np.array = None, initial: int = None):

        self.p_rew_list = p_rew_list
        self.p_index = 0  # Start at 0 in the list.
        super().__init__(p_rew_list[self.p_index], tones, actions,
                         countdown_start, criterion, reward_history, initial)

        self.p_rew_trial = []

    def change_p_rew(self):
        '''
        Sets reward probability to self.p_rew_list[p_index], and maps outcomes
        accordingly.
        '''
        self.p_reward = self.p_rew_list[self.p_index]
        self.map_outcomes()

    def trial_end(self):
        '''
        Determines what will occurr at the end of the trial. If there isn't
        an ongoing trial countdown, will check the criterion. If there is a
        countdown, will subtract 1 and trigger a reversal if == 0.
        '''
        if np.isnan(self.countdown):
            if self.check_criterion():
                # Criterion met, countdown starts.
                self.countdown = self.countdown_start-1

        else:
            if self.countdown == 0:

                if self.p_index == len(self.p_rew_list)-1:
                    # Reached the end of prob sequence, time for reversal.
                    self.reversal()
                    self.p_index = 0
                    self.change_p_rew()
                    self.countdown = np.nan
                    self.performance_history = []

                else:
                    # Change reward probability to next in sequence.
                    self.p_index += 1
                    self.change_p_rew()
                    self.countdown = self.countdown_start-1

            else:
                self.countdown -= 1

        self.countdown_trials.append(self.countdown)
        self.rule_trial.append(self.rule)
        self.p_rew_trial.append(self.p_reward)

    def store_data(self, data_dict: dict):
        super().store_data(data_dict)
        data_dict['p_reward'] = np.array(self.p_rew_trial)

        # Reset class instance.
        if type(self) == RevVarProb:
            self.__init__(self.p_rew_list, self.tones, self.actions,
                          self.countdown_start, self.criterion,
                          self.reward_history, self.initial)


class TwoArmBandit(Rule):
    '''
    A task rule for a two-armed bandit task. Every trial, the agent
    chooses one of two possible actions, each with a hidden reward probability.
    the probabilities change with blocks of pre-defined lengths.
    '''
    def __init__(self, p_reward: float, actions: list,
                 block_length: int, reward_history: np.array = None,
                 initial: int = None):
        self.p_reward = p_reward
        self.actions = actions
        self.reward_history = reward_history
        self.initial = initial
        self.block_length = block_length

        self.rule_trial = []
        self.performance = []
        self.reward = []
        self.reversals = []

        if self.initial is None:
            self.rule = int(np.random.choice([0, 1]))
        else:
            self.rule = self.initial

        self.map_outcomes()

    def map_outcomes(self):
        '''
        Using the rule, outputs an array that indicates, for each tone-action
        pair, whether the trial would be considered correct. Also outputs an
        array with associated reward probabilities for each tone-action pair.

        If properly formatted, the array indices represent the following
        tone-action pairs: [lowF-L, lowF-R], [hiF-L, hiF-R].
        '''
        correct_p = self.p_reward
        incorrect_p = 1-self.p_reward
        null_p = 0

        if self.rule == 0:
            self.correct_array = np.array([[1, 0, 0]])
            self.reward_probs = np.array([[correct_p, incorrect_p, null_p]])

        elif self.rule == 1:
            self.correct_array = np.array([[0, 1, 0]])
            self.reward_probs = np.array([[incorrect_p, correct_p, null_p]])

    def outcome(self, state, action):
        '''
        Determine whether the mouse will be rewarded for its choice.
        '''
        correct = int(self.correct_array[state, action])

        if self.reward_history is not None:
            # Take the first value of the vector, then remove it.
            reward = self.reward_history[0]
            self.reward_history = self.reward_history[1:]
        else:
            reward = int(np.random.rand() < self.reward_probs[state, action])

        self.performance.append(correct)
        self.reward.append(reward)
        return correct, reward

    def trial_end(self):
        '''
        Determines what will occurr at the end of the trial.
        '''
        # Check to see whether a reversal should occur on this trial.
        if (len(self.reward) % self.block_length) == 0:
            # Change rule, remap outcomes and record reversal trial.
            self.rule = int(1-self.rule)
            self.map_outcomes()
            self.reversals.append(len(self.reward))
        # Store trial data.
        self.rule_trial.append(self.rule)

    def store_data(self, data_dict: dict):
        super().store_data(data_dict)
        data_dict['reversals'] = np.array(self.reversals)
        # Reset class instance.
        if type(self) == TwoArmBandit:
            self.__init__(self.p_reward, self.actions, self.block_length,
                          self.reward_history, self.initial)
