class Tone:
    '''
    A tone is the cue presented to the agent that indicates the state of the 
    environment.
    '''
    def __init__(self, freq: int, index: int):
        self.freq = freq
        self.index = index

    def __str__(self) -> str:
        return str(self.freq)

    def __int__(self) -> int:
        return int(self.index)


class Action:
    '''
    An action represents a choice that's available to the mouse on each trial.
    '''
    def __init__(self, name: str, index: int):
        self.name = name
        self.index = index

    def __str__(self) -> str:
        return str(self.name)

    def __int__(self) -> int:
        return int(self.index)
