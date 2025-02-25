from dataclasses import dataclass
import numpy as np

@dataclass
class AnnotatedTrajectory:
  pass  # TODO

class Trajectory:
  def __init__(
      self,
      states: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
  ):
    self.num_timesteps = len(states)
    assert len(actions) == self.num_timesteps
    assert len(rewards) == self.num_timesteps

    self.states = states
    self.actions = actions
    self.rewards = rewards

  def unpack(self):
    """
    Convenience function for retrieving the states, actions, and rewards as
    separate arrays.
    """
    return self.states, self.actions, self.rewards

  def __iter__(self):
    """
    Allows for iteration through the trajectory's timesteps, retrieving
    (state, action, reward) tuples for each timestep.
    """
    for i in range(self.num_timesteps):
      yield (self.states[i], self.actions[i], self.rewards[i])
  
  def __len__(self):
    return self.num_timesteps
