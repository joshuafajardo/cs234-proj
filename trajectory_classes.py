import numpy as np


class AnnotatedTrajectory:
  pass  # TODO


class Trajectory:
  def __init__(
      self,
      states: np.ndarray,
      actions: np.ndarray,
      rewards: np.ndarray,
      num_possible_states: int = 2,
      num_possible_actions: int = 2,
  ):
    self.trajectory_length = len(states)
    assert len(actions) == self.trajectory_length
    assert len(rewards) == self.trajectory_length

    self.states = states
    self.actions = actions
    self.rewards = rewards

    self.num_possible_states = num_possible_states
    self.num_possible_actions = num_possible_actions

  # STATUS: Lightly tested
  def create_nan_expanded_rewards(self):
    """
    Expands the rewards from shape (trajectory_length) to shape (num_actions,
    trajectory_length), where actions that were not taken have value nan.
    """
    output = np.full((self.num_possible_actions, self.trajectory_length), np.nan)
    # There's room for optimization down below, but it won't save much time.
    for timestep, _, action, reward in self:
      output[action, timestep] = reward
    return output

  def unpack(self):
    """
    Convenience function for retrieving the states, actions, and rewards as
    separate arrays.
    """
    return self.states, self.actions, self.rewards

  def __iter__(self):
    """
    Allows for iteration through the trajectory's timesteps, retrieving
    (timestep, state, action, reward) tuples for each timestep.
    """
    for i in range(self.trajectory_length):
      yield (i, self.states[i], self.actions[i], self.rewards[i])
  
  def __len__(self):
    return self.trajectory_length
