from dataclasses import dataclass
import numpy as np

@dataclass
class AnnotatedTrajectory:
  pass  # TODO


@dataclass
class Trajectory:
  states: np.ndarray
  actions: np.ndarray
  rewards: np.ndarray
