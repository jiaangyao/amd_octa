import contextlib
import numpy as np


@ contextlib.contextmanager
def temp_seed(seed):
    """
    This function allows temporary creating a numpy random number generator state and is used to ensure that
    splitting the data can be performed with the same random seed 20194040 while the rest of the script is not affected
    by that random state

    :param seed: Desired random seed to be used
    """

    # Obtain the old random seed
    state = np.random.get_state()

    # Set the np random seed in the current environment to the desired seed number
    np.random.seed(seed)

    try:
        yield
    finally:
        # Reset the seed when the function is not called
        np.random.set_state(state)