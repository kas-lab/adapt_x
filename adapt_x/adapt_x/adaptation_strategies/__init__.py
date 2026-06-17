import inspect
from adapt_x.adaptation_strategies.qlearning_strategy import QLearningStrategy


def create_strategy(strategy_name, **kwargs):
    strategies = {
        "q-learning": QLearningStrategy,
    }

    chosen_strategy = strategies.get(strategy_name, None)

    if (chosen_strategy is not None):
        # Get the constructor signature
        sig = inspect.signature(chosen_strategy.__init__)
        # Extract parameter names, excluding 'self'
        required_args = [param for param in sig.parameters.keys() if param != 'self']

        # Check if all required parameters are in kwargs
        missing_args = [param for param in required_args if param not in kwargs]

        if missing_args:
            raise RuntimeError(
                f"Missing required arguments for {strategy_name}: {', '.join(missing_args)}"
            )

        return chosen_strategy(**kwargs)
    else:
        raise RuntimeError("Specified Strategy " + strategy_name + " did not exist")
