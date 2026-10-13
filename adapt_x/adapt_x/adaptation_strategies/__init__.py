import inspect
from adapt_x.adaptation_strategies.qlearning_strategy import QLearningStrategy
from adapt_x.adaptation_strategies.tactic_strategy import TacticStrategy
from adapt_x.adaptation_strategies.cbr_tactic_strategy import CBRTacticStrategy
from adapt_x.adaptation_strategies.constant_strategy import ConstantStrategy


def create_strategy(strategy_name, **kwargs):
    strategies = {
        "q-learning": QLearningStrategy,
        "tactic": TacticStrategy,
        "constant": ConstantStrategy,
        "cbr_tactic": CBRTacticStrategy
    }
    kwargs["strategy_name"] = strategy_name
    chosen_strategy = strategies.get(strategy_name, None)

    if chosen_strategy is not None:
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

        # Filter kwargs to only include parameters that the strategy accepts
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in required_args}

        return chosen_strategy(**filtered_kwargs)

    raise RuntimeError("Specified Strategy " + strategy_name + " did not exist")
