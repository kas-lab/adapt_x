import numpy as np
import pandas as pd


class QLearner():
    def __init__(self, num_states, num_actions, learning_rate, discount_factor):
        self.q_table = np.full((num_states, num_actions), 0.0)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def set_entry(self, key, value):
        s_i, a_i = key
        self.q_table[s_i, a_i] = value

    def print_q_table(self, logger, all_actions, all_states):
        q_df = pd.DataFrame({rule: self.q_table[:, i] for i, rule in enumerate(all_actions)},
                            index=[str(state.state_name) for state in all_states])
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            logger.info(f"\n{str(q_df.transpose())}")

    def update_q_table(self, s_index: int, action_index: int, s_prime_index: int, reward: float):
        Q_current = self.q_table[s_index, action_index]  # Q(S,A)
        Qs_next_state = self.q_table[s_prime_index]  # Q(S')
        discounted_optimal_next_state_Q = self.discount_factor * \
            np.max(Qs_next_state)  # γ * max_a Q(S',a)

        # Q(S,A) <- Q(S,A) + 𝛂[R + γmax_a Q(S',a) - Q(s,a)]
        self.q_table[s_index, action_index] = Q_current + self.learning_rate * \
            (reward + discounted_optimal_next_state_Q - Q_current)

        return f"s_index: {s_index} action_index: {action_index} | {Q_current} + {self.learning_rate} * ({reward} + {discounted_optimal_next_state_Q} - {Q_current})"

    def exploit(self, state) -> np.intp:
        return np.argmax(self.q_table[state])
