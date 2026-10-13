import numpy as np
import pandas as pd
import os


class QLearner():

    def __init__(self, num_states, num_actions, learning_rate, discount_factor, save_dir):
        self.q_table = np.full((num_states, num_actions), 0.0)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.save_dir = save_dir

    def load(self, path):
        self.q_table = np.load(path)

    def set_entry(self, key, value):
        s_i, a_i = key
        self.q_table[s_i, a_i] = value

    def _make_df(self, all_actions, all_states):
        return pd.DataFrame({rule: self.q_table[:, i] for i, rule in enumerate(all_actions)},
                            index=[str(state.state_name) for state in all_states])

    def print_q_table(self, logger, all_actions, all_states):
        q_df = self._make_df(all_actions, all_states)
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

    def save(self, suffix, all_actions, all_states):
        path = f"{self.save_dir}{os.sep}q_table_lr{self.learning_rate}_df{self.discount_factor}_{suffix}"
        np.save(path, self.q_table)

        q_df = self._make_df(all_actions, all_states)
        q_df.to_csv(path, sep=";")
        try:
            os.remove("latest_table.npy")
        except FileNotFoundError:
            pass
        os.symlink(src=path+".npy", dst="latest_table.npy")
        return path

    def exploit(self, state) -> np.intp:
        return np.argmax(self.q_table[state])
