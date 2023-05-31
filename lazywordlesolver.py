import gym
import gym_wordle
import numpy as np
import pandas as pd

from gym_wordle.utils import to_english, to_array

A_TO_Z = [chr(i) for i in range(ord('a'), ord('z') + 1)]
EPSILON = 1e-8  # Suppress divide by zero errors for log functions.
NUM_CHARS = 5

class WordleSolver:
    def __init__(self, env, debug=False):
        self._debug = debug
        self.env = env
        data = [{
            'word': to_english(w),
            'word_index': w,
            'active': True,
            'entropy': 0,
            } for w in self.env.action_space]
        self.guesses = pd.DataFrame(data=data)
        # Get a count of chars.
        for char in A_TO_Z:
            self.guesses[char] = self.guesses['word'].str.count(char)
        # Store the char in each position.
        self.guesses = self.guesses.merge(
            # List splits the word; the apply places each char in
            # it's own numbered column, from 0 to (NUM_CHARS-1).
            # https://stackoverflow.com/questions/43848680/pandas-split-dataframe-column-for-every-character
            self.guesses['word'].apply(lambda w: pd.Series(list(w))),
            left_index=True,
            right_index=True)
 
    def update_from_state(self, state):
        '''
        State is a 2D numpy array containing the game state.
        Here's an example, where the solution word is "prune":
        [
            [3, 2, 3, 3, 1, 0, 0, 0, 0, 0],  # Guess 1: fesse
            [3, 3, 1, 2, 3, 0, 0, 0, 0, 0],  # Guess 2: glues
            [3, 3, 3, 2, 3, 0, 0, 0, 0, 0],  # Guess 3: tajes
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Guess 4: not yet taken
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Guess 5: not yet taken
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Guess 6: not yet taken
        ]

        Numbers in the state array have the following meanings:
            - 0 means unguessed, or extra letters past word size
            - 1 means correct letter, correct spot
            - 2 means correct letter, incorrect spot
            - 3 means letter not in solution

        '''
        newest_state = state[self.env.round - 1]
        chars_in_solution = {c: 0 for c in A_TO_Z}
        for i, case in enumerate(newest_state):
            if i >= NUM_CHARS: break
            # Convert from number to letter of the alphabet.
            char = self.current_guess_english[i]
            if case == 1:  # Right letter, right spot
                self.guesses.loc[self.guesses[i] != char, 'active'] = False
                chars_in_solution[char] += 1
            elif case == 2:  # Right letter, wrong spot
                self.guesses.loc[(self.guesses[i] == char) | (self.guesses[char] == 0), 'active'] = False
                chars_in_solution[char] += 1
            elif case == 3:  # Wrong letter
                # If this char is in solution, only remove it from the
                # guessed position.
                if chars_in_solution[char] > 0:
                    self.guesses.loc[self.guesses[i] == char, 'active'] = False
                # Otherwise, remove it from all positions.
                else:
                    for j in range(NUM_CHARS):
                        self.guesses.loc[self.guesses[j] == char, 'active'] = False
            else:  # case == 4
                continue

    def calc_entropies(self):
        # Entropy is calculated using Shannon entropy, based on the 
        # frequency of a letter in each of the five char positions.
        # Start by getting counts of letters in each position.
        letter_probabilities = {}
        # Only count words that are still active.
        num_words = self.guesses[self.guesses['active']].shape[0]
        if self._debug: print(f'\tnum_words remaining:\t{num_words}')
        # If there's zero or one words left, we don't need to
        # calculate anything.
        if num_words <= 1: return
        for c in A_TO_Z:
            letter_probabilities[c] = {j: 0 for j in range(NUM_CHARS)}
        for _, row in self.guesses.iterrows():
            if not row['active']: continue
            for i, char in enumerate(row['word']):
                letter_probabilities[char][i] += 1
        # Normalize counts to probabilities.
        for char in letter_probabilities:
            for i in range(NUM_CHARS):
                letter_probabilities[char][i] /= num_words
        # Then, find the entropy of each guessable word.
        for j, row in self.guesses.iterrows():
            w = row['word']
            for i, char in enumerate(w):
                p = max(letter_probabilities[char][i], EPSILON)
                # https://en.wikipedia.org/wiki/Entropy_(information_theory)
                self.guesses.loc[j, 'entropy'] -= p * np.log2(p)

    def get_next_guess(self, state=None):
        # TODO: this is slow; vectorize where possible.
        if self._debug:
            print(f'Starting guess: {self.env.round + 1}')
        if state is not None and len(state) > 0:
            self.update_from_state(state)
        self.calc_entropies()
        # TODO: should I use entropy differently?
        # Take the word with max entropy among active words. If there
        # are several with the same (max) entropy, just grab the first.
        max_entropy = self.guesses.loc[self.guesses['active'], 'entropy'].max()
        max_entropy_row = self.guesses[(self.guesses['active']) & (self.guesses['entropy'] == max_entropy)].head(1)
        next_word = max_entropy_row.iloc[0]['word']
        next_word_index = max_entropy_row.iloc[0]['word_index']
        # Mark the word as no longer active and update other values.
        self.guesses.loc[self.guesses['word'] == next_word, 'active'] = False
        self.current_guess_english = next_word
        self.current_guess = self.env.action_space.index_of(next_word_index)
        return self.current_guess
    
    def guess_blind(self, filter_known_solutions=False):
        # Make a guess based on provided info, w/o having the solution.
        print('''
        This is an interactive mode using the solver\'s
        guesses, but with the user typing them into a real
        Wordle client.
        For each guess, type the word into Wordle and type the
        results back here:
        1 = green (right letter, right spot)
        2 = yellow (right letter, wrong spot)
        3 = grey (wrong letter)
        For example, a guess of "stone" for a solution of "close"
        would type "23131".
        ''')
        # Deactivate all solution words, since we assume there are never
        # repeats. This is not always true since the NYT started
        # curating solutions; use at your own risk.
        if filter_known_solutions:
            for word_array in self.env.solution_space.words:
                word = to_english(word_array)
                self.guesses.loc[self.guesses['word'] == word, 'active'] = False
        state = []
        for i in range(6):
            # Set the env round to smooth things out for the guessing
            # logic.
            self.env.round = i
            self.get_next_guess(state)
            print(f'\tGuess {i}: {self.current_guess_english}')
            new_state = [int(char) for char in input('\tResults:')]
            # All ones means we got it.
            if new_state == [1] * 5: break
            state.append(new_state)

    def test_run(self, solution=None):
        # Start a test run with a given solution.
        # solution is the desired solution word as a string.
        self.env.reset()
        if solution is not None:
            index = self.env.solution_space.index_of(to_array(solution))
            if index != -1:
                self.env.solution = index
        if self._debug:
            print(f'solution:\t{to_english(self.env.solution_space[self.env.solution])}')
        done = False
        state = None
        total_reward = 0
        guesses = []
        while not done:
            action = self.get_next_guess(state)
            guesses.append(to_english(self.env.action_space[action]))
            if self._debug:
                print(f'\tguess:\t{action} ({to_english(self.env.action_space[action])})')
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        return guesses, self.env.solution, total_reward

if __name__ == '__main__':
    env = gym.make("Wordle-v0")
    agent = WordleSolver(env, debug=True)
    # guesses, solution, reward = agent.test_run('zebra')
    # print(guesses, solution, reward)
    agent.guess_blind()
