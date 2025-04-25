import random
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from math import e, pow

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of the current script
GUESSES_PATH = os.path.join(SCRIPT_DIR, "guesses.txt")
ANSWERS_PATH = os.path.join(SCRIPT_DIR, "answers.txt")

def load_words(filepath):
    """Loads words from a file, one word per line."""
    try:
        with open(filepath, 'r') as f:
            # Read lines, strip whitespace/newlines, convert to lowercase
            words = [line.strip().lower() for line in f if line.strip()]
        return words
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None # Indicate failure

class WordleGame:
    def __init__(self, guesses_path=GUESSES_PATH, answers_path=ANSWERS_PATH, word_length=5, max_attempts=6, silent=False): # Add silent flag
        """Initializes the Wordle game."""
        # Keep track if the game should run silently (useful for training)
        self.silent = silent
        if not self.silent:
            print(f"Loading guesses from: {guesses_path}")
            print(f"Loading answers from: {answers_path}")

        raw_guesses = load_words(guesses_path)
        raw_answers = load_words(answers_path)

        if raw_guesses is None or raw_answers is None:
            raise ValueError("Failed to load word lists. Please check file paths.")

        # Filter words by the expected length and create a set for fast lookups
        self.valid_guesses = {word for word in raw_guesses if len(word) == word_length}
        # Ensure answers are also valid guesses and meet length requirement
        self.possible_answers = [word for word in raw_answers if len(word) == word_length and word in self.valid_guesses]

        if not self.possible_answers:
             raise ValueError(f"No valid answers of length {word_length} found in {answers_path} that are also in {guesses_path}.")

        self.word_length = word_length
        self.max_attempts = max_attempts
        self.answer = ""
        self.attempts_left = 0
        self.guesses = []
        self.feedback = [] # Store feedback for each guess

    def start_new_game(self):
        """Selects a new random answer and resets game state."""
        self.answer = random.choice(self.possible_answers)
        self.attempts_left = self.max_attempts
        self.guesses = []
        self.feedback = []
        if not self.silent:
            print("\n--- New Wordle Game Started ---")
        # print(f"DEBUG: The answer is {self.answer}") # Uncomment for debugging

    def is_valid_guess(self, guess):
        """Checks if the guess is valid (correct length and in guesses list)."""
        guess = guess.lower()
        if len(guess) != self.word_length:
            if not self.silent:
                print(f"Invalid guess: Must be {self.word_length} letters long.")
            return False
        if guess not in self.valid_guesses:
            if not self.silent:
                print("Invalid guess: Not in word list.")
            return False
        return True

    def submit_guess(self, guess):
        """Processes a player's guess and provides feedback."""
        guess = guess.lower()
        # Note: is_valid_guess already handles printing if not silent
        if not self.is_valid_guess(guess):
            return False # Indicate invalid guess

        if self.attempts_left <= 0:
            if not self.silent:
                print("No attempts left!")
            return False

        self.attempts_left -= 1
        self.guesses.append(guess)

        # Generate feedback (e.g., Green, Yellow, Gray)
        current_feedback = self._generate_feedback(guess)
        self.feedback.append(current_feedback)

        # --- Silence this print during training ---
        # if not self.silent:
        #     print(f"Guess {len(self.guesses)}/{self.max_attempts}: {guess.upper()} -> Feedback: {' '.join(current_feedback)}")

        if guess == self.answer:
            if not self.silent:
                print(f"Congratulations! You guessed the word: {self.answer.upper()}")
            return True # Indicate game won
        elif self.attempts_left == 0:
            if not self.silent:
                print(f"Game Over! The word was: {self.answer.upper()}")
            return True # Indicate game ended (loss)

        return False # Indicate game continues

    def _generate_feedback(self, guess):
        """Generates Wordle feedback (G, Y, -) for a guess."""
        feedback = ['-'] * self.word_length  # Initialize with Gray ('-')
        answer_list = list(self.answer)      # Mutable list of answer letters
        guess_list = list(guess)

        # First pass: Check for Green matches (correct letter, correct position)
        for i in range(self.word_length):
            if guess_list[i] == answer_list[i]:
                feedback[i] = 'G'
                answer_list[i] = None  # Mark letter as used
                guess_list[i] = None   # Mark letter as used

        # Second pass: Check for Yellow matches (correct letter, wrong position)
        for i in range(self.word_length):
            if guess_list[i] is not None and guess_list[i] in answer_list:
                feedback[i] = 'Y'
                # Mark the first occurrence of this letter in the answer as used
                answer_list[answer_list.index(guess_list[i])] = None

        return feedback

# --- Gym-like Environment Wrapper ---

class WordleEnv(gym.Env): # Inherit from gym.Env
    metadata = {'render_modes': ['human'], 'render_fps': 1}

    def __init__(self, guesses_path=GUESSES_PATH, answers_path=ANSWERS_PATH, word_length=5, max_attempts=6, silent_game=True): # Add silent_game flag
        """
        Initializes the Wordle environment.
        """
        super().__init__() # Initialize gym.Env

        # Pass the silent flag to the underlying game
        self.game = WordleGame(guesses_path=guesses_path, answers_path=answers_path, word_length=word_length, max_attempts=max_attempts, silent=silent_game)

        # --- Reward Constants ---
        # self.BASE_WIN_REWARD = 20.0       # Base reward (now calculated exponentially)
        # self.WIN_STEP_PENALTY = 3.0       # Penalty subtracted (replaced by exponential)
        self.LOSS_PENALTY = -50.0         # HEAVILY Increased loss penalty
        self.REPEAT_GUESS_PENALTY = -20.0 # HEAVILY Increased repeat penalty
        self.STEP_PENALTY = -1.0          # Increased step penalty (can adjust if needed)
        self.GREEN_BONUS = 0.3            # Bonus for each green letter (keep or adjust)
        self.YELLOW_BONUS = 0.1           # Bonus for each yellow letter (keep or adjust)
        self.WIN_REWARD_BASE = 10.0     # Base for exponential reward (2)
        self.MAX_ATTEMPTS_FOR_REWARD = max_attempts # Use max_attempts in reward calc

        # --- Define Mappings ---
        # Map characters 'a'-'z' to 0-25, plus a padding token (26)
        self.char_to_int = {chr(ord('a') + i): i for i in range(26)}
        self.char_to_int['<PAD>'] = 26 # Padding character for observation
        self.int_to_char = {v: k for k, v in self.char_to_int.items()}
        self.vocab_size = len(self.char_to_int)

        # Map feedback symbols 'G', 'Y', '-' to 0, 1, 2, plus padding (3)
        self.feedback_to_int = {'G': 0, 'Y': 1, '-': 2, '<PAD>': 3}
        self.int_to_feedback = {v: k for k, v in self.feedback_to_int.items()}
        self.feedback_vocab_size = len(self.feedback_to_int)

        # --- Action Space ---
        # Discrete action space: index corresponding to a word in the valid guesses list
        self.action_list = sorted(list(self.game.valid_guesses)) # Ensure consistent order
        self.action_to_word = {i: word for i, word in enumerate(self.action_list)}
        self.word_to_action = {word: i for i, word in enumerate(self.action_list)}
        self.action_space = spaces.Discrete(len(self.action_list))

        # --- Observation Space ---
        # Represents the state: max_attempts x (word_length + word_length)
        # Each row: [char1_int, ..., char5_int, feedback1_int, ..., feedback5_int]
        # Padded with special tokens if fewer than max_attempts guesses made.
        self.observation_shape = (self.game.max_attempts, self.game.word_length * 2)
        # Low values are 0 (for char/feedback ints), high values are max int values
        # Using Box space for numerical representation
        # Dtype needs to accommodate the highest integer value (vocab_size-1 or feedback_vocab_size-1)
        max_obs_val = max(self.vocab_size - 1, self.feedback_vocab_size - 1)
        self.observation_space = spaces.Box(low=0, high=max_obs_val,
                                            shape=self.observation_shape, dtype=np.int32)

        # Remove print statements from __init__
        # print(f"Action Space Size: {len(self.action_list)}")
        # print(f"Observation Space Shape: {self.observation_shape}")


    def reset(self, seed=None, options=None):
        """
        Resets the environment for a new episode.

        Args:
            seed (int, optional): The seed for the random number generator. Defaults to None.
            options (dict, optional): Additional options (e.g., specifying an answer). Defaults to None.

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): The initial state observation.
                - info (dict): Auxiliary information (empty in this case).
        """
        # We need to reset the environment's random number generator (for answer selection)
        super().reset(seed=seed)

        answer = options.get("answer", None) if options else None
        self.game.start_new_game() # For now, always random
        if answer and answer in self.game.possible_answers:
             self.game.answer = answer # Override if a valid answer is provided

        # Keep track of guesses within the episode for penalty
        self._episode_guesses = set()

        observation = self._get_observation()
        info = self._get_info() # Get initial info (e.g., the answer for debugging/evaluation)

        # print(f"Resetting Env. Answer: {self.game.answer}") # Debug
        return observation, info

    def step(self, action):
        """
        Takes a step in the environment using an action index.

        Args:
            action (int): The index of the guess word in self.action_list.

        Returns:
            tuple: A tuple containing:
                - observation (np.ndarray): The agent's observation of the current environment.
                - reward (float): The amount of reward returned after previous action.
                - terminated (bool): Whether the agent reaches the terminal state (game won or lost).
                - truncated (bool): Whether the episode is truncated (not used here, could be for time limits).
                - info (dict): Contains auxiliary diagnostic information useful for debugging.
        """
        if not isinstance(action, (int, np.integer)):
             raise ValueError(f"Action must be an integer index. Received: {action} (type: {type(action)})")
        if action < 0 or action >= len(self.action_list):
             raise ValueError(f"Action index {action} is out of bounds (0-{len(self.action_list)-1})")

        guess = self.action_to_word[action]

        # --- Check for Repeated Guess ---
        is_repeat = guess in self._episode_guesses
        if is_repeat:
            # Penalize repetition, but still consume an attempt
            # Silence this print
            # print(f"Warning: Agent repeated guess '{guess}'.")
            self.game.attempts_left -= 1
            # Add guess to history even if repeated, to potentially penalize multiple repeats
            self._episode_guesses.add(guess)
            terminated = self.game.attempts_left <= 0
            reward = self.REPEAT_GUESS_PENALTY # Apply the heavy penalty
            observation = self._get_observation()
            info = self._get_info()
            info['warning'] = 'Repeated guess'
            if terminated:
                 # Optionally add loss penalty if attempts run out *due to* repeat,
                 # but the repeat penalty itself is already heavy.
                 # reward += self.LOSS_PENALTY # Consider if needed in addition to REPEAT_GUESS_PENALTY
                 if not self.game.silent: # Only print if game is not silent
                    print(f"Game Over! The word was: {self.game.answer.upper()}")
            return observation, reward, terminated, False, info

        # Add valid, non-repeated guess to episode history
        self._episode_guesses.add(guess)

        # Check if the chosen action (word) is actually a valid guess in the game context
        if not self.game.is_valid_guess(guess):
            # Silence this print
            # print(f"Warning: Agent chose an invalid guess '{guess}' mapped from action {action}.")
            self.game.attempts_left -= 1
            terminated = self.game.attempts_left <= 0 or (len(self.game.guesses) > 0 and self.game.guesses[-1] == self.game.answer)
            reward = 0.0 # No reward for invalid step
            observation = self._get_observation() # State remains the same, but attempts decrease
            info = self._get_info()
            info['error'] = 'Invalid guess submitted by agent'
            if terminated and (len(self.game.guesses) == 0 or self.game.guesses[-1] != self.game.answer):
                 print(f"Game Over! The word was: {self.game.answer.upper()}")
            return observation, reward, terminated, False, info # False for truncated

        # Submit the valid guess to the game logic
        game_over = self.game.submit_guess(guess) # Game's internal prints are now controlled by its 'silent' flag

        observation = self._get_observation()
        # Pass the latest feedback to the reward calculation
        latest_feedback = self.game.feedback[-1] if self.game.feedback else ['-'] * self.game.word_length
        reward = self._calculate_reward(game_over, latest_feedback)
        terminated = game_over # Game terminates if won or lost
        truncated = False # No truncation condition here
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """
        Constructs the numerical observation based on the current game state.
        Pads with special tokens up to max_attempts.
        """
        obs = np.full(self.observation_shape, self.char_to_int['<PAD>'], dtype=np.int32) # Initialize with padding

        for i in range(len(self.game.guesses)):
            guess = self.game.guesses[i]
            feedback = self.game.feedback[i]

            # Encode guess characters
            guess_ints = [self.char_to_int.get(char, self.char_to_int['<PAD>']) for char in guess]
            # Encode feedback symbols
            feedback_ints = [self.feedback_to_int.get(fb, self.feedback_to_int['<PAD>']) for fb in feedback]

            # Fill the observation row
            obs[i, :self.game.word_length] = guess_ints
            obs[i, self.game.word_length:] = feedback_ints

        return obs

    def _get_info(self):
        """ Returns auxiliary information (e.g., remaining possible words) """
        # Could add more info later, like the set of possible answers remaining
        return {"answer": self.game.answer, "attempts_left": self.game.attempts_left}


    def _calculate_reward(self, game_over, current_feedback):
        """
        Calculates the reward based on the game outcome and intermediate steps.
        Uses exponential reward for winning based on attempts.
        """
        guess = self.game.guesses[-1] if self.game.guesses else ""

        # Check for repeated guess (should ideally be handled in step, but double-check)
        # if len(self.game.guesses) > 1 and guess in self.game.guesses[:-1]:
        #     return self.REPEAT_GUESS_PENALTY # Apply penalty if repeated

        if game_over:
            if guess == self.game.answer:
                # Game won - Exponential reward based on attempts taken
                attempts_taken = len(self.game.guesses)
                # Reward = e^(max_attempts - attempts_taken), higher for fewer attempts
                # Ensure attempts_taken is at least 1
                reward_exponent = self.MAX_ATTEMPTS_FOR_REWARD - max(1, attempts_taken)
                win_reward = pow(self.WIN_REWARD_BASE, reward_exponent)
                # Add small bonus based on feedback just to differentiate slightly? Optional.
                # num_greens = current_feedback.count('G')
                # win_reward += num_greens * 0.01 # Tiny bonus for greens on final step
                return win_reward
            else:
                # Game lost (ran out of attempts)
                return self.LOSS_PENALTY # Apply the heavy loss penalty
        else:
            # Intermediate step - reward based on feedback + step penalty
            num_greens = current_feedback.count('G')
            num_yellows = current_feedback.count('Y')
            intermediate_reward = (self.STEP_PENALTY +
                                   num_greens * self.GREEN_BONUS +
                                   num_yellows * self.YELLOW_BONUS)
            return intermediate_reward

    def render(self):
        """ Renders the environment state only if not silent """
        if self.silent_game: # Use self.silent_game
             return # Do nothing if silent
        if self.metadata['render_modes'][0] == 'human': # Check render mode
            print("--- Current Game State ---")
            obs_numeric = self._get_observation()
            for i in range(self.game.max_attempts):
                guess_ints = obs_numeric[i, :self.game.word_length]
                feedback_ints = obs_numeric[i, self.game.word_length:]

                # Check if the row is padding
                if np.all(guess_ints == self.char_to_int['<PAD>']):
                    print(f"Attempt {i+1}: -")
                    continue

                guess_str = "".join([self.int_to_char.get(g_int, '?') for g_int in guess_ints]).upper()
                feedback_str = " ".join([self.int_to_feedback.get(f_int, '?') for f_int in feedback_ints])

                print(f"Attempt {i+1}: {guess_str} -> {feedback_str}")

            print(f"Attempts left: {self.game.attempts_left}")
            print(f"Answer (Debug): {self.game.answer}") # Keep for debugging
            print("-------------------------")
        else:
             # Handle other render modes if needed, or just return
             return


# --- Basic Game Loop Example (Keep for manual play testing) ---
if __name__ == "__main__":
    try:
        # Keep the original game loop for direct play (non-silent)
        game = WordleGame(silent=False) # Ensure it's not silent when run directly
        game.start_new_game()

        while game.attempts_left > 0:
            player_guess = input(f"Enter your guess ({game.attempts_left} attempts left): ")
            game_over = game.submit_guess(player_guess)
            if game_over:
                play_again = input("Play again? (y/n): ").lower()
                if play_again == 'y':
                    game.start_new_game()
                else:
                    break

    except ValueError as e:
        print(f"Error initializing game: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    # --- Example Env Usage (Updated) ---
    print("\n--- Testing Environment ---")
    try:
        # Test with silent_game=False to see outputs
        env = WordleEnv(silent_game=False)
        # Optional: Check the spaces
        print("Action Space:", env.action_space)
        print("Observation Space:", env.observation_space)

        # Test reset
        observation, info = env.reset(seed=42) # Use a seed for reproducibility
        print("Initial Observation (shape):", observation.shape)
        # print("Initial Observation (values):\n", observation)
        print("Initial Info:", info)
        env.render()

        # Test step with a random valid action
        terminated = False
        truncated = False
        total_reward = 0
        step_count = 0

        while not terminated and not truncated:
            action = env.action_space.sample() # Sample a random action (index)
            word_guess = env.action_to_word[action]
            print(f"\nStep {step_count+1}: Agent chooses action {action} ('{word_guess.upper()}')")

            observation, reward, terminated, truncated, info = env.step(action)

            print("Observation (shape):", observation.shape)
            # print("Observation (values):\n", observation) # Can be large
            print("Reward:", reward)
            print("Terminated:", terminated)
            print("Truncated:", truncated)
            print("Info:", info)
            env.render()

            total_reward += reward
            step_count += 1

            # Safety break
            if step_count > env.game.max_attempts + 2:
                 print("Error: Exceeded max attempts in test loop.")
                 break


        print("\n--- Environment Test Finished ---")
        print(f"Total Steps: {step_count}")
        print(f"Final Reward: {total_reward}")
        if terminated and reward > 0:
            print("Outcome: Game Won")
        elif terminated:
            print("Outcome: Game Lost")
        else:
            print("Outcome: Game Truncated (or error)")


    except ValueError as e:
        print(f"Error initializing or running environment: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error occurred during env test: {e}")
        import traceback
        traceback.print_exc()
