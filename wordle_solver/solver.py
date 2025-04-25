import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import gymnasium as gym
from game import WordleEnv
from collections import deque # For storing episode data
from tqdm import tqdm # Import tqdm
import os # For saving plots and checking file existence
import matplotlib.pyplot as plt # Import matplotlib
import string # Import string for alphabet

# --- Configuration ---
ENV_CONFIG = {
    # "guesses_path": "guesses.txt", # Assuming they are in the same dir now
    # "answers_path": "answers.txt",
    "word_length": 5,
    "max_attempts": 6,
    "silent_game": True # Make the game silent during training
}
LEARNING_RATE = 0.00005 # Slightly lower LR might be needed with new rewards
GAMMA = 0.99
ENTROPY_COEFF = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EPISODES = 5000 # Might need more episodes
PRINT_EVERY = 500 # Report stats more often
MAX_EPISODE_LENGTH = ENV_CONFIG["max_attempts"] # No need for +1 if handled correctly
MODEL_SAVE_PATH = "wordle_solver_model.pth"
PLOT_SAVE_DIR = "plots" # Directory to save plots
EVAL_EPISODES = 500

WORD_LENGTH = ENV_CONFIG["word_length"] # Global for clarity
ALPHABET_SIZE = len(string.ascii_lowercase) # Global for clarity
INTERMEDIATE_OUTPUT_SIZE = WORD_LENGTH * ALPHABET_SIZE # Global for clarity

# Ensure plot directory exists
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Remove initial device print, will be clear from tqdm/logs
# print(f"Using device: {DEVICE}")

# --- Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, observation_shape, action_space_size): # action_space_size is not directly used for output layer size anymore
        """
        MLP Policy Network outputting intermediate representation.

        Args:
            observation_shape (tuple): Shape of the environment observation (e.g., (6, 10)).
            action_space_size (int): Number of possible actions (used for input size calculation if needed, but not output).
        """
        super(PolicyNetwork, self).__init__()
        input_size = np.prod(observation_shape)
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        # Change the final layer to output the intermediate size (130)
        self.fc_intermediate = nn.Linear(64, INTERMEDIATE_OUTPUT_SIZE) # Use global INTERMEDIATE_OUTPUT_SIZE

    # Keep only this forward method
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input observation tensor. Should be shape (batch_size, *observation_shape).

        Returns:
            torch.Tensor: Intermediate feature representation (shape: batch_size, INTERMEDIATE_OUTPUT_SIZE).
        """
        # Flatten the observation if it's not already flat
        # Handle potential batch dimension
        if x.dim() > 2:
             batch_size = x.size(0)
             x = x.view(batch_size, -1) # Reshape to (batch_size, input_size)
        else:
             # Assume batch size of 1 if 2D input (e.g., during single step inference)
             # Or if input is already flattened (e.g., during training batch)
             if x.dim() == 1:
                 x = x.unsqueeze(0) # Add batch dimension if completely flat
             elif x.dim() == 2 and x.size(0) != 1: # Already batched and flattened
                 pass # Use as is
             else: # Single item, not flattened
                 x = x.view(1, -1)


        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Return the intermediate features from the new final layer
        intermediate_output = self.fc_intermediate(x)
        return intermediate_output

    # Ensure the duplicate forward method below this line is REMOVED
    # def forward(self, x):
    #     """
    #     Forward pass through the network.
    #
    #     Args:
    #         x (torch.Tensor): Input observation tensor. Should be shape (batch_size, *observation_shape).
    #
    #     Returns:
    #         torch.Tensor: Logits for the action probability distribution (shape: batch_size, action_space_size).
    #     """
    #     # ... (code of the second forward method to be deleted) ...
    #     action_logits = self.fc_policy(x) # This line should be removed along with the method
    #     return action_logits # This line should be removed along with the method

# --- Agent ---
class ReinforceAgent:
    # Add env to init to access word lists and mappings
    def __init__(self, env, observation_shape, action_space_size, lr=LEARNING_RATE, gamma=GAMMA, entropy_coeff=ENTROPY_COEFF, device=DEVICE):
        """
        REINFORCE Agent with Entropy Regularization and Action Masking.

        Args:
            env (WordleEnv): The environment instance.
            observation_shape (tuple): Shape of the environment observation.
            action_space_size (int): Number of possible actions.
            lr (float): Learning rate.
            gamma (float): Discount factor.
            entropy_coeff (float): Coefficient for entropy bonus.
            device (torch.device): CPU or CUDA.
        """
        self.env = env # Store env reference
        # Pass action_space_size to PolicyNetwork, though it's not used for the final layer now
        self.policy_net = PolicyNetwork(observation_shape, action_space_size).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.device = device

        # --- Precompute Word Encoding Matrix ---
        self.action_space_size = action_space_size
        self.alphabet = string.ascii_lowercase
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self.word_encoding_matrix_t = self._create_word_encoding_matrix_t().to(self.device)
        # --- End Word Encoding ---

        # Storage for episode data
        self.log_probs = []
        self.rewards = []
        self.entropies = []

    def _create_word_encoding_matrix_t(self):
        """ Creates the one-hot word encoding matrix and transposes it. """
        print("Creating word encoding matrix...")
        # Matrix shape: (action_space_size, INTERMEDIATE_OUTPUT_SIZE)
        # Now uses the globally defined INTERMEDIATE_OUTPUT_SIZE
        encoding_matrix = torch.zeros((self.action_space_size, INTERMEDIATE_OUTPUT_SIZE), dtype=torch.float32)
        for action_index, word in self.env.action_to_word.items():
            if len(word) != WORD_LENGTH: # Use global WORD_LENGTH
                print(f"Warning: Word '{word}' at index {action_index} has incorrect length. Skipping.")
                continue
            word_vec = torch.zeros(INTERMEDIATE_OUTPUT_SIZE, dtype=torch.float32) # Use global INTERMEDIATE_OUTPUT_SIZE
            for pos, char in enumerate(word):
                if char in self.char_to_idx:
                    char_idx = self.char_to_idx[char]
                    encoding_idx = pos * ALPHABET_SIZE + char_idx # Use global ALPHABET_SIZE
                    if 0 <= encoding_idx < INTERMEDIATE_OUTPUT_SIZE: # Use global INTERMEDIATE_OUTPUT_SIZE
                        word_vec[encoding_idx] = 1.0
                    else:
                         print(f"Warning: Calculated index {encoding_idx} out of bounds for word '{word}'.")
                else:
                    print(f"Warning: Character '{char}' not in alphabet for word '{word}'.")
            encoding_matrix[action_index] = word_vec
        print("Word encoding matrix created.")
        # Transpose for efficient matmul: (INTERMEDIATE_OUTPUT_SIZE, action_space_size)
        return encoding_matrix.t()


    def _get_valid_actions_mask(self, observation):
        """
        Generates a mask for valid actions based on Wordle rules and current observation.

        Args:
            observation (np.ndarray): Current environment observation.

        Returns:
            torch.Tensor: A boolean mask of shape (action_space_size,) where True indicates a valid action.
        """
        mask = torch.ones(self.env.action_space.n, dtype=torch.bool, device=self.device) # Start with all actions valid

        known_greens = {} # {position: char}
        known_yellows = {} # {position: set(chars)} - chars known to be yellow at this position
        required_chars = {} # {char: min_count}
        misplaced_chars = set() # Chars known to be yellow somewhere
        banned_chars = set() # Chars known to be grey

        # Decode the observation to extract constraints
        for i in range(self.env.game.max_attempts):
            guess_ints = observation[i, :self.env.game.word_length]
            feedback_ints = observation[i, self.env.game.word_length:]

            # Stop if we hit padding
            if np.all(guess_ints == self.env.char_to_int['<PAD>']):
                break

            guess_chars = [self.env.int_to_char.get(g_int) for g_int in guess_ints]
            feedback_chars = [self.env.int_to_feedback.get(f_int) for f_int in feedback_ints]

            current_guess_required = {} # Track required chars based *only* on this guess's G/Y feedback

            for pos, (guess_char, fb_char) in enumerate(zip(guess_chars, feedback_chars)):
                if guess_char is None or fb_char is None: continue # Skip padding chars if any

                if fb_char == 'G':
                    # If green contradicts a previous green, something is wrong (shouldn't happen)
                    if pos in known_greens and known_greens[pos] != guess_char:
                         print(f"Warning: Contradictory green feedback for pos {pos}")
                         return torch.zeros_like(mask) # Invalidate all actions if state is contradictory
                    known_greens[pos] = guess_char
                    current_guess_required[guess_char] = current_guess_required.get(guess_char, 0) + 1
                elif fb_char == 'Y':
                    if pos not in known_yellows:
                        known_yellows[pos] = set()
                    known_yellows[pos].add(guess_char) # Cannot be this char at this position
                    misplaced_chars.add(guess_char)
                    current_guess_required[guess_char] = current_guess_required.get(guess_char, 0) + 1
                elif fb_char == '-':
                    # Only ban if not required by a G/Y in the *same* guess (handles double letters)
                    if guess_char not in current_guess_required:
                         banned_chars.add(guess_char)

            # Update overall required character counts based on this guess
            for char, count in current_guess_required.items():
                 required_chars[char] = max(required_chars.get(char, 0), count)


        # Filter the action list (words) based on constraints
        for action_index, word in self.env.action_to_word.items():
            valid = True
            word_char_counts = {c: word.count(c) for c in set(word)}

            # 1. Check Greens: Must match known green letters at specific positions
            for pos, green_char in known_greens.items():
                if word[pos] != green_char:
                    valid = False
                    break
            if not valid:
                mask[action_index] = False
                continue

            # 2. Check Yellows: Cannot have known yellow letters at those specific positions
            for pos, yellow_chars_at_pos in known_yellows.items():
                if word[pos] in yellow_chars_at_pos:
                    valid = False
                    break
            if not valid:
                mask[action_index] = False
                continue

            # 3. Check Required Chars (Greens + Yellows): Must contain at least the required count
            for req_char, min_count in required_chars.items():
                 # Ensure the word contains the required char with at least the minimum frequency derived from G/Y feedback
                 if word_char_counts.get(req_char, 0) < min_count:
                     valid = False
                     break
            if not valid:
                mask[action_index] = False
                continue

            # 4. Check Banned Chars: Cannot contain letters known to be grey
            #    (Unless that letter is also required by a Green/Yellow constraint)
            for banned_char in banned_chars:
                 if banned_char in word and banned_char not in required_chars:
                     valid = False
                     break
            if not valid:
                mask[action_index] = False
                continue

            # If all checks pass, the action remains valid (mask[action_index] is True)

        # Ensure at least one action is valid if possible, otherwise allow all (shouldn't happen in Wordle)
        if not mask.any():
             # print("Warning: No valid actions found based on constraints. Allowing all actions.") # Keep for debugging?
             return torch.ones_like(mask) # Fallback

        return mask


    def select_action(self, observation):
        """
        Selects an action using the new network structure and word encodings.
        """
        obs_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        # Get intermediate output from the network (shape: 1, 130)
        intermediate_output = self.policy_net(obs_tensor)

        # Calculate final action logits via dot product (matrix multiplication)
        # (1, 130) @ (130, action_space_size) -> (1, action_space_size)
        action_logits = torch.matmul(intermediate_output, self.word_encoding_matrix_t)

        # --- Action Masking (applied to final logits) ---
        valid_actions_mask = self._get_valid_actions_mask(observation)
        masked_logits = action_logits.masked_fill(~valid_actions_mask, -float('inf'))

        # --- Debugging/Robustness Checks (same as before) ---
        if not torch.isfinite(masked_logits).any():
            print("Warning: All actions masked to -inf in select_action! Falling back to uniform distribution over valid actions (if any) or original logits.")
            # Option 1: Fallback to uniform distribution over originally valid actions
            if valid_actions_mask.any():
                 num_valid = valid_actions_mask.sum().item()
                 valid_probs = torch.zeros_like(action_logits)
                 valid_probs[valid_actions_mask] = 1.0 / num_valid
                 action_probs = valid_probs
            # Option 2: Fallback to original logits if no actions were valid initially (should not happen with mask fallback)
            else:
                 action_probs = F.softmax(action_logits, dim=-1) # Use original logits
        else:
            # Calculate probabilities using softmax
            action_probs = F.softmax(masked_logits, dim=-1)

        if torch.isnan(action_probs).any():
            print(f"Warning: NaN detected in action probabilities! Logits: {action_logits.detach()}, Masked Logits: {masked_logits.detach()}")
            # Fallback strategy: Use uniform distribution over valid actions or original logits
            if valid_actions_mask.any():
                 print("Falling back to uniform distribution over valid actions.")
                 num_valid = valid_actions_mask.sum().item()
                 valid_probs = torch.zeros_like(action_logits)
                 valid_probs[valid_actions_mask] = 1.0 / num_valid
                 action_probs = valid_probs
            else:
                 # If somehow no actions are valid AND probs are NaN, use original logits
                 print("Falling back to softmax over original logits.")
                 action_probs = F.softmax(action_logits, dim=-1)
                 # If original logits also lead to NaN, something is fundamentally wrong (e.g., network outputting NaN)
                 if torch.isnan(action_probs).any():
                     print("ERROR: Original logits also resulted in NaN probabilities. Cannot sample action.")
                     # Return a default action or raise an error
                     # Returning a dummy action index 0 with zero log_prob/entropy
                     return 0, torch.tensor(0.0, device=self.device), torch.tensor(0.0, device=self.device)


        # Create distribution and sample
        dist = Categorical(probs=action_probs)
        try:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        except RuntimeError as e:
            # This handles potential sampling errors if probs are still problematic despite checks
            print(f"RuntimeError during sampling: {e}. Falling back to random valid action.")
            valid_indices = torch.where(valid_actions_mask)[0]
            if len(valid_indices) > 0:
                # Sample randomly from valid actions
                action_idx = valid_indices[torch.randint(len(valid_indices), (1,))].item()
                action = torch.tensor(action_idx, device=self.device)
                # Recalculate distribution with only valid actions (or use uniform)
                valid_probs = torch.zeros_like(action_probs)
                valid_probs[valid_actions_mask] = 1.0 / valid_actions_mask.sum().item()
                dist = Categorical(probs=valid_probs)
                log_prob = dist.log_prob(action) # Log prob might be inaccurate here, but better than crashing
                entropy = dist.entropy()
            else:
                # If no valid actions, fall back to original distribution (less likely now)
                print("No valid actions found during RuntimeError fallback. Using original logits.")
                dist = Categorical(logits=action_logits) # Use original logits here
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()


        return action.item(), log_prob, entropy

    def store_outcome(self, log_prob, reward, entropy):
        """ Stores the log probability, reward, and entropy for a step. """
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.entropies.append(entropy) # Store entropy

    def update(self):
        """ Performs the REINFORCE update with entropy regularization. """
        if not self.rewards:
            return 0.0

        T = len(self.rewards)
        returns = deque(maxlen=T)
        discounted_return = 0.0

        for r in self.rewards[::-1]:
            discounted_return = r + self.gamma * discounted_return
            returns.appendleft(discounted_return)

        returns = torch.tensor(list(returns), dtype=torch.float32).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        valid_log_probs = [lp for lp in self.log_probs if isinstance(lp, torch.Tensor)]
        valid_entropies = [ent for ent in self.entropies if isinstance(ent, torch.Tensor)] # Get valid entropies

        # Silence warnings during update if needed
        if not valid_log_probs or len(valid_log_probs) != len(valid_entropies):
             # print("Warning: No valid log probabilities or entropy mismatch found for update.")
             self._clear_memory()
             return 0.0, 0.0 # Return loss and policy loss

        # --- Add these lines back ---
        log_probs_tensor = torch.stack(valid_log_probs).to(self.device)
        entropies_tensor = torch.stack(valid_entropies).to(self.device)
        # --- End of added lines ---


        if log_probs_tensor.shape[0] != returns.shape[0]:
             # print(f"Warning: Mismatch between log_probs ({log_probs_tensor.shape[0]}) and returns ({returns.shape[0]}). Skipping update.")
             self._clear_memory()
             return 0.0, 0.0 # Return loss and policy loss

        # Calculate policy loss: -log_prob * Gt
        policy_loss = (-log_probs_tensor * returns).sum()

        # Calculate entropy bonus: -coeff * entropy (we want to maximize entropy, so minimize negative entropy)
        entropy_bonus = (-self.entropy_coeff * entropies_tensor).sum()

        # Total loss = policy loss + entropy bonus
        # (Adding the negative entropy term effectively subtracts the entropy bonus from the objective we maximize,
        # thus encourages higher entropy when minimizing the loss)
        total_loss = policy_loss + entropy_bonus

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._clear_memory()

        # Return both total loss and policy loss for logging
        return total_loss.item(), policy_loss.item()

    def _clear_memory(self):
        """ Clears the stored log probabilities, rewards, and entropies. """
        self.log_probs = []
        self.rewards = []
        self.entropies = [] # Clear entropies

# --- Plotting Function ---
def plot_stats(stats, save_dir):
    """Generates and saves plots for training statistics using Matplotlib."""
    episodes = stats['episode']
    avg_rewards = stats['avg_reward']
    avg_lengths = stats['avg_length']
    solve_rates = stats['solve_rate']
    avg_losses = stats['avg_loss']
    avg_policy_losses = stats['avg_policy_loss']

    plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style

    # Reward Plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, avg_rewards, label='Average Reward')
    plt.title('Average Reward per Reporting Interval')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "average_reward.png"))
    plt.close() # Close the figure to free memory

    # Length Plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, avg_lengths, label='Average Episode Length')
    plt.title('Average Episode Length per Reporting Interval')
    plt.xlabel('Episode')
    plt.ylabel('Average Steps')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "average_length.png"))
    plt.close()

    # Solve Rate Plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, solve_rates, label='Solve Rate')
    plt.title('Solve Rate per Reporting Interval')
    plt.xlabel('Episode')
    plt.ylabel('Solve Rate')
    plt.ylim(0, 1.05) # Set y-axis limits for rate
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "solve_rate.png"))
    plt.close()

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, avg_losses, label='Average Total Loss')
    plt.plot(episodes, avg_policy_losses, label='Average Policy Loss', linestyle='--')
    plt.title('Average Loss per Reporting Interval')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "average_loss.png"))
    plt.close()

    print(f"Plots saved as PNG images to {save_dir}")


# --- Training Loop ---
def train():
    print(f"Initializing Environment... Device: {DEVICE}")
    try:
        # Use relative paths assuming script is run from project root
        # or adjust ENV_CONFIG paths as needed
        script_dir = os.path.dirname(os.path.abspath(__file__))
        guesses_path = os.path.join(script_dir, "guesses.txt")
        answers_path = os.path.join(script_dir, "answers.txt")
        env = WordleEnv(guesses_path=guesses_path, answers_path=answers_path, **ENV_CONFIG)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error creating environment: {e}")
        print(f"Please ensure '{guesses_path}' and '{answers_path}' exist.")
        return

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n
    print(f"Observation Shape: {obs_shape}, Action Size: {action_size}")

    WORD_LENGTH = ENV_CONFIG["word_length"] # Get word length from config
    ALPHABET_SIZE = 26 # Standard English alphabet
    INTERMEDIATE_OUTPUT_SIZE = WORD_LENGTH * ALPHABET_SIZE # 5 * 26 = 130

    # Agent initialization now includes creating the encoding matrix
    agent = ReinforceAgent(env, obs_shape, action_size, lr=LEARNING_RATE, gamma=GAMMA, entropy_coeff=ENTROPY_COEFF, device=DEVICE)

    start_episode = 0 # Keep track of starting episode

    # --- Load existing model if it exists ---
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading existing model from {MODEL_SAVE_PATH}...")
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
            agent.policy_net.load_state_dict(checkpoint['model_state_dict'])
            # Load optimizer state if available and needed
            if 'optimizer_state_dict' in checkpoint:
                agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                print("Optimizer state loaded.")
            # Optionally load episode number if saved
            if 'episode' in checkpoint:
                start_episode = checkpoint['episode'] + 1 # Start from the next episode
                print(f"Resuming training from episode {start_episode}")
            agent.policy_net.train() # Ensure model is in training mode
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting training from scratch.")
            start_episode = 0 # Reset start episode if loading failed
    else:
        print("No existing model found. Starting training from scratch.")

    print(f"Starting Training for {NUM_EPISODES} episodes (from episode {start_episode})...")

    # Lists to store stats for plotting
    stats_history = {
        'episode': [],
        'avg_reward': [],
        'avg_length': [],
        'solve_rate': [],
        'avg_loss': [],
        'avg_policy_loss': []
    }
    # Temporary lists for current reporting interval
    current_rewards = []
    current_lengths = []
    current_losses = []
    current_policy_losses = []
    solved_count = 0

    # Use tqdm for the main loop
    pbar = tqdm(range(NUM_EPISODES), desc="Training Progress")
    for episode in pbar:
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        episode_loss = 0
        episode_policy_loss = 0
        num_updates = 0

        while not terminated and not truncated:
            action_idx, log_prob, entropy = agent.select_action(observation)

            # Avoid storing dummy values if sampling failed (heuristic check)
            store_outcome = not (isinstance(log_prob, torch.Tensor) and log_prob.item() == 0.0 and isinstance(entropy, torch.Tensor) and entropy.item() == 0.0)

            next_observation, reward, terminated, truncated, info = env.step(action_idx)

            if store_outcome:
                 agent.store_outcome(log_prob, reward, entropy)

            observation = next_observation
            episode_reward += reward
            steps += 1

            if steps >= MAX_EPISODE_LENGTH:
                # print(f"Warning: Episode {episode+1} truncated at {steps} steps.") # Silence this
                truncated = True

            if terminated or truncated:
                current_rewards.append(episode_reward)
                current_lengths.append(steps)
                # Check if solved based on the reward structure (positive reward on termination)
                # The new reward is positive only on win.
                if reward > 0 and terminated:
                     solved_count += 1
                break

        # Perform Agent Update after episode ends
        loss, policy_loss = agent.update()
        if loss is not None: # Update might return None if skipped
            current_losses.append(loss)
            current_policy_losses.append(policy_loss)


        # Logging and Stats Collection
        # Adjust episode number for logging if resuming
        log_episode_num = episode + 1 # This is the actual current episode number
        if log_episode_num % PRINT_EVERY == 0:
            avg_reward = np.mean(current_rewards) if current_rewards else 0
            avg_length = np.mean(current_lengths) if current_lengths else 0
            # Calculate solve rate based on the number of episodes in this interval (PRINT_EVERY)
            solve_rate = solved_count / min(PRINT_EVERY, episode + 1) # Correct calculation for interval
            avg_loss = np.mean(current_losses) if current_losses else 0
            avg_policy_loss = np.mean(current_policy_losses) if current_policy_losses else 0

            # --- Use pbar.write instead of print ---
            stats_string = (
                f"Ep {log_episode_num}/{NUM_EPISODES} | " # Use log_episode_num
                f"Avg R: {avg_reward:.2f} | "
                f"Avg L: {avg_length:.2f} | "
                f"Rate (last {PRINT_EVERY}): {solve_rate:.2f} | "
                f"Avg Loss: {avg_loss:.3f}"
            )
            pbar.write(stats_string) # Use pbar.write()

            # Update tqdm description (optional, can remove if pbar.write is enough)
            pbar.set_description(f"Ep {episode+1} | Avg R: {avg_reward:.2f} | Avg L: {avg_length:.2f} | Rate: {solve_rate:.2f} | Loss: {avg_loss:.3f}")

            # Store stats for plotting
            stats_history['episode'].append(log_episode_num) # Use log_episode_num
            stats_history['avg_reward'].append(avg_reward)
            stats_history['avg_length'].append(avg_length)
            stats_history['solve_rate'].append(solve_rate)
            stats_history['avg_loss'].append(avg_loss)
            stats_history['avg_policy_loss'].append(avg_policy_loss)

            # Reset counters for the next interval
            current_rewards = []
            current_lengths = []
            current_losses = []
            current_policy_losses = []
            solved_count = 0

    pbar.close() # Close the progress bar explicitly

    print("\nTraining Finished.")
    # Optional: Save model periodically
    # torch.save({'model_state_dict': agent.policy_net.state_dict()}, f"wordle_solver_ep{episode+1}.pth")


    print("\nTraining Finished.")
    # Save the final trained model (include episode number)
    torch.save({
        'episode': NUM_EPISODES - 1, # Save the last completed episode index
        'model_state_dict': agent.policy_net.state_dict(),
        'optimizer_state_dict': agent.optimizer.state_dict(),
        # Add any other info needed, like ENV_CONFIG?
    }, MODEL_SAVE_PATH)
    print(f"\nModel saved to {MODEL_SAVE_PATH}")

    # Generate and save plots
    plot_stats(stats_history, PLOT_SAVE_DIR)

    # --- Optional Evaluation Phase ---
    print("\nStarting Evaluation...")
    agent.policy_net.eval() # Set model to evaluation mode
    eval_rewards = []
    eval_lengths = []
    eval_solved_count = 0

    # Use the defined constant here
    for eval_episode in tqdm(range(EVAL_EPISODES), desc="Evaluating"):
        observation, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        steps = 0
        while not terminated and not truncated:
            with torch.no_grad(): # No gradient needed for evaluation
                 action_idx, _, _ = agent.select_action(observation) # Use greedy selection? Or keep sampling?

            next_observation, reward, terminated, truncated, info = env.step(action_idx)
            observation = next_observation
            episode_reward += reward
            steps += 1

            if steps >= MAX_EPISODE_LENGTH:
                truncated = True

            if terminated or truncated:
                eval_rewards.append(episode_reward)
                eval_lengths.append(steps)
                if terminated and info.get('solved', False): # Check if the game was won
                    eval_solved_count += 1
                break

    avg_eval_reward = np.mean(eval_rewards) if eval_rewards else 0
    avg_eval_length = np.mean(eval_lengths) if eval_lengths else 0
    eval_solve_rate = eval_solved_count / EVAL_EPISODES if EVAL_EPISODES > 0 else 0

    # Use the constant in the print statement
    print(f"\nEvaluation Results ({EVAL_EPISODES} episodes):")
    print(f"  Average Reward: {avg_eval_reward:.2f}")
    print(f"  Average Length: {avg_eval_length:.2f}")
    print(f"  Solve Rate: {eval_solve_rate:.2f}")

    env.close()


if __name__ == "__main__":
    train()
