import torch
import numpy as np
from game import WordleEnv, WordleGame # Need base game for feedback calc
from solver import PolicyNetwork, ReinforceAgent, ENV_CONFIG, DEVICE, MODEL_SAVE_PATH # Import necessary components
import argparse
import re # For parsing feedback

def parse_feedback(feedback_str):
    """ Parses G Y - feedback string into a list. """
    feedback_str = feedback_str.upper().strip()
    if not re.fullmatch(r"[GY-]{" + str(ENV_CONFIG['word_length']) + r"}", feedback_str):
        return None
    return list(feedback_str)

def run_inference(mode, target_word=None, model_path=MODEL_SAVE_PATH):
    print(f"--- Wordle Inference (Mode: {mode}) ---")
    print(f"Using device: {DEVICE}")

    # --- Initialize Environment (needed for mappings and state structure) ---
    # Use silent=True for env unless we want its render function later
    try:
        # We need a dummy env to get state shape, action space, mappings etc.
        # Set silent=False if you want to use env.render() in interactive mode
        env = WordleEnv(**ENV_CONFIG, silent=(mode=='known'))
    except (ValueError, FileNotFoundError) as e:
        print(f"Error creating environment: {e}")
        return

    obs_shape = env.observation_space.shape
    action_size = env.action_space.n

    # --- Load Model ---
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    # Instantiate only the policy network for inference
    policy_net = PolicyNetwork(obs_shape, action_size).to(DEVICE)
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
        policy_net.load_state_dict(checkpoint['model_state_dict'])
        policy_net.eval() # Set to evaluation mode
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model state dict: {e}")
        return

    # --- Create a temporary Agent instance (needed for select_action with masking) ---
    # We don't need the optimizer or training parts, but the masking logic is in the agent
    # Pass the dummy env to it
    agent = ReinforceAgent(env, obs_shape, action_size, device=DEVICE)
    agent.policy_net = policy_net # Use the loaded network

    # --- Reset State ---
    # We manage the observation manually in interactive mode
    # Start with a padded observation
    observation = np.full(obs_shape, env.char_to_int['<PAD>'], dtype=np.int32)
    current_attempt = 0
    game_won = False

    # --- Inference Loop ---
    with torch.no_grad():
        while current_attempt < env.game.max_attempts:
            print(f"\n--- Attempt {current_attempt + 1}/{env.game.max_attempts} ---")

            # 1. Agent Selects Action (with masking based on current observation)
            action_idx, _, _ = agent.select_action(observation) # We only need the action index
            guess_word = env.action_to_word[action_idx]
            print(f"Agent Guess: {guess_word.upper()}")

            # 2. Get Feedback
            feedback = None
            if mode == 'known':
                if target_word is None:
                    print("Error: Target word needed for 'known' mode.")
                    return
                if guess_word == target_word:
                    feedback = ['G'] * env.game.word_length
                    game_won = True
                else:
                    # Use the game's logic to get feedback
                    feedback = WordleGame.get_feedback(guess_word, target_word)
                feedback_str = "".join(feedback)
                print(f"Feedback:    {feedback_str}")

            elif mode == 'interactive':
                while feedback is None:
                    feedback_str = input(f"Enter Feedback (e.g., -GY--): ")
                    feedback = parse_feedback(feedback_str)
                    if feedback is None:
                        print(f"Invalid feedback format. Please use {env.game.word_length} chars from G, Y, -.")
                if feedback == ['G'] * env.game.word_length:
                    game_won = True

            else:
                print(f"Error: Unknown mode '{mode}'")
                return

            # 3. Update Observation for the Agent
            try:
                guess_ints = [env.char_to_int[c] for c in guess_word]
                feedback_ints = [env.feedback_to_int[fb] for fb in feedback]
                observation[current_attempt, :env.game.word_length] = guess_ints
                observation[current_attempt, env.game.word_length:] = feedback_ints
            except KeyError as e:
                print(f"Error updating observation: Invalid character/feedback {e}")
                return

            # 4. Check for Win/End
            if game_won:
                print(f"\nSuccess! Agent found the word '{guess_word.upper()}' in {current_attempt + 1} attempts.")
                break

            current_attempt += 1

    if not game_won:
        print(f"\nFailure! Agent could not find the word in {env.game.max_attempts} attempts.")
        if mode == 'known' and target_word:
             print(f"The target word was: {target_word.upper()}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Wordle Solver Inference")
    parser.add_argument("mode", choices=['known', 'interactive'], help="Inference mode: 'known' (provide answer) or 'interactive' (provide feedback).")
    parser.add_argument("-w", "--word", type=str, default=None, help="The target word (required for 'known' mode). Must be 5 letters.")
    parser.add_argument("-m", "--model", type=str, default=MODEL_SAVE_PATH, help=f"Path to the trained model file (default: {MODEL_SAVE_PATH}).")

    args = parser.parse_args()

    if args.mode == 'known' and (args.word is None or len(args.word) != ENV_CONFIG['word_length']):
        parser.error(f"'known' mode requires a target word of length {ENV_CONFIG['word_length']} using --word.")
    if args.word:
        args.word = args.word.lower() # Ensure lowercase

    run_inference(args.mode, target_word=args.word, model_path=args.model)