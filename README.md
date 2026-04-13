# Flappy Bird Reinforcement Learning

A deep reinforcement learning project that trains a DQN agent to play Flappy Bird using `gymnasium` and `flappy_bird_gymnasium`.

## Overview

This project uses a Deep Q-Network (DQN) with:

- A policy network for action selection
- A target network for stable Q-value updates
- Experience replay to break correlation between samples
- Epsilon-greedy exploration during training

The training script stores the best model weights and logs into a local `runs/` directory.

## Project Structure

- `agent.py` - training and evaluation loop
- `dqn.py` - neural network used to approximate Q-values
- `replay_memory.py` - replay buffer implementation
- `parameters.yaml` - hyperparameter configuration
- `runs/` - generated at runtime for logs and saved models

## Requirements

Use Python 3.10+ if possible. Install the main dependencies:

```bash
pip install torch gymnasium flappy-bird-gymnasium pyyaml
```

Depending on your system, you may also need the environment package that registers `FlappyBird-v0`.

## How It Works

1. The environment returns an observation vector describing the current game state.
2. The DQN predicts a Q-value for each possible action.
3. During training, the agent sometimes explores by choosing a random action.
4. Transitions are stored in replay memory.
5. A random mini-batch is sampled from memory to update the network.
6. The target network is synced periodically to stabilize learning.
7. The best-performing model is saved to disk.

## Configuration

Hyperparameters live in `parameters.yaml`.

The default profile is:

- `alpha`: learning rate
- `gamma`: discount factor
- `epsilon_init`: starting exploration rate
- `epsilon_min`: minimum exploration rate
- `epsilon_decay`: exploration decay per episode
- `replay_memory_size`: number of stored transitions
- `mini_batch_size`: batch size sampled from replay memory
- `network_sync_rate`: how often the target network is updated
- `reward_threshold`: optional episode stop threshold

If you want to try a different setup, add another top-level key to `parameters.yaml` and pass that key to the script.

## Training

Run training with:

```bash
python agent.py default --train
```

What this does:

- Creates the environment
- Builds the policy and target DQN networks
- Starts epsilon-greedy training
- Saves the best model to `runs/default.pt`
- Appends progress logs to `runs/default.log`

## Evaluation

After training, run the agent in render mode:

```bash
python agent.py default
```

This loads `runs/default.pt` and plays the game with visual rendering enabled.

## Output Files

Generated artifacts are written to `runs/`:

- `runs/default.pt` - saved model weights
- `runs/default.log` - training log of best episodes

## Notes

- The environment is currently hardcoded as `FlappyBird-v0` in `agent.py`.
- The project automatically uses MPS, CUDA, or CPU depending on what is available.
- If you delete the `runs/` folder, it will be recreated automatically when the script runs.

## Suggested Workflow

1. Install dependencies
2. Train the model with `python agent.py default --train`
3. Review the saved weights in `runs/`
4. Launch evaluation with `python agent.py default`

## License

Add a license here if you want to publish the project publicly.
