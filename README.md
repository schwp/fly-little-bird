# Fly Little Bird 🐦

A Reinforcement Learning Agent that learns to play Flappy Bird using Deep 
Q-Network (DQN) and PyTorch.

## Overview

This project implements a Deep Q-Learning agent that learns to play Flappy Bird 
through reinforcement learning. The agent uses a neural network to predict the
best actions (flap or not flap) based on the current game state (observations),
gradually improving its performance through trial and error.

## Features

- **Deep Q-Network (DQN)** implementation using PyTorch
- **Custom Flappy Bird Environment** compatible with OpenAI Gym/Gymnasium
- **Pygame-based visualization** for watching the agent play
- **Training and inference modes** with model saving/loading
- **Configurable training parameters** (episodes, learning rate, etc.)

## Setup
Create a Virtual Environnement and install all dependencies to run the project
without any problem :
```bash
python3 -m venv .venv
pip install -r requirements.txt
```

## Usage

### Training a New Agent

Train a new DQN agent for 1000 episodes (default):
```bash
cd src
python main.py "model_name.pth" --train
```

Train with custom number of episodes:
```bash
cd src
python main.py "model_name.pth" --train --episodes 2000
```

### Loading and Testing a Trained Agent

Load a pre-trained model and watch it play:
```bash
cd src
python main.py "model_name.pth" --load
```
