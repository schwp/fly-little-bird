
from agents.train import train_dqn_pygame
from agents.dqn import DQN
import argparse
import flappy_bird_gymnasium
import gymnasium as gym
import os
import pygame
from time import sleep
import torch

parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Path to the model file')
parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true', help='Train the agent')
group.add_argument('--load', action='store_true', help='Load the trained model')

args = parser.parse_args()

file = "models/" + args.filename
env = gym.make("FlappyBird-v0", render_mode='human', use_lidar=False)

if args.train:
    trained_agent = train_dqn_pygame(env, num_episodes=args.episodes, filename=file)

elif args.load:
    if not os.path.exists(file):
        print(f"Error: Model file '{args.filename}' not found.")
        exit(1)
    trained_agent = DQN(env.observation_space.shape[0], env.action_space.n)
    trained_agent.load_state_dict(torch.load(file, weights_only=True))

    state = env.reset()
    done = False
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
                env.close()
                break

        with torch.no_grad():
            q_values = trained_agent(torch.FloatTensor(state))
            action = torch.argmax(q_values).item()

        state, reward, done, _, _ = env.step(action)
        env.render()
        sleep(0.03)

    env.close()
