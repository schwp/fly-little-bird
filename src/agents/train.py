from .dqn import DQN
import torch
from utils.ReplayBuffer import ReplayBuffer
import random

def train_dqn(env, num_episodes=500, batch_size=64, gamma=0.99, 
              lr=1e-3, epsilon_decay=0.995, min_epsilon=0.01, filename=None):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy_net = DQN(state_dim, action_dim)
    target_net = DQN(state_dim, action_dim)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(10000)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        reward_total = 0

        while not done:
            if random.random() < max(min_epsilon, epsilon_decay**episode):
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.FloatTensor(state))
                    action = q_values.argmax().item()

            next_state, reward, done, _ = env.step(action)
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state
            reward_total += reward

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                state_batch = torch.FloatTensor(states)
                action_batch = torch.LongTensor(actions).unsqueeze(1)
                reward_batch = torch.FloatTensor(rewards).unsqueeze(1)
                next_state_batch = torch.FloatTensor(next_states)
                done_batch = torch.FloatTensor(dones).unsqueeze(1)

                q_values = policy_net(state_batch).gather(1, action_batch)
                with torch.no_grad():
                    next_q_values = target_net(next_state_batch).max(1)[0].unsqueeze(1)
                    target = reward_batch + (1 - done_batch) * gamma * next_q_values

                loss = torch.nn.MSELoss()(q_values, target)

                optimizer.zero_grad()
                loss.backward() 
                optimizer.step()

        epsilon = max(min_epsilon, epsilon_decay**episode)

        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {reward_total}, Epsilon: {epsilon:.2f}")

    torch.save(policy_net.state_dict(), filename if filename else "default.pth")
    return policy_net
