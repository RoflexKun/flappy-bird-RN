import random
from collections import deque

import cv2
import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc_input_dim = self._get_conv_output(input_shape)

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_output(self, shape):
        x = torch.zeros(1, *shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return int(torch.prod(torch.tensor(x.size())))

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward),
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)


class FlappyBirdPreprocess(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(84, 84), dtype=np.uint8)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        frame = self.env.render()
        return self._process_frame(frame), reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        frame = self.env.render()
        return self._process_frame(frame), info

    def _process_frame(self, frame):
        if frame is None:
            return np.zeros((84, 84), dtype=np.uint8)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84))
        return resized


if __name__ == "__main__":
    BATCH_SIZE = 32
    GAMMA = 0.99
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 10000
    TARGET_UPDATE = 1000
    LR = 1e-4
    MEMORY_SIZE = 20000
    NUM_EPISODES = 500

    device = torch.device("cpu")
    print(f"Using device: {device}")

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = FlappyBirdPreprocess(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    policy_net = DQN((4, 84, 84), env.action_space.n).to(device)
    target_net = DQN((4, 84, 84), env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    steps_done = 0

    for i_episode in range(NUM_EPISODES):
        state, info = env.reset()
        state = np.array(state)

        total_reward = 0
        done = False

        while not done:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                            np.exp(-1. * steps_done / EPS_DECAY)

            if random.random() > eps_threshold:
                with torch.no_grad():
                    state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0) / 255.0
                    q_values = policy_net(state_t)
                    action = q_values.max(1)[1].item()
            else:
                action = env.action_space.sample()

            steps_done += 1

            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = np.array(next_state)
            done = terminated or truncated

            if not done:
                reward += 0.1

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if len(memory) > BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                states_t = torch.tensor(states, dtype=torch.float32, device=device) / 255.0
                actions_t = torch.tensor(actions, dtype=torch.int64, device=device).unsqueeze(1)
                rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_states_t = torch.tensor(next_states, dtype=torch.float32, device=device) / 255.0
                dones_t = torch.tensor(dones, dtype=torch.float32, device=device)

                current_q = policy_net(states_t).gather(1, actions_t)

                with torch.no_grad():
                    max_next_q = target_net(next_states_t).max(1)[0]
                    target_q = rewards_t + (GAMMA * max_next_q * (1 - dones_t))

                loss = F.smooth_l1_loss(current_q.squeeze(), target_q)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if steps_done % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {i_episode}, Total Reward: {total_reward:.2f}, Epsilon: {eps_threshold:.3f}")

    print("Training Complete.")
    env.close()