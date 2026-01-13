from deep_q_network import *
from replay_buffer import *
from flappybird_preprocess import *

import flappy_bird_gymnasium
import torch.optim as optim
import os

def save_checkpoint(path, episode, policy_net, target_net, optimizer, steps_done):
    torch.save({
        'episode': episode,
        'policy_net_state_dict': policy_net.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'steps_done': steps_done,
    }, path)

def load_checkpoint(path, policy_net, target_net, optimizer):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        target_net.load_state_dict(checkpoint['target_net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        episode = checkpoint['episode']
        steps_done = checkpoint['steps_done']
        return episode, steps_done
    else:
        return 0, 0

if __name__ == "__main__":
    BATCH_SIZE = 128
    GAMMA = 0.95
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 5000
    TARGET_UPDATE = 500
    LR = 3e-4
    MEMORY_SIZE = 50000
    NUM_EPISODES = 100000
    CHECKPOINT_FILE = "checkpoint.pth"

    device = torch.device("cpu")
    print(f"Using device: {device}")

    env = gym.make("FlappyBird-v0", render_mode="rgb_array")
    env = FlappyBirdPreprocess(env)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    policy_net = DuelingDQN((4, 84, 84), env.action_space.n).to(device)
    target_net = DuelingDQN((4, 84, 84), env.action_space.n).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    memory = ReplayBuffer(MEMORY_SIZE)

    start_episode, steps_done = load_checkpoint(CHECKPOINT_FILE, policy_net, target_net, optimizer)
    best_score = -np.inf
    for i_episode in range(start_episode, NUM_EPISODES):
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
        if total_reward > best_score:
            best_score = total_reward
            with open('best_score.txt', 'a') as file:
                file.write(str(f'NEW BEST SCORE: {best_score:.2f} AT EPISODE {i_episode}\n'))


        if i_episode % 10 == 0:
            save_checkpoint(CHECKPOINT_FILE, i_episode, policy_net, target_net, optimizer, steps_done)



    print("Training Complete.")
    env.close()