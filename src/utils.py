import torch
from tqdm.notebook import tqdm_notebook as tqdm

if torch.backends.mps.is_available():
    DEVICE = torch.device(device="mps")
elif torch.cuda.is_available():
    DEVICE = torch.device(device="cuda")
else:
    DEVICE = torch.device(device="cpu")


def run_env(env, online_value_function, episodes = 100):
    
    reached_end_goal = 0
    rewards = []
    
    for episode in tqdm(range(episodes)):
        observation = env.reset()
        observation = list(observation[0])
        terminated = False
        
        reward_per_episode = 0
        
        # For each episode
        while not terminated:
            # Make a Transistion
            action = torch.argmax(online_value_function.act(observation), dim=1).item()
            new_observation, reward, terminated, truncated, _ = env.step(action)
            terminated = terminated or truncated
            reward_per_episode += reward
            observation = new_observation
        
        rewards.append(reward_per_episode)
    
    env.close()
    return rewards