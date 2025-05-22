import torch
import gym
import numpy as np
import time
from dqn import QNetwork, make_env
from hyperparams import Hyperparameters as params

def test_model(model_path, episodes=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = f"{params.env_id}__{params.exp_name}__{params.seed}__test_{int(time.time())}"
    
    # Set up environment (no vector env for simplicity)
    env = make_env(params.env_id, params.seed, 0, True, run_name)()
    
    # Patch in single_action_space so QNetwork doesn't fail
    env.single_action_space = env.action_space  

    # Load model
    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()

    for episode in range(episodes):
        
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(np.array(obs), device=device).unsqueeze(0)
            with torch.no_grad():
                q_values = q_network(obs_tensor)
            action = torch.argmax(q_values, dim=1).item()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            # env.render()  # OpenCV window or terminal rendering

        print(f"Episode {episode + 1} reward: {total_reward}")

    env.close()

if __name__ == "__main__":
    # model_path = "tested_models\DQN_Breakout_step_1800000_model.pt"
    model_path = "tested_models\DQN_Breakout_step_5900000_model.pt"
    # model_path = "tested_models\DQN_Breakout_step_9000000_model.pt"

    test_model(model_path)

