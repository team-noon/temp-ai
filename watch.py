from stable_baselines3 import PPO
from game import SoccerEnv

# Load the model
model = PPO.load("soccer_ppo")

# Run it
env = SoccerEnv(render_mode="human")
obs, _ = env.reset()
env.opponent.difficulty = 1.0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print(info)
        obs, _ = env.reset()
        env.opponent.difficulty = 1.0