import gymnasium_robotics
import gymnasium as gym
import numpy as np
import envpool

from agent.agent import RobotArmAgent
from train import start_train
import settings

gym.register_envs(gymnasium_robotics)

train_envs = envpool.make_gymnasium(
    "FetchReach-v4", 
    num_envs=settings.ENVS_NUM, seed=0, 
    max_episode_steps=50,
    render_mode=None
)
eval_env = gym.make("FetchReach-v4", render_mode="rgb_array", max_episode_steps=250)

observation_space = train_envs.observation_space["observation"].shape[0]
action_space = train_envs.action_space.shape[0]

agent = RobotArmAgent(state_space=observation_space, action_space=action_space)
agent.reset_noise()

start_train(train_envs, eval_env, agent)
