import gymnasium as gym
import time
import os
import matplotlib.pyplot as plt

import text_flappy_bird_gym

from tqdm import tqdm
import numpy as np

import argparse


def play_episode(env, max_iter=1000, render=False):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done and max_iter > 0:
        action = env.action_space.sample()
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if render:
            print(chr(27) + "[2J")
            print(env.render())
            time.sleep(0.1)
        max_iter -= 1
    return total_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test a random agent to play Flappy Bird.')
    parser.add_argument('--render_after_training', action='store_true', help='Render the trained agent after training.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the agent after training.', default=True)
    parser.add_argument('--num_eval_episodes', type=int, default=10000, help='Number of episodes to evaluate the agent.')
    args = parser.parse_args()
    
    # Parameters
    render_after_training = args.render_after_training
    evaluate = args.evaluate
    num_eval_episodes = args.num_eval_episodes

    # initiate environment
    env = gym.make('TextFlappyBird-v0')
    # render the trained agent
    if render_after_training:
        play_episode(env, render=True)

    # evaluate the agent
    if evaluate:
        total_rewards = []
        for _ in range(num_eval_episodes):
            total_reward = play_episode(env, render=False)
            total_rewards.append(total_reward)
        print(f'Mean total reward: {np.mean(total_rewards)}')
        print(f'Std total reward: {np.std(total_rewards)}')
        print(f'Min total reward: {np.min(total_rewards)}')
        print(f'Max total reward: {np.max(total_rewards)}')
        print(f'Quantiles: {np.quantile(total_rewards, [0.25, 0.5, 0.75])}')
