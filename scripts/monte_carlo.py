import gymnasium as gym
import time
import os
import matplotlib.pyplot as plt

import text_flappy_bird_gym

from src.agents import OffPolicyMonteCarloAgent
from src.utils.data_structures import MonteCarloParams

from tqdm import tqdm
import numpy as np

import argparse

def train(env, agent: OffPolicyMonteCarloAgent, num_episodes, max_steps, visualize=False):
    """
    Train the agent.

    Args:
        env (gym.Env): Environment.
        agent (SarsaAgent): SARSA agent.
        num_episodes (int): Number of episodes.
        max_steps (int): Maximum number of steps per episode.
    """
    rewards = np.zeros(num_episodes)
    for e in tqdm(range(num_episodes)):
        state, _ = env.reset()
        action = agent.act(state)
        total_reward = 0
        episode = []
        for _ in range(max_steps):
            next_state, reward, done, _, _ = env.step(action)
            next_action = agent.act(next_state)
            total_reward += reward
            if done:
                break
            episode.append((state, action, reward))
            state = next_state
            action = next_action
        rewards[e] = total_reward
        agent.update(episode)
   
    if visualize:
        plt.plot(rewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Total Reward vs Episode')
        plt.show()
    
    return rewards

def play_episode(agent: OffPolicyMonteCarloAgent, env, max_iter=1000, render=False):
    state, _ = env.reset()
    total_reward = 0
    done = False
    while not done and max_iter > 0:
        action = agent.get_best_action(state)
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
        if render:
            print(chr(27) + "[2J")
            print(env.render())
            time.sleep(0.1)
        max_iter -= 1
    return total_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a MonteCarlo agent to play Flappy Bird.')
    parser.add_argument('--num_episodes', type=int, default=50000, help='Number of episodes to train the agent.')
    parser.add_argument('--max_steps', type=int, default=1000, help='Maximum number of steps per episode.')
    parser.add_argument('--alpha', type=float, default=0.05, help='Learning rate.')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor.')
    parser.add_argument('--epsilon', type=float, default=0.1, help='Exploration rate.')
    parser.add_argument('--visualize', action='store_true', help='Visualize the training process.')
    parser.add_argument('--render_after_training', action='store_true', help='Render the trained agent after training.')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the agent after training.', default=True)
    parser.add_argument('--num_eval_episodes', type=int, default=100, help='Number of episodes to evaluate the agent.')
    args = parser.parse_args()
    
    # Parameters
    num_episodes = args.num_episodes
    max_steps = args.max_steps
    alpha = args.alpha
    gamma = args.gamma
    epsilon = args.epsilon
    visualize = args.visualize
    render_after_training = args.render_after_training
    evaluate = args.evaluate
    num_eval_episodes = args.num_eval_episodes

    # initiate environment
    env = gym.make('TextFlappyBird-v0')

    # initiate MonteCarlo parameters
    montecarlo_params = MonteCarloParams(num_actions=env.action_space.n, gamma=gamma, epsilon=epsilon)
    agent = OffPolicyMonteCarloAgent(montecarlo_params)

    # train the agent
    train(env, agent, num_episodes=num_episodes, max_steps=max_steps, visualize=visualize)

    # render the trained agent
    if render_after_training:
        play_episode(agent, env, render=True)

    # evaluate the agent
    if evaluate:
        total_rewards = []
        for _ in range(num_eval_episodes):
            total_reward = play_episode(agent, env, render=False)
            total_rewards.append(total_reward)
        print(f'Mean total reward: {np.mean(total_rewards)}')
        print(f'Std total reward: {np.std(total_rewards)}')
        print(f'Min total reward: {np.min(total_rewards)}')
        print(f'Max total reward: {np.max(total_rewards)}')
        print(f'Quantiles: {np.quantile(total_rewards, [0.25, 0.5, 0.75])}')
