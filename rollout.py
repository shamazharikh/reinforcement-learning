# import click
import time
import numpy as np
import gym
import matplotlib.pyplot as plt
import itertools 

from gym.envs.registration import registry, register, make, spec
from gym import spaces
from agents import PolicyGradient
from visualizers import PolicyGraph, LearningGraph

# def chakra_get_action(theta, ob, rng=np.random):
#     ob_1 = include_bias(ob)
#     mean = theta.dot(ob_1)
#     return rng.normal(loc=mean, scale=1.)		 
def get_shape(space):
    if type(space) == spaces.Discrete:
        return space.n
    if type(space) == spaces.Box:
        return space.shape[0]

def main(env, config_name, learning_graph, max_iters=1000, batch_size=10, discount=0.9, max_episode_steps = 100):
    
    max_step_size = 0.025
    obs_dim = get_shape(env.observation_space)
    action_dim = get_shape(env.action_space)
    
    agent = PolicyGradient(action_dim=action_dim, observation_dim=obs_dim, max_movement=max_step_size, gamma=discount)
    agent.set_train_mode()
    
    for itr_no in range(max_iters):
        agent.set_train_mode()
        batch_rewards = []
    
        for i in range(batch_size):
            done = False
            ob = env.reset()
            episode_rewards = []
            for t in range(max_episode_steps):
                action = agent.get_action(ob)
                new_ob, rew, done, _ = env.step(action)
                ob = np.copy(new_ob)
                agent.store_reward(np.copy(rew))
                episode_rewards.append(np.copy(rew))
                if done:
                    break
            batch_rewards.append(np.sum(episode_rewards))
        agent.update()
        batch_rewards_mean = np.mean(batch_rewards)

        if itr_no % (max_iters/100) == 0:
            done=False
            agent.set_eval_mode()
            ob = env.reset()
            env.render()
            while not done:
                action = agent.get_action(ob)
                new_ob, rew, done, _ = env.step(action)
                env.render()
                time.sleep(0.1)
                ob = np.copy(new_ob)
            print('Iteration no: {}Average Rewards:{}'.format(itr_no, batch_rewards_mean))
    return agent
    

if __name__ == "__main__":
    max_episode_steps = 100
    
    
    batch_sizes = [10]
    max_iters = [2000]
    discounts = [0.9, 0.5]
    iterator = itertools.product(batch_sizes, max_iters, discounts)
    
    # learning_graph =LearningGraph('Hyperparameter Search for LunarLander-v2')
    env = gym.make('LunarLander-v2')
    env.reset()
    for batch_size, max_iter, discount in iterator:
        # config_name = 'b={},m={},d={}'.format(batch_size, max_iter, discount)
        main(env, batch_size=batch_size, max_iters=max_iter, discount=discount,max_episode_steps=max_episode_steps)
    # learning_graph.save_figure_and_close('./Chakra_hyperparameters.png')

    # learning_graph =LearningGraph('Hyperparameter Search for VishamC')
    # iterator = itertools.product(batch_sizes, max_iters, discounts)
    # for batch_size, max_iter, discount in iterator:
    #     config_name = 'b={},m={},d={}'.format(batch_size, max_iter, discount)
    #     main('VishamC', config_name, learning_graph, batch_size=batch_size, 
    #         max_iters=max_iter, discount=discount,max_episode_steps=max_episode_steps)
    # learning_graph.save_figure_and_close('./VishamC_hyperparameters.png')
    # main('VishamC', max_episode_steps=max_episode_steps)