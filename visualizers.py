import matplotlib.pyplot as plt
import numpy as np

class PolicyGraph(object):

	def __init__(self, name):
		self.fig, self.ax = plt.subplots(figsize=(7,7))
		self.fig.suptitle(name)
		self.ax.set_ylim(-1.0, 1.0)
		self.ax.set_xlim(-1.0, 1.0)

	def visualize_policy(self, states, actions):
		# print(states, actions)
		U = actions[:,0].clip(-0.025, 0.025)
		V = actions[:,1].clip(-0.025, 0.025)
		X = states[:,0]
		Y = states[:,1]
		q = self.ax.quiver(X, Y, U, V, pivot='tail')

	def save_figure_and_close(self, savepath):
		self.fig.savefig(savepath)
		plt.close(self.fig)

class LearningGraph(object):
	def __init__(self, name, max_episode_steps=200):
		self.fig, self.ax1 = plt.subplots(figsize=(8,7))
		self.fig.suptitle(name)
		self.alpha = 1.0
		self.ax1.set(xlabel='Batch_Number',ylabel='Batch Reward')
		# self.ax1.set_ylim(0,1.7)
		# self.ax2.set_ylim(0, 1.1*max_episode_steps)
		# self.ax2.set(xlabel='Episode Number', ylabel='Episode Length')
		self.ax1_lines = []
		# self.ax2_lines = []
	def visualize_average_episode_rewards(self, reward_history, label):
		line, = self.ax1.plot(reward_history, label=label, alpha=self.alpha)
		self.ax1_lines.append(line)
		self.alpha -= 0.1

	def save_figure_and_close(self, savepath):
		self.ax1.legend(handles=self.ax1_lines, loc=4)
		# self.ax2.legend(handles=self.ax2_lines, loc=4)
		self.fig.savefig(savepath)
		plt.close(self.fig)