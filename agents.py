import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

def one_hot_encoding(actions, l=10):
    return np.stack([actions==i for i in range(l)], axis=1)

def init_weights(model):
    for parameter in model.parameters():
        if parameter.ndimension() == 2:
            torch.nn.init.xavier_uniform(parameter, gain=0.01)

class Policy(nn.Module):
    def __init__(self, input_dim, hidden_layer, output_dim):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(input_dim, hidden_layer, bias=True)
        self.affine2 = nn.Linear(hidden_layer, output_dim, bias=True) 
        
        self.saved_log_probs = []
        self.rewards = []
        
    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)

class PolicyGradient(object):
    def __init__(self, action_dim=2, observation_dim=2, max_movement=0.025, gamma=0.9, alpha=0.1, decay=0.9995):
        self.action_dim = action_dim
        self.observation_dim = observation_dim
        self.max_movement = max_movement
        self.gamma = gamma
        self.alpha = alpha
        self.decay = decay
        self.model = Policy(observation_dim, 100, action_dim)
        self.model.cuda()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.alpha, weight_decay=self.decay)
        # self.seed()
        self.reset_memory()

    def set_train_mode(self):
        self.model.train()

    def set_eval_mode(self):
        self.model.eval()

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.model(Variable(state).cuda())
        m = Categorical(probs)
        action = m.sample() 
        self.model.saved_log_probs.append(m.log_prob(action)) 
        return action.data[0]

    def store_reward(self, reward):
        self.model.rewards.append(reward)

    def update(self):
        policy_loss = []
        rewards = self.get_discounted_rewards(self.model.rewards)
            
        # turn rewards to pytorch tensor and standardize
        rewards = torch.Tensor(rewards).cuda()
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
        
        for log_prob, reward in zip(self.model.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        
        # clean rewards and saved_actions
        del self.model.rewards[:]
        del self.model.saved_log_probs[:]

    def reset_memory(self):
        init_weights(self.model)

    def get_discounted_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        l = len(rewards)
        discounts = np.array([self.gamma**i for i in range(l)])
        for i in range(len(rewards)):
            discounted_rewards[i]=np.sum(rewards[i:]*discounts[:l-i])
        return discounted_rewards

    def save_model(self, savepath): 
        torch.save(self.model.state_dict(), savepath)

    def load_model(self, savepath):
        self.model.load_state_dict(torch.load(savepath))

