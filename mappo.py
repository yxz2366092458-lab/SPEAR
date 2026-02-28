import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
import os
from copy import deepcopy

class MAPPO:
    '''
    Implementation of MAPPO algorithm for multi-agent traffic light control
    Based on: Multi-Agent Proximal Policy Optimization
    '''
    def __init__(self, n_rows, n_cols, config):
        '''
        Attributes
        ----------
        - n_rows : the number of rows in the grid
        - n_cols : the number of columns in the grid
        - config : configuration object
        '''
        self.name = "MAPPO"
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.num_lights = n_rows * n_cols
        self.discount = config.alg.discount
        self.gae_lambda = config.alg.gae_lambda if hasattr(config.alg, 'gae_lambda') else 0.95
        self.clip_epsilon = config.alg.clip_epsilon if hasattr(config.alg, 'clip_epsilon') else 0.2
        self.value_coef = config.alg.value_coef if hasattr(config.alg, 'value_coef') else 0.5
        self.entropy_coef = config.alg.entropy_coef if hasattr(config.alg, 'entropy_coef') else 0.01
        self.lr = config.alg.lr if hasattr(config.alg, 'lr') else 0.0003
        self.config = config
        
        # Initialize actor and critic networks for each agent
        self.actors = []
        self.critics = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(self.num_lights):
            # Actor network (policy)
            actor = Actor(n_rows, n_cols, lr=self.lr, agent_id=i)
            
            # Critic network (value function)
            critic = Critic(n_rows, n_cols, lr=self.lr, agent_id=i)
            
            self.actors.append(actor)
            self.critics.append(critic)
            
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=self.lr))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=self.lr))
        
        # Experience buffers
        self.buffer_size = config.alg.buffer_size if hasattr(config.alg, 'buffer_size') else 2048
        self.batch_size = config.alg.batch_size if hasattr(config.alg, 'batch_size') else 64
        self.num_epochs = config.alg.num_epochs if hasattr(config.alg, 'num_epochs') else 10
        
        # Initialize buffers
        self.reset_buffers()
    
    def reset_buffers(self):
        '''Reset experience buffers'''
        self.buffers = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': [],
            'advantages': [],
            'returns': []
        }
    
    def select_action(self, local_obs, evaluate=False):
        '''
        Select actions for all agents
        
        Attributes
        ----------
        - local_obs : local observations for each agent [num_lights, obs_dim]
        - evaluate : whether to use evaluation mode (no exploration)
        
        Returns
        -------
        - actions : selected actions for each agent [num_lights]
        - log_probs : log probabilities of selected actions [num_lights]
        - values : state values [num_lights]
        '''
        actions = []
        log_probs = []
        values = []
        
        for i in range(self.num_lights):
            obs_tensor = torch.FloatTensor(local_obs[i]).unsqueeze(0)
            
            with torch.no_grad():
                # Get action distribution
                action_probs = self.actors[i](obs_tensor)
                action_dist = torch.distributions.Categorical(action_probs)
                
                # Sample action
                if evaluate:
                    action = torch.argmax(action_probs, dim=1).item()
                else:
                    action = action_dist.sample().item()
                
                # Get log probability
                log_prob = action_dist.log_prob(torch.tensor([action]))
                
                # Get state value
                value = self.critics[i](obs_tensor)
            
            actions.append(action)
            log_probs.append(log_prob.item())
            values.append(value.item())
        
        return np.array(actions), np.array(log_probs), np.array(values)
    
    def store_transition(self, local_obs, actions, rewards, values, log_probs, dones):
        '''
        Store transition in buffer
        
        Attributes
        ----------
        - local_obs : current local observations [num_lights, obs_dim]
        - actions : actions taken [num_lights]
        - rewards : rewards received [num_lights]
        - values : state values [num_lights]
        - log_probs : log probabilities [num_lights]
        - dones : whether episode is done [num_lights]
        '''
        self.buffers['observations'].append(local_obs.copy())
        self.buffers['actions'].append(actions.copy())
        self.buffers['rewards'].append(rewards.copy())
        self.buffers['values'].append(values.copy())
        self.buffers['log_probs'].append(log_probs.copy())
        self.buffers['dones'].append(dones.copy())
    
    def compute_advantages(self, last_values, last_dones):
        '''
        Compute advantages using GAE
        
        Attributes
        ----------
        - last_values : values of last states [num_lights]
        - last_dones : whether last states are terminal [num_lights]
        '''
        buffer_length = len(self.buffers['rewards'])
        
        # Initialize advantage and return arrays
        advantages = np.zeros((buffer_length, self.num_lights))
        returns = np.zeros((buffer_length, self.num_lights))
        
        # Compute advantages for each agent
        for i in range(self.num_lights):
            last_advantage = 0
            last_return = last_values[i]
            
            # Reverse traversal
            for t in reversed(range(buffer_length)):
                if t == buffer_length - 1:
                    next_non_terminal = 1.0 - last_dones[i]
                    next_value = last_values[i]
                else:
                    next_non_terminal = 1.0 - self.buffers['dones'][t+1][i]
                    next_value = self.buffers['values'][t+1][i]
                
                delta = self.buffers['rewards'][t][i] + self.discount * next_value * next_non_terminal - self.buffers['values'][t][i]
                advantage = delta + self.discount * self.gae_lambda * next_non_terminal * last_advantage
                
                advantages[t, i] = advantage
                returns[t, i] = advantage + self.buffers['values'][t][i]
                
                last_advantage = advantage
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        self.buffers['advantages'] = advantages
        self.buffers['returns'] = returns
    
    def train(self):
        '''
        Train MAPPO networks using collected experiences
        '''
        if len(self.buffers['rewards']) < self.batch_size:
            return 0, 0, 0  # Not enough samples
        
        buffer_length = len(self.buffers['rewards'])
        
        # Convert buffers to arrays
        observations = np.array(self.buffers['observations'])  # [buffer_length, num_lights, obs_dim]
        actions = np.array(self.buffers['actions'])  # [buffer_length, num_lights]
        old_log_probs = np.array(self.buffers['log_probs'])  # [buffer_length, num_lights]
        advantages = self.buffers['advantages']  # [buffer_length, num_lights]
        returns = self.buffers['returns']  # [buffer_length, num_lights]
        
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        
        # Train for multiple epochs
        for epoch in range(self.num_epochs):
            # Create random indices for mini-batch sampling
            indices = np.random.permutation(buffer_length)
            
            # Mini-batch training
            for start_idx in range(0, buffer_length, self.batch_size):
                end_idx = min(start_idx + self.batch_size, buffer_length)
                batch_indices = indices[start_idx:end_idx]
                
                # Train each agent
                for i in range(self.num_lights):
                    # Get batch data for agent i
                    obs_batch = torch.FloatTensor(observations[batch_indices, i, :])
                    actions_batch = torch.LongTensor(actions[batch_indices, i])
                    old_log_probs_batch = torch.FloatTensor(old_log_probs[batch_indices, i])
                    advantages_batch = torch.FloatTensor(advantages[batch_indices, i])
                    returns_batch = torch.FloatTensor(returns[batch_indices, i])
                    
                    # 1. Update actor
                    self.actor_optimizers[i].zero_grad()
                    
                    # Get current policy
                    action_probs = self.actors[i](obs_batch)
                    action_dist = torch.distributions.Categorical(action_probs)
                    
                    # New log probabilities
                    new_log_probs = action_dist.log_prob(actions_batch)
                    
                    # Ratio for PPO clip
                    ratio = torch.exp(new_log_probs - old_log_probs_batch)
                    
                    # Surrogate loss
                    surr1 = ratio * advantages_batch
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages_batch
                    
                    # Actor loss with entropy bonus
                    actor_loss = -torch.min(surr1, surr2).mean()
                    entropy = action_dist.entropy().mean()
                    total_actor_loss += actor_loss.item()
                    total_entropy += entropy.item()
                    
                    # Total loss with entropy regularization
                    loss = actor_loss - self.entropy_coef * entropy
                    loss.backward()
                    self.actor_optimizers[i].step()
                    
                    # 2. Update critic
                    self.critic_optimizers[i].zero_grad()
                    
                    # Get current value estimates
                    values = self.critics[i](obs_batch).squeeze()
                    
                    # Critic loss (MSE between value estimates and returns)
                    critic_loss = F.mse_loss(values, returns_batch)
                    total_critic_loss += critic_loss.item()
                    
                    critic_loss.backward()
                    self.critic_optimizers[i].step()
        
        # Reset buffers after training
        self.reset_buffers()
        
        # Average losses
        num_updates = self.num_epochs * (buffer_length // self.batch_size + 1) * self.num_lights
        avg_actor_loss = total_actor_loss / num_updates if num_updates > 0 else 0
        avg_critic_loss = total_critic_loss / num_updates if num_updates > 0 else 0
        avg_entropy = total_entropy / num_updates if num_updates > 0 else 0
        
        return avg_actor_loss, avg_critic_loss, avg_entropy
    
    def save_models(self, path):
        '''
        Save model parameters
        
        Attributes
        ----------
        - path : directory to save models
        '''
        os.makedirs(path, exist_ok=True)
        
        for i in range(self.num_lights):
            torch.save(self.actors[i].state_dict(), os.path.join(path, f'actor_{i}.pth'))
            torch.save(self.critics[i].state_dict(), os.path.join(path, f'critic_{i}.pth'))
    
    def load_models(self, path):
        '''
        Load model parameters
        
        Attributes
        ----------
        - path : directory to load models from
        '''
        for i in range(self.num_lights):
            actor_path = os.path.join(path, f'actor_{i}.pth')
            critic_path = os.path.join(path, f'critic_{i}.pth')
            
            if os.path.exists(actor_path):
                self.actors[i].load_state_dict(torch.load(actor_path))
            if os.path.exists(critic_path):
                self.critics[i].load_state_dict(torch.load(critic_path))


class Actor(nn.Module):
    '''
    Actor network for MAPPO
    '''
    def __init__(self, n_rows, n_cols, lr=0.0003, agent_id=0):
        super(Actor, self).__init__()
        self.num_lights = n_rows * n_cols
        self.agent_id = agent_id
        
        # Input: local observation (18 features + one-hot agent ID)
        self.input_dim = 18 + self.num_lights
        self.output_dim = 2  # 2 actions: switch or not
        
        # Actor network
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, self.output_dim)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        
        # Initialize biases
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, x):
        '''
        Forward pass
        
        Attributes
        ----------
        - x : input tensor [batch_size, input_dim]
        
        Returns
        -------
        - action_probs : action probabilities [batch_size, output_dim]
        '''
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs


class Critic(nn.Module):
    '''
    Critic network for MAPPO
    '''
    def __init__(self, n_rows, n_cols, lr=0.0003, agent_id=0):
        super(Critic, self).__init__()
        self.num_lights = n_rows * n_cols
        self.agent_id = agent_id
        
        # Input: local observation
        self.input_dim = 18 + self.num_lights
        
        # Critic network
        self.fc1 = nn.Linear(self.input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # Initialize weights
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        
        # Initialize biases
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
    
    def forward(self, x):
        '''
        Forward pass
        
        Attributes
        ----------
        - x : input tensor [batch_size, input_dim]
        
        Returns
        -------
        - value : state value [batch_size, 1]
        '''
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        value = self.fc3(x)
        return value