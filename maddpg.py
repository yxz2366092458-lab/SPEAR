import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
import os
from copy import deepcopy

class MADDPG:
    '''
    Implementation of MADDPG algorithm for multi-agent traffic light control
    Based on: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
    '''
    def __init__(self, n_rows, n_cols, config):
        '''
        Attributes
        ----------
        - n_rows : the number of rows in the grid
        - n_cols : the number of columns in the grid
        - config : configuration object
        '''
        self.name = "MADDPG"
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.num_lights = n_rows * n_cols
        self.discount = config.alg.discount
        self.tau = config.alg.tau if hasattr(config.alg, 'tau') else 0.01  # soft update parameter
        self.lr_actor = config.alg.lr_actor if hasattr(config.alg, 'lr_actor') else 0.001
        self.lr_critic = config.alg.lr_critic if hasattr(config.alg, 'lr_critic') else 0.001
        self.config = config
        
        # Initialize actors and critics for each agent
        self.actors = []
        self.critics = []
        self.actor_targets = []
        self.critic_targets = []
        self.actor_optimizers = []
        self.critic_optimizers = []
        
        for i in range(self.num_lights):
            # Actor network (policy)
            actor = Actor(n_rows, n_cols, lr=self.lr_actor, agent_id=i)
            actor_target = Actor(n_rows, n_cols, lr=self.lr_actor, agent_id=i)
            actor_target.load_state_dict(actor.state_dict())
            
            # Critic network (Q-function)
            critic = Critic(n_rows, n_cols, lr=self.lr_critic, agent_id=i)
            critic_target = Critic(n_rows, n_cols, lr=self.lr_critic, agent_id=i)
            critic_target.load_state_dict(critic.state_dict())
            
            self.actors.append(actor)
            self.critics.append(critic)
            self.actor_targets.append(actor_target)
            self.critic_targets.append(critic_target)
            
            self.actor_optimizers.append(optim.Adam(actor.parameters(), lr=self.lr_actor))
            self.critic_optimizers.append(optim.Adam(critic.parameters(), lr=self.lr_critic))
        
        # Replay buffer (shared among agents)
        self.replay_buffer = []
        self.buffer_size = config.alg.buffer_size if hasattr(config.alg, 'buffer_size') else 10000
        self.batch_size = config.alg.batch_size if hasattr(config.alg, 'batch_size') else 32
        
        # Exploration noise
        self.noise_scale = config.alg.noise_scale if hasattr(config.alg, 'noise_scale') else 0.1
        self.noise_decay = config.alg.noise_decay if hasattr(config.alg, 'noise_decay') else 0.999
        
    def select_action(self, local_obs, explore=True):
        '''
        Select actions for all agents
        
        Attributes
        ----------
        - local_obs : local observations for each agent [num_lights, obs_dim]
        - explore : whether to add exploration noise
        
        Returns
        -------
        - actions : selected actions for each agent [num_lights]
        '''
        actions = []
        
        for i in range(self.num_lights):
            obs_tensor = torch.FloatTensor(local_obs[i]).unsqueeze(0)
            
            with torch.no_grad():
                action_probs = self.actors[i](obs_tensor)
                action = torch.argmax(action_probs, dim=1).item()
            
            # Add exploration noise
            if explore and random.random() < self.noise_scale:
                action = random.randint(0, 1)  # 0 or 1 for traffic light
            
            actions.append(action)
        
        return np.array(actions)
    
    def store_transition(self, local_obs, actions, rewards, next_local_obs, done):
        '''
        Store transition in replay buffer
        
        Attributes
        ----------
        - local_obs : current local observations [num_lights, obs_dim]
        - actions : actions taken [num_lights]
        - rewards : rewards received [num_lights]
        - next_local_obs : next local observations [num_lights, obs_dim]
        - done : whether episode is done
        '''
        transition = {
            'local_obs': local_obs.copy(),
            'actions': actions.copy(),
            'rewards': rewards.copy(),
            'next_local_obs': next_local_obs.copy(),
            'done': done
        }
        
        self.replay_buffer.append(transition)
        
        # Maintain buffer size
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def train(self):
        '''
        Train MADDPG networks using experience replay
        '''
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0  # Not enough samples
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        total_actor_loss = 0
        total_critic_loss = 0
        
        # Convert batch to tensors
        local_obs_batch = np.array([t['local_obs'] for t in batch])  # [batch, num_lights, obs_dim]
        actions_batch = np.array([t['actions'] for t in batch])  # [batch, num_lights]
        rewards_batch = np.array([t['rewards'] for t in batch])  # [batch, num_lights]
        next_local_obs_batch = np.array([t['next_local_obs'] for t in batch])  # [batch, num_lights, obs_dim]
        done_batch = np.array([t['done'] for t in batch])  # [batch]
        
        # Train each agent
        for i in range(self.num_lights):
            # Prepare tensors for agent i
            obs_i = torch.FloatTensor(local_obs_batch[:, i, :])  # [batch, obs_dim]
            actions_i = torch.LongTensor(actions_batch[:, i])  # [batch]
            rewards_i = torch.FloatTensor(rewards_batch[:, i])  # [batch]
            next_obs_i = torch.FloatTensor(next_local_obs_batch[:, i, :])  # [batch, obs_dim]
            done_i = torch.FloatTensor(done_batch)  # [batch]
            
            # Get actions of all other agents for critic input
            other_actions = []
            for j in range(self.num_lights):
                if j != i:
                    other_actions.append(torch.LongTensor(actions_batch[:, j]))
            
            # 1. Update critic
            self.critic_optimizers[i].zero_grad()
            
            # Current Q-value
            current_Q = self.critics[i](obs_i, actions_i.unsqueeze(1))
            
            # Target Q-value
            with torch.no_grad():
                # Target actions from target actor
                target_action_probs = self.actor_targets[i](next_obs_i)
                target_actions = torch.argmax(target_action_probs, dim=1)
                
                # Target Q-value
                target_Q = self.critic_targets[i](next_obs_i, target_actions.unsqueeze(1))
                target_Q = rewards_i + (1 - done_i) * self.discount * target_Q.squeeze()
            
            # Critic loss
            critic_loss = F.mse_loss(current_Q.squeeze(), target_Q)
            critic_loss.backward()
            self.critic_optimizers[i].step()
            
            # 2. Update actor
            self.actor_optimizers[i].zero_grad()
            
            # Actor loss: maximize Q-value
            action_probs = self.actors[i](obs_i)
            actions_pred = torch.argmax(action_probs, dim=1)
            actor_loss = -self.critics[i](obs_i, actions_pred.unsqueeze(1)).mean()
            
            actor_loss.backward()
            self.actor_optimizers[i].step()
            
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            
            # 3. Soft update target networks
            self.soft_update(self.actors[i], self.actor_targets[i])
            self.soft_update(self.critics[i], self.critic_targets[i])
        
        # Decay exploration noise
        self.noise_scale *= self.noise_decay
        
        return total_actor_loss / self.num_lights, total_critic_loss / self.num_lights
    
    def soft_update(self, local_model, target_model):
        '''
        Soft update model parameters:
        θ_target = τ * θ_local + (1 - τ) * θ_target
        
        Attributes
        ----------
        - local_model : source model
        - target_model : target model
        '''
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
    
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
            torch.save(self.actor_targets[i].state_dict(), os.path.join(path, f'actor_target_{i}.pth'))
            torch.save(self.critic_targets[i].state_dict(), os.path.join(path, f'critic_target_{i}.pth'))
    
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
            actor_target_path = os.path.join(path, f'actor_target_{i}.pth')
            critic_target_path = os.path.join(path, f'critic_target_{i}.pth')
            
            if os.path.exists(actor_path):
                self.actors[i].load_state_dict(torch.load(actor_path))
            if os.path.exists(critic_path):
                self.critics[i].load_state_dict(torch.load(critic_path))
            if os.path.exists(actor_target_path):
                self.actor_targets[i].load_state_dict(torch.load(actor_target_path))
            if os.path.exists(critic_target_path):
                self.critic_targets[i].load_state_dict(torch.load(critic_target_path))


class Actor(nn.Module):
    '''
    Actor network for MADDPG
    '''
    def __init__(self, n_rows, n_cols, lr=0.001, agent_id=0):
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
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
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
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs


class Critic(nn.Module):
    '''
    Critic network for MADDPG
    '''
    def __init__(self, n_rows, n_cols, lr=0.001, agent_id=0):
        super(Critic, self).__init__()
        self.num_lights = n_rows * n_cols
        self.agent_id = agent_id
        
        # Input: local observation + action
        self.obs_dim = 18 + self.num_lights
        self.action_dim = 1  # action index
        self.input_dim = self.obs_dim + self.action_dim
        
        # Critic network
        self.fc1 = nn.Linear(self.input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
    
    def forward(self, obs, action):
        '''
        Forward pass
        
        Attributes
        ----------
        - obs : observation tensor [batch_size, obs_dim]
        - action : action tensor [batch_size, 1]
        
        Returns
        -------
        - Q_value : Q-value [batch_size, 1]
        '''
        # Ensure action is float
        if action.dtype != torch.float32:
            action = action.float()
        
        x = torch.cat([obs, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        Q_value = self.fc3(x)
        return Q_value