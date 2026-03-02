import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
import os
from copy import deepcopy
import datetime
from .risk_net import RiskNet

class QCOMBO_GREEDY_RISK:
    '''
    QCOMBO with Greedy Rule Fusion and Risk Awareness
    '''
    def __init__(self, n_rows, n_cols, config):
        self.name = "QCOMBO_GREEDY_RISK"
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.num_lights = n_rows * n_cols
        self.discount = config.alg.discount
        self.qcombo_lam = config.alg.qcombo_lam
        self.lam = config.alg.lam
        self.exploration_rate = config.alg.exploration_rate
        self.config = config
        
        # Greedy rule parameters
        self.greedy_epsilon = getattr(config.alg, 'greedy_epsilon', 0.3)
        self.risk_weight = getattr(config.alg, 'risk_weight', 0.5)
        self.adaptive_greedy = getattr(config.alg, 'adaptive_greedy', True)
        
        # Initialize networks
        from .qcombo import LocalNet, GlobalNet
        self.local_net = LocalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)
        self.global_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)
        self.local_target_net = LocalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)
        self.global_target_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)
        self.local_target_net.eval()
        self.global_target_net.eval()
        self.global_copy_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount, config=config)
        
        # Initialize Risk Network
        obs_dim = n_rows * n_cols * 10  # Approximate observation dimension
        self.risk_net = RiskNet(input_dim=obs_dim, hidden_dims=[128, 64], lr=config.critic.lr)
        
    def get_greedy_rule_action(self, local_obs):
        '''
        Greedy rule: Select action based on queue length heuristic
        Choose green for direction with longest queue
        '''
        actions = []
        for i in range(self.num_lights):
            # Extract queue length from observation (assuming it's in the first few dims)
            state = local_obs[:, i, :]
            # Simple heuristic: if queue length > threshold, switch phase
            queue_len = torch.sum(state[:, :4], dim=1)  # Sum first 4 dims as queue proxy
            action = (queue_len > 5).long()  # Switch if queue > 5
            actions.append(action)
        return torch.stack(actions, dim=1)
    
    def get_risk_adjusted_q(self, local_obs, global_obs):
        '''
        Get Q-values adjusted by risk assessment
        Q_adjusted = Q - risk_weight * Risk
        '''
        batch_size = local_obs.shape[0]
        global_flat = global_obs.view(batch_size, -1)
        
        # Compute risk
        risk = self.risk_net.get_risk(global_flat)  # [batch_size, 1]
        
        # Get Q values for each light
        adjusted_q_list = []
        for i in range(self.num_lights):
            q_values = self.local_net(local_obs[:, i, :])  # [batch_size, 2]
            # Adjust Q by risk
            risk_penalty = self.risk_weight * risk
            adjusted_q = q_values - risk_penalty
            adjusted_q_list.append(adjusted_q)
        
        return adjusted_q_list, risk
    
    def select_action(self, local_obs, global_obs, training=True):
        '''
        Select action using ε-greedy mix of rule-based and RL policy
        with risk awareness
        '''
        batch_size = local_obs.shape[0]
        
        # Adaptive epsilon decay
        if training and self.adaptive_greedy:
            current_epsilon = max(0.1, self.greedy_epsilon * (0.995 ** self.config.main.train_iters))
        else:
            current_epsilon = self.greedy_epsilon if training else 0
        
        # Decide whether to use greedy rule or RL
        use_greedy = random.random() < current_epsilon
        
        if use_greedy:
            # Use rule-based greedy action
            actions = self.get_greedy_rule_action(local_obs)
        else:
            # Use risk-adjusted RL policy
            adjusted_q_list, _ = self.get_risk_adjusted_q(local_obs, global_obs)
            actions = torch.stack([torch.argmax(q, dim=1) for q in adjusted_q_list], dim=1)
        
        # Convert to global action
        binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1) for i in range(self.num_lights)])
        global_action = torch.matmul(actions.float(), binary_coeff)
        
        return actions, global_action.long()
    
    def compute_risk_reward(self, state, next_state, reward):
        '''
        Compute risk-based auxiliary reward
        Penalize high-risk transitions
        '''
        with torch.no_grad():
            current_risk = self.risk_net.get_risk(state)
            next_risk = self.risk_net.get_risk(next_state)
            risk_penalty = 0.1 * (next_risk - current_risk)
            adjusted_reward = reward - risk_penalty.squeeze()
        return adjusted_reward
    
    def train_step(self, replay, summarize=True):
        '''Training step with risk-aware loss'''
        total_loss = 0
        for i in range(self.config.alg.num_minibatches):
            actions, global_reward, old_local_obs, old_global_obs, new_local_obs, new_global_obs, local_rewards = \
                replay.sample(self.config.alg.minibatch_size)
            
            # Standard QCOMBO losses
            individual_loss = self.local_net.get_loss(
                old_state=old_local_obs, new_state=new_local_obs, actions=actions, rewards=local_rewards)
            
            _, global_greedy_actions = self.select_action(new_local_obs, new_global_obs, training=False)
            global_loss = self.global_net.get_loss(
                old_global_state=old_global_obs, new_global_state=new_global_obs,
                reward=global_reward, actions=actions, greedy_actions=global_greedy_actions)
            
            reg_loss = self._get_reg_loss(global_obs=old_global_obs, local_obs=old_local_obs, actions=actions)
            
            # Risk-aware adjustment
            batch_size = old_global_obs.shape[0]
            old_global_flat = old_global_obs.view(batch_size, -1)
            new_global_flat = new_global_obs.view(batch_size, -1)
            adjusted_reward = self.compute_risk_reward(old_global_flat, new_global_flat, global_reward)
            
            # Combined loss
            loss = individual_loss + global_loss + self.qcombo_lam * reg_loss
            
            if self.config.alg.perturb:
                adv_reg_loss = self._get_adv_reg_loss(old_global_obs)
                loss = loss + self.lam * adv_reg_loss
            
            # Backpropagation
            self.local_net.optimizer.zero_grad()
            self.global_net.optimizer.zero_grad()
            self.risk_net.optimizer.zero_grad()
            
            loss.backward()
            
            self.local_net.optimizer.step()
            self.global_net.optimizer.step()
            self.risk_net.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / self.config.alg.num_minibatches
    
    def _get_reg_loss(self, global_obs, local_obs, actions):
        '''Regularization loss'''
        binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1) for i in range(self.num_lights)])
        global_actions = torch.matmul(actions.float(), binary_coeff)
        global_Q = self.global_net(global_obs)
        global_Q_taken = global_Q[torch.arange(global_obs.shape[0]), global_actions.long()]
        
        local_Q = torch.zeros(global_obs.shape[0])
        for i in range(self.num_lights):
            local_obs_tensor = local_obs[:, i, :]
            local_actions = actions[:, i]
            Q = self.local_net(local_obs_tensor)
            Q_taken = Q[torch.arange(global_obs.shape[0]), local_actions.long()]
            local_Q += Q_taken
        local_Q /= self.num_lights
        
        return self.local_net.loss_function(local_Q, global_Q_taken)
    
    def _get_adv_reg_loss(self, state_tensor):
        '''Adversarial regularization'''
        perturbation = torch.normal(torch.zeros_like(state_tensor), torch.ones_like(state_tensor) * 1e-3)
        perturbed_tensor = state_tensor + perturbation * torch.abs(state_tensor.detach())
        normal_Q = self.global_net(state_tensor)
        perturbed_Q = self.global_net(perturbed_tensor)
        return torch.norm(normal_Q - perturbed_Q, p="fro")
    
    def save(self, path):
        '''Save model checkpoints'''
        torch.save({
            'local_net': self.local_net.state_dict(),
            'global_net': self.global_net.state_dict(),
            'risk_net': self.risk_net.state_dict(),
        }, path)
    
    def load(self, path):
        '''Load model checkpoints'''
        checkpoint = torch.load(path)
        self.local_net.load_state_dict(checkpoint['local_net'])
        self.global_net.load_state_dict(checkpoint['global_net'])
        self.risk_net.load_state_dict(checkpoint['risk_net'])
