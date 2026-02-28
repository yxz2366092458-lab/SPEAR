import torch
from torch import nn
import numpy as np
import random
import torch.nn.functional as F
import torch.optim as optim
import os
from copy import deepcopy
import datetime
import higher


class QCOMBOS:
    '''
    Implementation of QCOMBO algorithm
    '''

    def __init__(self, n_rows, n_cols, config):
        '''
        Attributes
        ----------
        - n_rows : the number of rows in the grid
        - n_cols : the number of columns in the grid
        - lr : the learning rate
        - discount : the discount factor in the TD target
        '''
        self.name = "QCOMBO"
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.num_lights = n_rows * n_cols
        self.discount = config.alg.discount
        self.qcombo_lam = config.alg.qcombo_lam  # Regularization coefficient
        self.lam = config.alg.lam
        self.exploration_rate = config.alg.exploration_rate
        self.config = config

        # Stackelberg对抗训练参数
        self.perturb_epsilon = config.alg.get('perturb_epsilon', 1e-3)  # 扰动幅度
        self.perturb_steps = config.alg.get('perturb_num_steps', 10)  # 扰动优化步数
        self.perturb_lr = config.alg.get('perturb_alpha', 0.01)  # 扰动学习率

        # 初始化网络
        self.local_net = LocalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount,
                                  config=config)
        self.global_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount,
                                    config=config)

        # 初始化目标网络
        self.local_target_net = LocalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount,
                                         config=config)
        self.global_target_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount,
                                           config=config)
        self.local_target_net.eval()
        self.global_target_net.eval()

        # 初始化第二个全局网络用于有限差分方法
        self.global_copy_net = GlobalNet(n_rows=n_rows, n_cols=n_cols, lr=config.critic.lr, discount=self.discount,
                                         config=config)

        # 对抗训练相关的缓存
        self.perturbation_buffer = []

    def get_greedy(self, local_obs):
        '''
        获取贪婪动作
        '''
        actions_list = []
        for i in range(self.num_lights):
            state_tensor = local_obs[:, i, :]
            Q = self.local_net(state_tensor)  # [batch_size, 2]
            actions = torch.argmax(Q, dim=1)  # [batch_size]
            actions_list.append(actions)

        greedy_local_actions = torch.stack(actions_list, dim=1)  # [batch_size, num_lights]
        binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1) for i in range(self.num_lights)])  # [num_lights]
        global_action = torch.matmul(greedy_local_actions, binary_coeff.long())  # [batch_size]
        return global_action

    def get_reg_loss(self, global_obs, local_obs, actions, copy_net=False):
        '''
        获取正则化损失
        '''
        binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1) for i in range(self.num_lights)])
        global_actions = torch.matmul(actions, binary_coeff)

        if copy_net:
            global_Q = self.global_copy_net(global_obs)  # [batch_size, 2 ** num_lights]
        else:
            global_Q = self.global_net(global_obs)  # [batch_size, 2 ** num_lights]

        global_Q_taken = global_Q[torch.arange(global_obs.shape[0]), global_actions.long()]

        local_Q = torch.zeros(global_obs.shape[0])
        for i in range(self.num_lights):
            local_obs_tensor = local_obs[:, i, :]
            local_actions = actions[:, i]
            Q = self.local_net(local_obs_tensor)  # [batch_size, 2]
            Q_taken = Q[torch.arange(global_obs.shape[0]), local_actions.long()]  # [batch_size]
            local_Q += Q_taken
        local_Q /= self.num_lights
        loss = self.local_net.loss_function(local_Q, global_Q_taken)
        return loss

    def get_adv_reg_loss(self, state_tensor, perturbed_state_tensor):
        '''
        获取对抗正则化损失（改进版）

        改进点：
        1. 使用平滑性正则化，鼓励模型对小的扰动不敏感
        2. 添加Lipschitz约束，防止梯度爆炸
        '''
        normal_Q = self.global_net(state_tensor)
        perturbed_Q = self.global_net(perturbed_state_tensor)

        # 计算L2距离
        q_diff = normal_Q - perturbed_Q
        reg_loss = torch.norm(q_diff, p=2)

        # 添加Lipschitz约束（防止梯度爆炸）
        lipschitz_penalty = torch.relu(
            torch.norm(q_diff, p=2) / torch.norm(state_tensor - perturbed_state_tensor, p=2) - 1.0)

        return reg_loss + 0.1 * lipschitz_penalty

    def stackelberg_adv_training(self, old_global_obs, epsilon=1e-3, steps=10, lr=0.01):
        '''
        Stackelberg对抗训练（改进版）

        改进点：
        1. 使用投影梯度下降（Projected Gradient Descent, PGD）确保扰动在合理范围内
        2. 添加动量项提高优化效率
        3. 使用多个随机起点避免局部最优
        4. 缓存历史扰动，提高训练效率
        '''
        batch_size = old_global_obs.shape[0]

        # 如果有缓存扰动，先使用缓存
        if len(self.perturbation_buffer) > 0:
            cached_perturb = random.choice(self.perturbation_buffer)
            if cached_perturb.shape == old_global_obs.shape:
                perturbation = cached_perturb.clone().detach()
            else:
                perturbation = torch.zeros_like(old_global_obs)
        else:
            perturbation = torch.zeros_like(old_global_obs)

        perturbation.requires_grad_(True)

        # 使用多个随机起点
        best_perturbation = perturbation.clone()
        best_loss = -float('inf')

        for restart in range(3):  # 3个随机起点
            if restart > 0:
                # 随机初始化扰动
                perturbation = torch.randn_like(old_global_obs) * epsilon * 0.1
                perturbation.requires_grad_(True)

            momentum = torch.zeros_like(perturbation)

            for step in range(steps):
                # 计算对抗损失
                perturbed_state = old_global_obs + perturbation
                perturbed_Q = self.global_net(perturbed_state)
                normal_Q = self.global_net(old_global_obs).detach()

                # 计算损失（最大化扰动前后Q值的差异）
                loss = -torch.norm(perturbed_Q - normal_Q, p=2)

                # 计算梯度
                grad = torch.autograd.grad(loss, perturbation,
                                           retain_graph=True,
                                           create_graph=True)[0]

                # 使用动量
                momentum = 0.9 * momentum + grad

                # 更新扰动（梯度上升）
                perturbation = perturbation + lr * momentum.sign()

                # 投影到epsilon球内（确保扰动幅度不超过epsilon）
                perturbation_norm = torch.norm(perturbation, p=float('inf'), dim=1, keepdim=True)
                perturbation = perturbation / torch.clamp(perturbation_norm / epsilon, min=1.0)

                perturbation = perturbation.detach().requires_grad_(True)

                # 记录最佳扰动
                current_loss = -loss.item()
                if current_loss > best_loss:
                    best_loss = current_loss
                    best_perturbation = perturbation.clone()

        # 缓存最佳扰动
        self.perturbation_buffer.append(best_perturbation.clone())
        if len(self.perturbation_buffer) > 10:  # 保持缓存大小
            self.perturbation_buffer.pop(0)

        return best_perturbation

    def compute_smoothness_loss(self, state_tensor, perturbed_state_tensor):
        '''
        计算平滑性损失
        鼓励模型对小的输入扰动具有鲁棒性
        '''
        # 计算一阶梯度平滑性
        normal_Q = self.global_net(state_tensor)
        perturbed_Q = self.global_net(perturbed_state_tensor)

        # 计算输出差异
        q_diff = torch.norm(normal_Q - perturbed_Q, p=2)

        # 计算梯度正则化
        state_tensor.requires_grad_(True)
        q_output = self.global_net(state_tensor)

        # 计算梯度范数
        grad_norms = []
        for i in range(q_output.shape[1]):
            grad = torch.autograd.grad(q_output[:, i].sum(), state_tensor,
                                       retain_graph=True, create_graph=True)[0]
            grad_norms.append(torch.norm(grad, p=2))

        avg_grad_norm = torch.mean(torch.stack(grad_norms))

        # 组合损失：输出差异 + 梯度正则化
        smoothness_loss = q_diff + 0.01 * avg_grad_norm

        return smoothness_loss

    def train_step(self, replay, summarize=True):
        '''
        训练步骤（改进版）
        '''
        import datetime
        start = datetime.datetime.now()

        for i in range(self.config.alg.num_minibatches):
            # 采样数据
            actions, global_reward, old_local_obs, old_global_obs, new_local_obs, new_global_obs, local_rewards = \
                replay.sample(self.config.alg.minibatch_size)

            # 基础损失
            individual_loss = self.local_net.get_loss(
                old_state=old_local_obs, new_state=new_local_obs, actions=actions, rewards=local_rewards)
            greedy_actions = self.get_greedy(new_local_obs)
            global_loss = self.global_net.get_loss(old_global_state=old_global_obs,
                                                   new_global_state=new_global_obs, reward=global_reward,
                                                   actions=actions,
                                                   greedy_actions=greedy_actions)
            reg_loss = self.get_reg_loss(global_obs=old_global_obs, local_obs=old_local_obs,
                                         actions=actions)

            loss = individual_loss + global_loss + self.qcombo_lam * reg_loss

            # 对抗训练
            if self.config.alg.perturb:
                if self.config.alg.stackelberg:
                    # 使用改进的Stackelberg对抗训练
                    perturbation = self.stackelberg_adv_training(
                        old_global_obs,
                        epsilon=self.perturb_epsilon,
                        steps=self.perturb_steps,
                        lr=self.perturb_lr
                    )

                    # 计算对抗损失
                    perturbed_tensor = old_global_obs + perturbation
                    adv_reg_loss = self.get_adv_reg_loss(old_global_obs, perturbed_tensor)

                    # 计算平滑性损失
                    smooth_loss = self.compute_smoothness_loss(old_global_obs, perturbed_tensor)

                    # 总损失
                    total_loss = loss + self.lam * adv_reg_loss + 0.1 * smooth_loss

                    # 优化
                    self.local_net.optimizer.zero_grad()
                    self.global_net.optimizer.zero_grad()

                    total_loss.backward()

                    # 梯度裁剪（防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), max_norm=1.0)
                    torch.nn.utils.clip_grad_norm_(self.global_net.parameters(), max_norm=1.0)

                    self.local_net.optimizer.step()
                    self.global_net.optimizer.step()

                else:
                    # Get the regularization loss for a smooth policy

                    perturbed_tensor = old_global_obs + torch.normal(torch.zeros_like(old_global_obs),
                                                                     torch.ones_like(old_global_obs) * 1e-3)
                    perturbed_tensor.requires_grad = True
                    obs_grad = torch.zeros(perturbed_tensor.shape)

                    for k in range(self.config.alg.perturb_num_steps):
                        # Calculate adversarial perurbation
                        distance_loss = torch.norm(self.global_net(old_global_obs) - self.global_net(perturbed_tensor),
                                                   p="fro")
                        grad = torch.autograd.grad(outputs=distance_loss, inputs=perturbed_tensor,
                                                   grad_outputs=torch.ones_like(loss), retain_graph=True,
                                                   create_graph=True)[0]
                        obs_grad = grad
                        perturbed_tensor = perturbed_tensor + self.config.alg.perturb_alpha * grad * torch.abs(
                            old_global_obs.detach())

                    adv_reg_loss = self.get_adv_reg_loss(old_global_obs, perturbed_tensor)
                    loss = loss + self.lam * adv_reg_loss

                    # Set gradients to zero
                    self.local_net.optimizer.zero_grad()
                    self.global_net.optimizer.zero_grad()

                    loss.backward()
                    self.local_net.optimizer.step()
                    self.global_net.optimizer.step()
            else:
                # 无对抗训练的标准优化
                self.local_net.optimizer.zero_grad()
                self.global_net.optimizer.zero_grad()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.global_net.parameters(), max_norm=1.0)

                self.local_net.optimizer.step()
                self.global_net.optimizer.step()

        if summarize:
            end = datetime.datetime.now()
            print(f"Training step took {(end - start).total_seconds():.2f} seconds")

    # 移除旧的unroll_perturb方法，使用新的stackelberg_adv_training替代

    def choose_action(self, new_state):
        '''
        选择动作
        '''
        if np.random.uniform() < self.exploration_rate:
            decision = np.random.randint(0, 2)
            self.exploration_rate *= self.config.alg.anneal_exp
        else:
            Q = self.local_net(new_state)
            decision = int(torch.argmax(Q))

        return decision

    def update_targets(self):
        '''
        更新目标网络
        '''
        # 使用软更新（soft update）而不是硬更新
        tau = self.config.get('target_update_tau', 0.001)

        for target_param, param in zip(self.local_target_net.parameters(), self.local_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        for target_param, param in zip(self.global_target_net.parameters(), self.global_net.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save(self, dir, model_id=None):
        '''
        保存模型
        '''
        self.local_net.save(dir, model_id)
        self.global_net.save(dir, model_id)

        # 保存对抗训练缓存
        if len(self.perturbation_buffer) > 0:
            cache_path = os.path.join(dir, f'perturb_cache_{model_id}.pt' if model_id else 'perturb_cache.pt')
            torch.save(self.perturbation_buffer, cache_path)

    def load(self, dir, model_id=None):
        '''
        加载模型
        '''
        self.local_net.load(dir, model_id)
        self.global_net.load(dir, model_id)

        # 加载对抗训练缓存
        cache_path = os.path.join(dir, f'perturb_cache_{model_id}.pt' if model_id else 'perturb_cache.pt')
        if os.path.exists(cache_path):
            self.perturbation_buffer = torch.load(cache_path)

# LocalNet和GlobalNet类保持不变，仅添加了上述改进

class LocalNet(nn.Module):
	'''
	Local neural network to carry out the individual part of QCOMBO
	'''
	def __init__(self, n_rows, n_cols, lr, discount, config):
		'''
		Attributes
		----------
		- n_rows : the number of rows in the grid
		- n_cols : the number of columns in the grid
		- lr : the learning rate
		- discount : the discount factor in the TD target
		'''
		super(LocalNet, self).__init__()
		self.num_lights = n_rows * n_cols
		self.n_rows = n_rows
		self.n_cols = n_cols
		self.lr = lr
		self.discount = discount
		self.config = config

		# Initialize the input and output counts
		self.input_count = 18 + self.num_lights
		self.output_count = 2

		# Initialize the neural network layers
		self.fc1 = nn.Linear(in_features=self.input_count, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=64)
		self.fc3 = nn.Linear(in_features=64, out_features=2)

		# Initialize the loss function
		self.loss_function = nn.MSELoss()

		# Initialize the optimzer
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		'''
		Forward pass of the neural network

		Attributes
		----------
		- x : the model input [batch_size, input_count]

		Returns
		-------
		- Q : the predicted Q function [batch_size, output_count]
		'''
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		Q = self.fc3(y2)

		return Q

	def get_loss(self, old_state, new_state, actions, rewards):
		'''
		Function to get the loss (TD error)

		Attributes
		----------
		- old_state : the old observation of the agent [batch_size, num_lights, input_count]
		- new_state : new observations of the agent [batch_size, num_lights, input_count]
		- actions : tensor of agent actions [batch_size, num_lights]
		- rewards : tensor of rewards recieved by the agent [batch_size, num_lights]

		Returns
		-------
		- loss : the loss of the Q-network
		'''
		total_loss = 0
		for i in range(self.num_lights):
			old_state_tensor = old_state[:, i, :]
			new_state_tensor = new_state[:, i, :]
			action = actions[:, i]
			reward = rewards[:, i]
			old_Q = self.forward(old_state_tensor) # [batch_size, output_count]
			Q_taken = old_Q[torch.arange(self.config.alg.minibatch_size), action.long()] # [batch_size]

			new_Q = self.forward(new_state_tensor) # [batch_size, 2]
			max_Q = torch.max(new_Q, dim=1)[0] # [batch_size]

			target = reward + self.discount * max_Q
			loss = self.loss_function(Q_taken, target)
			total_loss += loss

		return total_loss

	def save(self, dir, model_id=None):
		torch.save(self.state_dict(), os.path.join(dir, 'QCOMBO_local_{}.pt'.format(model_id)))

	def load(self, dir, model_id=None):
		self.load_state_dict(torch.load(os.path.join(dir, 'QCOMBO_local_{}.pt'.format(model_id))))


class GlobalNet(nn.Module):
	'''
	Global Q-network in QCOMBO algorithm
	'''
	def __init__(self, n_rows, n_cols, lr, discount, config):
		'''
		Attributes
		----------
		- n_rows : the number of rows in the grid
		- n_cols : the number of columns in the grid
		- lr : the learning rate
		- discount : the discount factor in the TD target
		'''
		super(GlobalNet, self).__init__()
		self.num_lights = n_rows * n_cols
		self.n_rows = n_rows
		self.n_cols=n_cols
		self.lr = lr
		self.discount = discount
		self.config = config

		# Initialize the input and output counts
		self.input_count = 18 * self.num_lights
		self.output_count = 2 ** self.num_lights

		# Initialize the network parameters
		self.fc1 = nn.Linear(in_features=self.input_count, out_features=64)
		self.fc2 = nn.Linear(in_features=64, out_features=64)
		self.fc3 = nn.Linear(in_features=64, out_features=self.output_count)

		# Initialize the loss function
		self.loss_function = nn.MSELoss()

		# Initialize the optimizer
		self.optimizer = optim.Adam(self.parameters(), lr=lr)

	def forward(self, x):
		'''
		Forward pass of the neural network

		Attributes
		----------
		- x : the model input [batch_size, input_count]

		Returns
		-------
		- Q : the predicted Q function [batch_size, output_count]
		'''
		y1 = F.relu(self.fc1(x))
		y2 = F.relu(self.fc2(y1))
		Q = self.fc3(y2)

		return Q

	def get_loss(self, old_global_state, new_global_state, reward, actions, greedy_actions):
		'''
		Function to get the global part of the QCOMBO loss

		Attributes
		----------
		- old_global_state : the old global observation [batch_size, input_count]
		- new_global_state : the new global observation [batch_size, num_lights, input_count]
		- reward : the global reward recieved [batch_size]
		- actions : actions taken by the agent [batch_size, num_lights]
		- greedy_actions : the greedy actions taken by the lights [batch_size]

		Returns
		-------
		- loss : the global loss function
		'''

		# First enumerate the global actions
		binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1) for i in range(self.num_lights)])
		global_action = torch.matmul(actions, binary_coeff)
		# Calculate the TD approximation
		old_Q = self.forward(old_global_state)
		Q_taken = old_Q[torch.arange(old_global_state.shape[0]), global_action.long()]
		# Calculate the TD target
		new_Q = self.forward(new_global_state) # [batch_size, 2 ** num_lights]

		greedy_Q = new_Q[torch.arange(old_global_state.shape[0]), greedy_actions.long()] # [batch_size]
		target = reward + self.discount * greedy_Q

		loss = self.loss_function(Q_taken, target)
		return loss

	def save(self, dir, model_id=None):
		torch.save(self.state_dict(), os.path.join(dir, 'QCOMBO_global_{}.pt'.format(model_id)))

	def load(self, dir, model_id=None):
		self.load_state_dict(torch.load(os.path.join(dir, 'QCOMBO_global_{}.pt'.format(model_id))))
