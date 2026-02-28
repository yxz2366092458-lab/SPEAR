"""Multi-agent traffic light"""
import numpy as np
import torch
import torch.optim as optim
from torch import nn

from Algorithms.qcombo_Stackelberg import QCOMBOS
from PBTManager import QCOMBO_PBT # 添加QCOMBO_PBT导入
from Env import TrafficGridEnv, TrafficAction
from PBTManager import PBTManager
from replay_buffer import ReplayBuffer
from Algorithms.qcombo import QCOMBO
from Algorithms.coma import COMA
from collections import deque
import argparse
import os
import time
from eval_configs import eval1, eval2, eval3, eval4, eval5, eval6, eval7, eval8, eval_4x4, eval_6x6_0, eval_6x6_1, \
    eval_6x6_2, eval_6x6_3, eval_6x6_4, train_2x2, eval_custom_0, eval_custom_1
import random

parser = argparse.ArgumentParser(description='RL Experiment.')
parser.add_argument('--alg', type=str, default='qcombo', choices=['qcombo', 'coma', 'qcombo_adv', 'PBT_adv'])
parser.add_argument('--nrow', type=int, default=2,
                    help='n_row of environment')
parser.add_argument('--ncol', type=int, default=2,
                    help='n_col of environment')
parser.add_argument('--pbt_pop_size', type=int, default=8,
                    help='PBT population size')
parser.add_argument('--pbt_generations', type=int, default=50,
                    help='Number of PBT generations')
parser.add_argument('--pbt_eval_interval', type=int, default=20,
                    help='PBT evaluation interval (training steps)')
parser.add_argument('--seed', type=int, default=0,
                    help='Random seed')


class LightTrainer:
    '''
    Class to train flow traffic lights from a partially observed MDP
    Attributes
    ----------
    - num_lights : the number of traffic lights in the network
    - model : the model we are using for the Q-network
    '''

    def __init__(self, alg, N_ROWS, N_COLUMNS, config, saving_dir=None, seed=0, pbt_config=None):
        self.alg = alg
        self.N_ROWS = N_ROWS
        self.N_COLUMNS = N_COLUMNS
        self.num_lights = N_COLUMNS * N_ROWS
        self.config = config
        self.seed = seed
        self.pbt_config = pbt_config or {}

        # 根据算法类型调整保存目录
        alg_name = self.alg.name if hasattr(self.alg, 'name') else str(self.alg)
        if 'PBT' in alg_name:
            dir_suffix = f"PBT_pop{self.pbt_config.get('population_size', 8)}_gen{self.pbt_config.get('num_generations', 50)}"
        else:
            dir_suffix = f"{self.config.alg.perturb}_{self.alg.lam}_{self.config.alg.perturb_alpha}"

        if saving_dir is None:
            self.saving_dir = f"results/{alg_name}_policy{N_ROWS}x{N_COLUMNS}_{dir_suffix}_seed{seed}"
        else:
            self.saving_dir = saving_dir

        os.makedirs(self.saving_dir, exist_ok=True)

        # 创建PBT检查点目录
        if 'PBT' in alg_name:
            self.pbt_checkpoint_dir = os.path.join(self.saving_dir, 'pbt_checkpoints')
            os.makedirs(self.pbt_checkpoint_dir, exist_ok=True)
            print(f"PBT checkpoint directory: {self.pbt_checkpoint_dir}")

    def process_local_observation(self, obs, last_change, n_lights=None):
        '''
        Function to combine environment observation, last light change,
        and observation area into torch tensors for use in local Q-network
        '''
        if n_lights is None:
            n_lights = self.num_lights

        last_change = np.ndarray.flatten(last_change)
        vals = list(obs.values())
        obs_tuple = []
        for i in range(n_lights):
            one_hot = np.zeros(self.num_lights)
            one_hot[i % self.num_lights] = 1
            local_obs = vals[i]
            obs_arr = np.concatenate([local_obs, [last_change[i]], one_hot])
            obs_tensor = torch.from_numpy(obs_arr).float()
            obs_tuple.append(obs_tensor)

        obs_tensor = torch.stack(obs_tuple)
        return obs_tensor

    def process_global_observation(self, obs, last_change):
        '''
        Function to combine environment observation, last light change,
        and observation area into torch tensors for use in local Q-network
        '''
        last_change = np.ndarray.flatten(last_change)
        vals = list(obs.values())
        obs_arr = np.concatenate(vals)
        obs_arr = np.concatenate([obs_arr, last_change])
        obs_tensor = torch.from_numpy(obs_arr).float()

        return obs_tensor

    def random_action(self):
        '''Function to choose random action'''
        decision = np.random.randint(0, 2)
        return decision

    def train_episodes(self, n_episodes, save_reward=False, use_pbt=True):
        '''
        Function to train agent over multiple episodes with optional PBT
        '''
        print(f"Training directory: {self.saving_dir}")

        # 初始化经验回放缓冲区
        replay = ReplayBuffer(self.config.main.replay_capacity)

        # 为PBT准备评估数据收集
        eval_data_buffer = []

        for episode in range(n_episodes):
            print(f"\n{'='*60}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'='*60}")

            # 初始化环境
            flow = self.config.env.train_parameters
            my_env = TrafficGridEnv(N_ROWS=self.N_ROWS, N_COLUMNS=self.N_COLUMNS, flow=flow)
            env = my_env.make_env()

            # 获取初始观察
            obs = env.reset()
            last_change = env.last_change

            # 存储之前的观察和动作
            old_local_obs = self.process_local_observation(obs, last_change)
            old_global_obs = self.process_global_observation(obs, last_change)
            old_actions = torch.zeros(self.num_lights)

            # 初始化动作
            actions = TrafficAction(act=0, N_ROWS=self.N_ROWS, N_COLUMNS=self.N_COLUMNS)
            is_training = False
            episode_rewards = []
            episode_data = []  # 用于PBT评估的数据

            start_iter = 1
            end_iter = self.config.main.train_iters + 1

            self.config.main.target_update_interval = getattr(self.config.main, 'target_update_interval', 1000)

            for i in range(start_iter, end_iter):
                # 前1000个时间步采取随机动作
                if i <= 1000:
                    for j in range(self.num_lights):
                        if i % 10 == 0:
                            action = self.random_action()
                            actions["center{}".format(j)] = action
                        else:
                            action = 0
                            actions["center{}".format(j)] = action

                if i % 100 == 0:
                    print(f"Episode: {episode}, Iter: {i}")

                if i >= 1000 or episode > 0:
                    is_training = True

                # 执行动作
                obs, _, done, info = env.step(actions)
                last_change = env.last_change
                rewards = env.compute_individual_reward(actions, obs, last_change)

                # 准备经验数据
                actions_tensor = torch.Tensor([int(actions["center{}".format(i)]) for i in range(self.num_lights)])
                global_reward = sum(rewards.values())

                new_local_obs = self.process_local_observation(obs, last_change)
                new_global_obs = self.process_global_observation(obs, last_change)

                local_rewards = torch.Tensor(list(rewards.values()))
                experience = [actions_tensor, global_reward, old_local_obs, old_global_obs,
                              new_local_obs, new_global_obs, local_rewards]
                replay.append(experience)

                # 为PBT评估收集数据
                if use_pbt and hasattr(self.alg, 'pbt_manager'):
                    eval_data_buffer.append({
                        'old_global_obs': old_global_obs.clone(),
                        'actions': actions_tensor.clone(),
                        'global_reward': global_reward
                    })
                    # 保持缓冲区大小
                    if len(eval_data_buffer) > 1000:
                        eval_data_buffer.pop(0)

                # 每5步训练一次
                if is_training and i % self.config.main.update_period == 0:
                    # 如果是PBT算法，传入评估数据缓冲区
                    if use_pbt and hasattr(self.alg, 'pbt_manager'):
                        # 创建评估函数
                        def evaluate_pbt_individual(config, replay_buffer=None, eval_data=None):
                            if eval_data is None or len(eval_data) == 0:
                                return 0.0

                            # 创建临时agent进行评估
                            temp_agent = QCOMBOS(self.N_ROWS, self.N_COLUMNS, config)
                            # 复制当前网络的权重（保持架构相同）
                            temp_agent.local_net.load_state_dict(self.alg.local_net.state_dict())
                            temp_agent.global_net.load_state_dict(self.alg.global_net.state_dict())

                            # 评估性能
                            total_reward = 0.0
                            eval_samples = min(100, len(eval_data))
                            indices = np.random.choice(len(eval_data), eval_samples, replace=False)

                            for idx in indices:
                                data = eval_data[idx]
                                with torch.no_grad():
                                    # 计算Q值
                                    global_Q = temp_agent.global_net(data['old_global_obs'].unsqueeze(0))
                                    binary_coeff = torch.Tensor([2 ** (self.num_lights - i - 1)
                                                                for i in range(self.num_lights)])
                                    global_action = torch.matmul(data['actions'].unsqueeze(0), binary_coeff)
                                    q_value = global_Q[0, global_action.long().item()]
                                    total_reward += q_value.item()

                            avg_q_value = total_reward / eval_samples
                            return avg_q_value

                        # 训练步骤（PBT会在内部处理评估）
                        self.alg.train_step(replay, summarize=False)

                        # 定期显示PBT进度
                        if i % (self.config.main.update_period * 10) == 0:
                            if hasattr(self.alg, 'get_tuning_progress'):
                                progress = self.alg.get_tuning_progress()
                                print(f"[PBT] Generation: {progress['generation']}, "
                                      f"Best Fitness: {progress['best_fitness']:.4f}")
                    else:
                        # 普通训练步骤
                        self.alg.train_step(replay, summarize=False)

                # 做出下一个决策
                old_actions_list = []
                if i >= 1000 or episode > 0:
                    for j in range(self.num_lights):
                        new_state_tensor = new_local_obs[j, :]
                        action = self.alg.choose_action(new_state_tensor)
                        actions["center{}".format(j)] = action
                        old_actions_list.append(action)

                # 更新观察
                old_local_obs = new_local_obs
                old_global_obs = new_global_obs
                old_actions = actions_tensor
                episode_rewards.append(global_reward)

                # 定期更新目标网络
                if i % self.config.main.target_update_interval == 0:
                    self.alg.update_targets()

            # 保存奖励
            episode_rewards = np.array(episode_rewards)
            if save_reward:
                np.save(f"{self.saving_dir}/training_reward_ep{episode}.npy", episode_rewards)

            print(f"Episode {episode} average reward: {episode_rewards.mean():.2f} ± {episode_rewards.std():.2f}")

            # 定期保存模型（如果使用PBT，PBT有自己的保存机制）
            if episode % 5 == 0:
                if use_pbt and hasattr(self.alg, 'pbt_manager'):
                    # PBT算法会自己保存最佳模型
                    pass
                else:
                    self.save(f"episode_{episode}")

        # 训练完成后，如果使用PBT，保存最终结果
        if use_pbt and hasattr(self.alg, 'pbt_manager'):
            # 获取最佳超参数
            best_params = self.alg.get_best_hyperparams()
            print("\n" + "="*60)
            print("PBT Optimization Complete!")
            print("Best Hyperparameters:")
            for param, value in best_params.items():
                print(f"  {param}: {value:.6f}")
            print("="*60)

            # 保存最佳超参数
            import json
            with open(f"{self.saving_dir}/best_pbt_params.json", 'w') as f:
                json.dump({k: float(v) for k, v in best_params.items()}, f, indent=2)

            # 绘制进化过程
            try:
                self.alg.plot_evolution(f"{self.saving_dir}/pbt_evolution.png")
            except:
                print("Could not generate PBT evolution plot")

    # 其他方法保持不变...
    def eval(self, render=False, eval_dict=None, perturb=False, perturb_size=0, speed=35, n_lights=4):
        '''
        Function to evaluate model on SUMO environment
        Attributes
        ----------
        - N_ROWS : the number of rows in the traffic grid
        - N_COLUMNS : the number of columns in the traffic grid
        '''
        flow = 700
        # Initialize environment
        if n_lights == 4:
            my_env = TrafficGridEnv(N_ROWS=2, N_COLUMNS=2, render=False, eval_dict=eval_dict,
                                    speed=speed)
        elif n_lights == 16:
            my_env = TrafficGridEnv(N_ROWS=4, N_COLUMNS=4, render=False, eval_dict=eval_dict,
                                    speed=speed)
        env = my_env.make_env()

        # Get initial observation from environment
        obs = env.reset()
        last_change = env.last_change

        # Store previous observation and actions
        old_local_obs = self.process_local_observation(obs, last_change, n_lights=n_lights)
        old_global_obs = self.process_global_observation(obs, last_change)
        old_actions = torch.zeros(self.num_lights)

        # Initialize action
        actions = TrafficAction(act=0, N_ROWS=self.N_ROWS, N_COLUMNS=self.N_COLUMNS)

        # Run simulation
        r = []
        for i in range(1, self.config.main.eval_iters + 1):
            if i % 100 == 0:
                print(i)
            if i <= 1000:
                for j in range(n_lights):
                    if i % 9 == 0:
                        action = self.random_action()
                        actions["center{}".format(j)] = action
                    else:
                        action = 0
                        actions["center{}".format(j)] = action

            obs, _, done, info = env.step(actions)
            last_change = env.last_change
            rewards = env.compute_individual_reward(actions, obs, last_change)

            actions_tensor = torch.Tensor([int(actions["center{}".format(i)]) for i in range(n_lights)])
            global_reward = sum(rewards.values())

            new_local_obs = self.process_local_observation(obs, last_change, n_lights=n_lights)
            new_global_obs = self.process_global_observation(obs, last_change)

            local_rewards = torch.Tensor(list(rewards.values()))

            # Make next decision
            old_actions = []
            if i >= 1000:
                for j in range(n_lights):
                    new_state_tensor = new_local_obs[j, :]

                    if perturb:
                        # Perturb new_state_tensor
                        new_state_tensor = new_state_tensor + torch.normal(torch.zeros(new_state_tensor.shape),
                                                                           torch.ones_like(
                                                                               new_state_tensor) * perturb_size)

                    action = self.alg.choose_action(new_state_tensor)
                    actions["center{}".format(j)] = action
                    old_actions.append(action)

            # Update observations
            old_local_obs = new_local_obs
            old_global_obs = new_global_obs
            old_actions = actions_tensor
            r.append(global_reward)

        r = np.array(r)

        np.save(f"{self.saving_dir}/eval_reward_{perturb}_{perturb_size}_{speed}_{n_lights}.npy", r)

    def eval_marl(self, perturb_size=1e-1, render=False, eval_dict=None, eval_num=0, seed=0, cutoff=0.05):
        '''
        Function to evaluate an agent trained on the 2x2 grid on a 4x4 grid
        Attributes
        ----------
        - N_ROWS : the number of rows in the traffic grid
        - N_COLUMNS : the number of columns in the traffic grid
        '''
        n_lights = 4
        flow = 700
        # Initialize environment
        my_env = TrafficGridEnv(N_ROWS=2, N_COLUMNS=2, render=render, eval_dict=eval_dict)
        env = my_env.make_env()

        # Get initial observation from environment
        obs = env.reset()
        last_change = env.last_change

        # Store previous observation and actions
        old_local_obs = self.process_local_observation(obs, last_change,
                                                       n_lights=n_lights)  # Change to include MF Action
        old_global_obs = self.process_global_observation(obs, last_change)
        old_actions = torch.zeros(self.num_lights)

        # Initialize action
        actions = TrafficAction(act=0, N_ROWS=self.N_ROWS, N_COLUMNS=self.N_COLUMNS)

        # Run simulation
        is_training = False
        r = []

        log_path = self.saving_dir
        header = 'timestamp,time,'
        header += ','.join(['reward_{}'.format(idx) for idx in range(self.num_lights)] + ['global_reward'])
        header += '\n'
        start_time = time.time()
        with open(os.path.join(log_path, 'eval_log_{}.csv'.format(flow)), 'w') as f:
            f.write(header)

        for i in range(1, self.config.main.eval_iters + 1):
            if i % 100 == 0:
                print(i)
            if i <= 1000:
                for j in range(n_lights):
                    if i % 9 == 0:
                        action = self.random_action()
                        actions["center{}".format(j)] = action
                    else:
                        action = 0
                        actions["center{}".format(j)] = action

            obs, _, done, info = env.step(actions)
            last_change = env.last_change
            rewards = env.compute_individual_reward(actions, obs, last_change)

            actions_tensor = torch.Tensor([int(actions["center{}".format(i)]) for i in range(n_lights)])
            global_reward = sum(rewards.values())

            new_local_obs = self.process_local_observation(obs, last_change, n_lights=n_lights)
            new_global_obs = self.process_global_observation(obs, last_change)

            # Find mean field actions if the algorithm is a mean field algorithm
            if self.alg.mean_field:
                new_local_obs, new_actions = self.alg.get_mf_action(old_actions=old_actions,
                                                                    new_local_obs=new_local_obs)

            local_rewards = torch.Tensor(list(rewards.values()))

            # Train every fifth step
            if i % self.config.main.log_period == 0:
                s = '{},{}'.format(i, time.time() - start_time)
                for idx in range(self.num_lights):
                    s += ',{}'.format(rewards['center{}'.format(idx)])
                s += ',{}'.format(global_reward)
                s += '\n'
                with open(os.path.join(log_path, 'eval_log_{}.csv'.format(flow)), 'a') as f:
                    f.write(s)
            # print(i, global_reward)

            # Make next decision
            old_actions = []
            if i >= 1000:
                for j in range(n_lights):
                    if self.alg.mean_field:
                        action = new_actions[j]
                    elif self.alg.share_obs:
                        action = self.alg.local_agents[j].choose_action(new_global_obs)
                    else:
                        new_state_tensor = new_local_obs[j, :]
                        # Perturb new_state_tensor
                        new_state_tensor = new_state_tensor + torch.normal(torch.zeros(new_state_tensor.shape),
                                                                           torch.ones_like(
                                                                               new_state_tensor) * perturb_size)
                        action = self.alg.choose_action(new_state_tensor)
                        if np.random.uniform(0, 1) < cutoff:
                            action = 1 - action
                    actions["center{}".format(j)] = action
                    old_actions.append(action)
                to_print = [int(actions["center{}".format(k)]) for k in range(self.num_lights)]
            # Update observations
            old_local_obs = new_local_obs
            old_global_obs = new_global_obs
            old_actions = actions_tensor
            r.append(global_reward)
        # self.tfboard_writer.add_scalar('GlobalReward/eval', global_reward, i)

        r = np.array(r)

        print(f"{self.saving_dir}/eval_reward_perturbed_agent" + "_" + str(cutoff) + str(seed) + "_.npy")
        np.save(f"{self.saving_dir}/eval_reward_perturbed_agent" + "_" + str(cutoff) + str(seed) + "_.npy", r)

    def save(self, subdir=""):
        """
        Save the agent
        """
        save_dir = f"{self.saving_dir}/{subdir}" if subdir else self.saving_dir
        os.makedirs(save_dir, exist_ok=True)
        self.alg.save(save_dir)

    def load(self, dir=None):
        """
        Load the agent
        """
        if dir is None:
            dir = self.saving_dir
        self.alg.load(dir)


if __name__ == "__main__":
    args = parser.parse_args()

    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # 选择算法
    if args.alg == 'qcombo':
        from Algorithms.configs import config_qcombo
        config = config_qcombo.get_config()
        alg = QCOMBO(n_rows=config.env.n_rows, n_cols=config.env.n_cols, config=config)

    elif args.alg == 'coma':
        from Algorithms.configs import config_coma
        config = config_coma.get_config()
        alg = COMA(n_rows=config.env.n_rows, n_cols=config.env.n_cols, config=config)

    elif args.alg == 'qcombo_adv':
        from Algorithms.configs import config_qcombo_adv
        config = config_qcombo_adv.get_config()
        alg = QCOMBOS(n_rows=config.env.n_rows, n_cols=config.env.n_cols, config=config)

    elif args.alg == 'PBT_adv':
        from Algorithms.configs import config_qcombo_adv
        config = config_qcombo_adv.get_config()

        # 配置PBT参数
        pbt_config = {
            'population_size': args.pbt_pop_size,
            'eval_interval': args.pbt_eval_interval,
            'checkpoint_interval': 10,  # 每10代保存一次检查点
            'checkpoint_dir': None,  # 将在LightTrainer中设置
            'optimize_params': [
                ('perturb_epsilon', 1e-5, 1e-2, 'log'),
                ('perturb_num_steps', 1, 30, 'linear'),
                ('perturb_alpha', 1e-4, 1e-1, 'log'),
                ('lam', 0.01, 10.0, 'log'),
                ('qcombo_lam', 0.01, 10.0, 'log'),
                ('exploration_rate', 0.01, 0.5, 'linear'),
                ('discount', 0.9, 0.999, 'linear'),
            ]
        }

        # 创建PBT优化的QCOMBO
        alg = QCOMBO_PBT(
            n_rows=config.env.n_rows,
            n_cols=config.env.n_cols,
            config=config,
            pbt_config=pbt_config
        )

    # 设置网格大小
    config.env.n_rows = args.nrow
    config.env.n_cols = args.ncol

    # 创建保存目录
    save_dir = f"results/{args.alg}_grid{args.nrow}x{args.ncol}_seed{args.seed}"
    if args.alg == 'PBT_adv':
        save_dir += f"_pop{args.pbt_pop_size}"

    # 创建训练器
    trainer = LightTrainer(
        alg=alg,
        N_ROWS=config.env.n_rows,
        N_COLUMNS=config.env.n_cols,
        config=config,
        saving_dir=save_dir,
        seed=args.seed,
        pbt_config=pbt_config if args.alg == 'PBT_adv' else None
    )

    # 设置PBT检查点目录
    if args.alg == 'PBT_adv':
        trainer.alg.pbt_config['checkpoint_dir'] = trainer.pbt_checkpoint_dir

    # 训练
    print("\n" + "="*60)
    print(f"Starting training with {args.alg}")
    print(f"Grid: {args.nrow}x{args.ncol}")
    print(f"Seed: {args.seed}")
    if args.alg == 'PBT_adv':
        print(f"PBT Population: {args.pbt_pop_size}")
        print(f"PBT Generations: {args.pbt_generations}")
    print("="*60 + "\n")

    # 训练多个episode
    num_episodes = 10  # 可以根据需要调整

    # 对于PBT，我们可以根据代数调整训练episode数
    if args.alg == 'PBT_adv':
        # 每个个体训练更少的episode，但更多代
        num_episodes = 3  # 每代训练3个episode

    s = 2
    torch.manual_seed(s)
    np.random.seed(s)
    random.seed(s)

    trainer.train_episodes(
        n_episodes=num_episodes,
        save_reward=True,
        use_pbt=(args.alg == 'PBT_adv')
    )

    # 保存最终模型
    trainer.save("final")

    # 评估模型
    print("\n" + "="*60)
    print("Evaluating trained model...")
    print("="*60)

    # 使用训练环境进行评估
    trainer.eval(eval_dict=train_2x2, speed=35, perturb=True, perturb_size=1e-1)

    print("\nTraining completed successfully!")
    print(f"Results saved to: {trainer.saving_dir}")