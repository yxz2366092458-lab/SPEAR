from .configdict import ConfigDict


def get_config():
    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.discount = 0.95
    config.alg.tau = 0.01  # soft update parameter
    config.alg.lr_actor = 1e-4
    config.alg.lr_critic = 1e-3
    config.alg.buffer_size = 10000
    config.alg.batch_size = 32
    config.alg.noise_scale = 0.1
    config.alg.noise_decay = 0.999
    config.alg.exploration_rate = 0.1

    config.critic = ConfigDict()
    config.critic.lr = 1e-3
    config.critic.size = [128, 128]

    config.actor = ConfigDict()
    config.actor.lr = 1e-4
    config.actor.size = [64, 64]

    config.main = ConfigDict()
    config.main.dir = 'results/MADDPG'
    config.main.train_iters = 3000
    config.main.eval_iters = 5000
    config.main.update_period = 10  # Run an update every _ steps
    config.main.replay_capacity = 30000

    config.env = ConfigDict()
    config.env.n_rows = 2
    config.env.n_cols = 2
    config.env.train_parameters = [700] * 8

    return config