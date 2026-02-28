from .configdict import ConfigDict


def get_config():
    config = ConfigDict()

    config.alg = ConfigDict()
    config.alg.discount = 0.95
    config.alg.gae_lambda = 0.95
    config.alg.clip_epsilon = 0.2
    config.alg.value_coef = 0.5
    config.alg.entropy_coef = 0.01
    config.alg.lr = 3e-4
    config.alg.buffer_size = 2048
    config.alg.batch_size = 64
    config.alg.num_epochs = 10
    config.alg.exploration_rate = 0.1

    config.critic = ConfigDict()
    config.critic.lr = 3e-4
    config.critic.size = [64, 64]

    config.actor = ConfigDict()
    config.actor.lr = 3e-4
    config.actor.size = [64, 64]

    config.main = ConfigDict()
    config.main.dir = 'results/MAPPO'
    config.main.train_iters = 3000
    config.main.eval_iters = 5000
    config.main.update_period = 10  # Run an update every _ steps
    config.main.replay_capacity = 30000

    config.env = ConfigDict()
    config.env.n_rows = 2
    config.env.n_cols = 2
    config.env.train_parameters = [700] * 8

    return config