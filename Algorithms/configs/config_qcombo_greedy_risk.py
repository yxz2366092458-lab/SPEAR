from .configdict import ConfigDict

def get_config():
    config = ConfigDict()
    
    config.alg = ConfigDict()
    config.alg.discount = .95
    config.alg.exploration_rate = .05
    config.alg.anneal_exp = .99
    config.alg.lam = 3.18
    config.alg.qcombo_lam = 1.7
    config.alg.perturb_alpha = 0.2
    config.alg.perturb_num_steps = 1
    config.alg.perturb_radius = 1
    config.alg.num_minibatches = 100
    config.alg.minibatch_size = 30
    config.alg.perturb = True
    config.alg.stackelberg = True
    
    # Greedy-Risk specific parameters
    config.alg.greedy_epsilon = 0.3  # Probability of using greedy rule
    config.alg.risk_weight = 0.5     # Weight for risk penalty in Q-adjustment
    config.alg.adaptive_greedy = True  # Adaptive decay of greedy epsilon
    
    config.critic = ConfigDict()
    config.critic.lr = 1e-4
    config.critic.size = [256, 256]
    
    config.main = ConfigDict()
    config.main.dir = 'results/QCOMBO_GREEDY_RISK'
    config.main.train_iters = 3000
    config.main.eval_iters = 5000
    config.main.update_period = 10
    config.main.replay_capacity = 30000
    
    config.env = ConfigDict()
    config.env.n_rows = 2
    config.env.n_cols = 2
    config.env.train_parameters = [700] * 8
    
    return config
