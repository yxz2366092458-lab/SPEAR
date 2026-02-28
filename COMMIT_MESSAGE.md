# Git提交信息

## 提交标题
feat: 添加MADDPG和MAPPO算法实现

## 提交描述
### 新增功能
- 实现完整的MADDPG算法（maddpg.py）
- 实现完整的MAPPO算法（mappo.py）
- 添加算法配置文件系统
- 修改训练框架支持新算法

### 算法特性
#### MADDPG
- 分散执行集中训练架构
- 经验回放缓冲区（10000容量）
- 目标网络和软更新（tau=0.01）
- Ornstein-Uhlenbeck探索噪声
- 每个智能体独立的Actor-Critic网络

#### MAPPO
- 集中式Critic，分散式Actor
- GAE优势估计（lambda=0.95）
- PPO裁剪目标（epsilon=0.2）
- 熵正则化（coef=0.01）
- 多轮经验重用（num_epochs=10）

### 框架修改
- 扩展train_policy.py支持新算法
- 统一的算法类型检查系统
- 适配不同算法的训练循环
- 支持不同算法的动作选择接口
- 统一的经验存储和训练接口

### 配置文件
- config_maddpg.py: MADDPG算法配置
- config_mappo.py: MAPPO算法配置
- configdict.py: 配置字典工具类

### 项目结构
```
Algorithms/
├── maddpg.py              # MADDPG算法实现
├── mappo.py               # MAPPO算法实现
├── qcombo.py              # 原始QCOMBO算法
├── coma.py                # 原始COMA算法
└── configs/
    ├── config_maddpg.py   # MADDPG配置
    ├── config_mappo.py    # MAPPO配置
    └── configdict.py      # 配置工具类
```

### 使用方式
```bash
# 训练MADDPG
python3 train_policy.py --alg maddpg --nrow 2 --ncol 2 --seed 0

# 训练MAPPO
python3 train_policy.py --alg mappo --nrow 2 --ncol 2 --seed 0
```

### 研究价值
- 系统比较5种MARL算法：QCOMBO, COMA, MADDPG, MAPPO, ERNIE变体
- 建立多智能体交通信号控制基准
- 提供完整的MARL算法实现库

### 测试状态
- 代码结构测试通过
- 算法接口测试通过
- 配置文件测试通过
- 训练框架集成测试通过

### 依赖要求
- PyTorch >= 1.9.0
- NumPy >= 1.19.0
- SUMO环境（用于交通模拟）

### 注意事项
- 需要进一步调整超参数优化
- 建议使用GPU加速训练
- 实验结果保存到maddpg_results/和mappo_results/目录

## 影响范围
- 新增：MADDPG和MAPPO算法实现
- 修改：train_policy.py训练框架
- 新增：算法配置文件系统
- 不影响：现有QCOMBO和COMA算法功能

## 相关Issue
- 实现5种算法对比实验框架
- 扩展SPEAR项目算法库
- 支持多智能体强化学习研究