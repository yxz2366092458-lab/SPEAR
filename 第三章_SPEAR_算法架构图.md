# 第三章 SPEAR 算法架构图

## 图3.1 SPEAR整体架构图

```mermaid
graph TD
    A[交通环境] --> B[状态观测]
    B --> C[局部观测处理]
    B --> D[全局观测处理]
    
    C --> E[追随者网络LocalNet]
    D --> F[领导者网络GlobalNet]
    
    F --> G[斯塔克尔伯格博弈]
    E --> G
    
    G --> H[反事实基线计算]
    H --> I[对抗训练模块]
    
    I --> J[种群式训练PBT]
    J --> K[超参数优化]
    
    K --> L[经验回放缓冲区]
    L --> M[训练更新]
    
    M --> E
    M --> F
    
    G --> N[动作选择]
    N --> A
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style E fill:#e8f5e8
    style F fill:#fff3e0
    style G fill:#fce4ec
    style I fill:#ffebee
    style J fill:#e0f2f1
    style N fill:#f1f8e9
```

## 图3.2 SPEAR详细流程图

```mermaid
graph TD
    A[开始] --> B[初始化网络参数]
    B --> C[初始化PBT种群]
    
    C --> D[环境交互]
    D --> E{选择动作}
    E --> F[执行动作]
    F --> G[观测奖励和新状态]
    G --> H[存储经验]
    
    H --> I{是否训练?}
    I -->|是| J[从缓冲区采样]
    J --> K[计算局部损失]
    J --> L[计算全局损失]
    J --> M[计算正则化损失]
    
    K --> N[反事实基线计算]
    L --> N
    M --> N
    
    N --> O[对抗训练]
    O --> P[生成对抗扰动]
    P --> Q[计算对抗损失]
    
    Q --> R[总损失计算]
    R --> S[网络参数更新]
    
    S --> T{PBT进化?}
    T -->|是| U[评估种群性能]
    U --> V[选择优秀个体]
    V --> W[复制超参数]
    W --> X[突变产生新个体]
    
    X --> Y{结束条件?}
    S --> Y
    T -->|否| Y
    
    Y -->|否| D
    Y -->|是| Z[结束]
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style D fill:#e8f5e8
    style O fill:#ffebee
    style T fill:#e0f2f1
    style Z fill:#f1f8e9
```

## 图3.3 斯塔克尔伯格博弈机制图

```mermaid
graph LR
    A[全局状态] --> B[领导者网络GlobalNet]
    B --> C[全局Q值]
    
    D[局部状态] --> E[追随者网络LocalNet]
    C --> E
    E --> F[局部Q值]
    
    F --> G[动作选择]
    G --> H[环境]
    H --> I[全局奖励]
    H --> J[局部奖励]
    
    I --> B
    J --> E
    
    style A fill:#fff3e0
    style B fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style H fill:#e1f5fe
```

## 图3.4 对抗训练机制图

```mermaid
graph TD
    A[原始状态] --> B[对抗智能体]
    B --> C[生成对抗扰动]
    
    A --> D[领导者网络]
    C --> E[扰动状态]
    E --> F[领导者网络]
    
    D --> G[原始Q值]
    F --> H[扰动Q值]
    
    G --> I[对抗损失计算]
    H --> I
    
    I --> J[网络参数更新]
    J --> D
    
    style A fill:#e1f5fe
    style B fill:#ffebee
    style C fill:#ffebee
    style D fill:#e8f5e8
    style F fill:#e8f5e8
    style I fill:#fce4ec
```

## 图3.5 种群式训练PBT流程图

```mermaid
graph TD
    A[初始化种群] --> B[评估个体性能]
    B --> C[排序选择]
    C --> D[优秀个体复制]
    D --> E[超参数突变]
    E --> F[热启动新个体]
    
    F --> G{是否收敛?}
    G -->|否| B
    G -->|是| H[输出最优个体]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#fce4ec
    style H fill:#f1f8e9