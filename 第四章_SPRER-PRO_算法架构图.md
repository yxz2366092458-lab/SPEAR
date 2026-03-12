# 第四章 SPRER-PRO 算法架构图

## 图4.1 SPRER-PRO整体架构图

```mermaid
graph TD
    A[交通环境] --> B[状态观测]
    B --> C[基础SPEAR模块]
    
    C --> D[追随者网络LocalNet]
    C --> E[领导者网络GlobalNet]
    C --> F[对抗训练模块]
    C --> G[种群式训练PBT]
    
    B --> H[启发式规则模块]
    H --> I[安全规则]
    H --> J[效率规则]
    H --> K[适应规则]
    
    B --> L[自注意力意图识别模块]
    L --> M[短期意图识别]
    L --> N[长期意图识别]
    L --> O[群体意图识别]
    L --> P[不确定性量化]
    
    B --> Q[风险感知优化模块]
    Q --> R[风险评估]
    Q --> S[风险度量]
    Q --> T[风险优化]
    
    D --> U[决策融合模块]
    E --> U
    F --> U
    G --> U
    I --> U
    J --> U
    K --> U
    M --> U
    N --> U
    O --> U
    P --> U
    R --> U
    S --> U
    T --> U
    
    U --> V[最终决策]
    V --> A
    
    style A fill:#e1f5fe
    style C fill:#e8f5e8
    style H fill:#fff3e0
    style L fill:#f3e5f5
    style Q fill:#ffebee
    style U fill:#fce4ec
```

## 图4.2 SPRER-PRO详细模块图

```mermaid
graph TD
    A[输入状态] --> B[特征提取]
    
    B --> C[SPEAR基础模块]
    C --> D[斯塔克尔伯格博弈]
    C --> E[反事实基线]
    C --> F[对抗训练]
    C --> G[PBT优化]
    
    B --> H[启发式规则模块]
    H --> I[规则引擎]
    H --> J[规则置信度计算]
    H --> K[规则权重调整]
    
    B --> L[意图识别模块]
    L --> M[时空特征编码]
    L --> N[自注意力计算]
    L --> O[意图解码]
    L --> P[不确定性估计]
    
    B --> Q[风险感知模块]
    Q --> R[风险价值分布]
    Q --> S[CVaR计算]
    Q --> T[风险自适应调整]
    
    D --> U[决策融合]
    I --> U
    O --> U
    T --> U
    
    U --> V[注意力权重计算]
    V --> W[加权融合]
    W --> X[最终决策输出]
    
    style A fill:#e1f5fe
    style C fill:#e8f5e8
    style H fill:#fff3e0
    style L fill:#f3e5f5
    style Q fill:#ffebee
    style U fill:#fce4ec
    style X fill:#f1f8e9
```

## 图4.3 启发式规则模块详细图

```mermaid
graph TD
    A[交通状态观测] --> B[规则特征提取]
    
    B --> C[安全规则]
    C --> C1[最小绿灯时间]
    C --> C2[最大红灯时间]
    C --> C3[冲突避免]
    
    B --> D[效率规则]
    D --> D1[最大压力规则]
    D --> D2[最小延误规则]
    D --> D3[流量平衡规则]
    
    B --> E[适应规则]
    E --> E1[时段适应]
    E --> E2[天气适应]
    E --> E3[事件适应]
    
    C1 --> F[规则置信度网络]
    C2 --> F
    C3 --> F
    D1 --> F
    D2 --> F
    D3 --> F
    E1 --> F
    E2 --> F
    E3 --> F
    
    F --> G[规则权重计算]
    G --> H[规则输出融合]
    
    H --> I[规则决策输出]
    
    style A fill:#e1f5fe
    style C fill:#e8f5e8
    style D fill:#fff3e0
    style E fill:#f3e5f5
    style F fill:#fce4ec
    style I fill:#f1f8e9
```

## 图4.4 自注意力意图识别模块图

```mermaid
graph TD
    A[交通时空序列] --> B[输入编码层]
    
    B --> C[位置编码]
    C --> D[自注意力层1]
    D --> E[自注意力层2]
    E --> F[自注意力层3]
    
    F --> G[短期意图解码]
    F --> H[长期意图解码]
    F --> I[群体意图解码]
    
    G --> J[意图融合]
    H --> J
    I --> J
    
    J --> K[不确定性量化]
    K --> L[意图输出]
    
    style A fill:#e1f5fe
    style B fill:#e8f5e8
    style D fill:#fff3e0
    style G fill:#f3e5f5
    style H fill:#f3e5f5
    style I fill:#f3e5f5
    style J fill:#fce4ec
    style L fill:#f1f8e9
```

## 图4.5 风险感知优化模块图

```mermaid
graph TD
    A[状态特征] --> B[风险价值网络]
    
    B --> C[分位数估计]
    C --> D[风险分布]
    
    D --> E[CVaR计算]
    E --> F[风险度量]
    
    F --> G[风险自适应网络]
    G --> H[风险权重调整]
    
    H --> I[风险调整价值]
    I --> J[风险感知输出]
    
    style A fill:#e1f5fe
    style B fill:#ffebee
    style C fill:#fff3e0
    style E fill:#f3e5f5
    style G fill:#e8f5e8
    style J fill:#f1f8e9
```

## 图4.6 决策融合机制图

```mermaid
graph TD
    A[SPEAR输出] --> B[决策编码]
    C[规则输出] --> D[决策编码]
    E[意图输出] --> F[决策编码]
    G[风险输出] --> H[决策编码]
    
    B --> I[注意力融合]
    D --> I
    F --> I
    H --> I
    
    I --> J[注意力权重]
    J --> K[加权融合]
    K --> L[融合决策]
    
    L --> M[最终动作输出]
    
    style A fill:#e8f5e8
    style C fill:#fff3e0
    style E fill:#f3e5f5
    style G fill:#ffebee
    style I fill:#fce4ec
    style M fill:#f1f8e9
```

## 图4.7 SPRER-PRO训练流程图

```mermaid
graph TD
    A[初始化] --> B[各模块初始化]
    B --> C[数据预处理]
    
    C --> D[环境交互]
    D --> E[多模块并行处理]
    
    E --> F[SPEAR模块学习]
    E --> G[规则模块学习]
    E --> H[意图模块学习]
    E --> I[风险模块学习]
    
    F --> J[决策融合训练]
    G --> J
    H --> J
    I --> J
    
    J --> K[性能评估]
    K --> L{收敛判断}
    
    L -->|否| D
    L -->|是| M[模型保存]
    
    M --> N[部署应用]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style D fill:#e8f5e8
    style E fill:#fff3e0
    style J fill:#fce4ec
    style M fill:#f1f8e9
    style N fill:#e0f2f1