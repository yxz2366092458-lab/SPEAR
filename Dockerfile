# 使用Ubuntu 18.04作为基础镜像，与文档中SUMO安装脚本的目标系统一致
FROM ubuntu:18.04

# 避免在安装过程中出现交互式提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖：构建工具、库、Git、Python等
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    ca-certificates \
    # SUMO 构建依赖 \
    autoconf \
    automake \
    libtool \
    pkg-config \
    libgdal-dev \
    libproj-dev \
    libxerces-c-dev \
    libfox-1.6-dev \
    # Python 开发头文件 用于编译某些Python包 \
    python3-dev \
    # 清理缓存以减小镜像大小 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安装 Miniconda（用于创建Flow的Python环境）
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR && \
    rm ~/miniconda.sh && \
    echo "export PATH=$CONDA_DIR/bin:\$PATH" >> /etc/profile.d/conda.sh
ENV PATH $CONDA_DIR/bin:$PATH

# 克隆Flow仓库（包含environment.yml和安装脚本）
WORKDIR /workspace
RUN git clone https://github.com/flow-project/flow.git && \
    cd flow && \
    # 可选切换到特定稳定版本 如有需要 可指定commit或tag 文档未强制 \
    # git checkout <desired-commit> \
    # 使用environment.yml创建conda环境\
    conda env create -f environment.yml

# 激活conda环境的快捷方式（通过设置PATH）
ENV PATH $CONDA_DIR/envs/flow/bin:$PATH

# 安装Flow（以可编辑模式）
WORKDIR /workspace/flow
RUN pip install -e .

# 安装SUMO（从GitHub源码编译特定兼容版本）
WORKDIR /workspace
# 克隆SUMO仓库并切换到文档指定的commit（2147d1551b）
RUN git clone https://github.com/eclipse/sumo.git && \
    cd sumo && \
    git checkout 2147d1551b && \
    make -f Makefile.cvs && \
    # 配置编译选项 针对Ubuntu \
    ./configure CXXFLAGS="-std=c++11" --prefix=/usr/local && \
    make -j$(nproc) && \
    make install

# 设置SUMO环境变量
ENV SUMO_HOME /workspace/sumo
ENV PATH $SUMO_HOME/bin:$PATH
ENV PYTHONPATH $SUMO_HOME/tools:$PYTHONPATH

# （可选）安装RLlib——Flow的conda环境已包含Ray，无需额外操作
# 如需安装h-baselines等扩展库，可在此添加
# 例如：
# RUN git clone https://github.com/AboudyKreidieh/h-baselines.git && \
#     cd h-baselines && \
#     pip install -e .

# 清理不必要的文件以减小镜像体积
RUN conda clean --all -y && \
    rm -rf /workspace/sumo/.git /workspace/flow/.git

# 设置工作目录
WORKDIR /workspace/flow

# 设置默认命令：启动bash（用户可交互运行Flow示例）
CMD ["/bin/bash"]