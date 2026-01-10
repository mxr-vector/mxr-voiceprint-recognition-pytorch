# 使用官方 python 镜像作为基础镜像
FROM python:3.11-slim AS base
LABEL maintainer="782353676@qq.com" \
    description="An Acoustic Feature Detection Mirror Construction Project" \
    license="MIT" \
    nickname="YuanJie"

# 安装 uv, curl 和 procps（提供 ps/pgrep） 
RUN apt-get update && \
    apt-get install -y curl procps espeak-ng && \
    # 给 espeak-ng 创建别名 espeak  音素包phonemizer要用
    ln -s /usr/bin/espeak-ng /usr/bin/espeak && \
    # 安装 uv
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    # 清理缓存
    rm -rf /var/lib/apt/lists/*


# 将 uv 放到 PATH
ENV PATH="/root/.local/bin:${PATH}"

# 设置工作目录
WORKDIR /workspace

# 拷贝依赖文件（先拷贝依赖有利于 Docker 层缓存）
COPY pyproject.toml .python-version ./

# 使用 uv 安装虚拟环境和依赖
RUN uv sync --extra cu128

# 给 run.sh 可执行权限
RUN chmod +x run.sh && mkdir -p logs

# 再拷贝项目代码
COPY . .

# uv 会创建 .venv，这里将其添加到 PATH
ENV PATH="/workspace/.venv/bin:${PATH}"

# 容器启动命令（替换成你的入口）
CMD ["bash", "run.sh", "start"]