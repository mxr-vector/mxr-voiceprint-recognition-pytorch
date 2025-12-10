# 使用官方 python 镜像作为基础镜像
FROM python:3.11-slim AS base
LABEL maintainer="782353676@qq.com"
# 安装 uv（最新方式：一条 curl 命令）
RUN apt-get update && apt-get install -y curl && \
    curl -LsSf https://astral.sh/uv/install.sh | sh

# 将 uv 放到 PATH
ENV PATH="/root/.local/bin:${PATH}"

# 设置工作目录
WORKDIR /workspace

# 拷贝依赖文件（先拷贝依赖有利于 Docker 层缓存）
COPY pyproject.toml .python-version ./

# 使用 uv 安装虚拟环境和依赖
RUN uv sync --python 3.11

# 手动安装 GPU PyTorch
RUN uv pip install \
    torch==2.9.0 \
    torchvision==0.24.0 \
    torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu128

# 再拷贝项目代码
COPY . .

# uv 会创建 .venv，这里将其添加到 PATH
ENV PATH="/workspace/.venv/bin:${PATH}"

# 容器启动命令（替换成你的入口）
CMD ["python", "main.py"]