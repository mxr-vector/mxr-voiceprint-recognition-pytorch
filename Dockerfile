# 使用官方 python 镜像作为基础镜像
FROM python:3.11-slim AS base
LABEL maintainer="782353676@qq.com" \
    description="An Acoustic Feature Detection Mirror Construction Project" \
    license="MIT" \
    nickname="YuanJie"

# 写入阿里云 Debian 12 源（deb822 格式）
RUN printf "Types: deb\n\
    URIs: http://mirrors.aliyun.com/debian\n\
    Suites: bookworm bookworm-updates bookworm-security\n\
    Components: main contrib non-free non-free-firmware\n\
    Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg\n" \
    > /etc/apt/sources.list.d/debian.sources

# 安装 uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
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