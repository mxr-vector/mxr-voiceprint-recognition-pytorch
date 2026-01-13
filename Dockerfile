# 使用官方 python 镜像作为基础镜像
FROM python:3.11-slim AS base
LABEL maintainer="YuanJie" \
    description="An Acoustic Feature Detection Mirror Construction Project" \
    license="MIT" \
    email="wangjh0825@qq.com"

# 写入阿里云 Debian 12 源（deb822 格式）
RUN cat > /etc/apt/sources.list.d/debian.sources <<'EOF'
Types: deb
URIs: http://mirrors.aliyun.com/debian
Suites: bookworm bookworm-updates
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg

Types: deb
URIs: http://mirrors.aliyun.com/debian-security
Suites: bookworm-security
Components: main contrib non-free non-free-firmware
Signed-By: /usr/share/keyrings/debian-archive-keyring.gpg
EOF

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    procps \
    espeak-ng \
    ca-certificates && \
    ln -s /usr/bin/espeak-ng /usr/bin/espeak && \
    rm -rf /var/lib/apt/lists/*

# 用 PyPI 国内源安装 uv（稳定）
ENV UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
# 增加 uv 下载超时（单位秒，建议 300+）
ENV UV_HTTP_TIMEOUT=600
# 用 pip3 安装 uv
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -U uv -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    uv --version
# ENV PATH="/root/.local/bin:${PATH}"


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