# ========== 构建阶段 ==========
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ========== 运行阶段 ==========
FROM python:3.11-slim

WORKDIR /app

# 从构建阶段复制已安装的依赖
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

COPY . .

EXPOSE 8000

CMD ["python", "api_server.py"]
