# ========== 前端构建阶段 ==========
FROM node:20-slim AS frontend-builder

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ========== Python 依赖构建阶段 ==========
FROM python:3.11-slim AS python-builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ========== 运行阶段 ==========
FROM python:3.11-slim

WORKDIR /app

# 复制 Python 依赖
COPY --from=python-builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制应用代码
COPY . .

# 复制前端构建产物
COPY --from=frontend-builder /build/frontend/dist /app/frontend/dist

EXPOSE 8000

CMD ["python", "api_server.py"]
