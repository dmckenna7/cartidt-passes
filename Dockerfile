FROM nvcr.io/nvidia/pytorch:24.02-py3

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libsm6 libxext6 git-lfs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/cartidt

COPY pyproject.toml requirements.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements.txt

COPY . .
RUN python -m pip install --no-deps -e .

ENV CARTIDT_HOME=/workspace/cartidt
ENV PYTHONPATH=/workspace/cartidt

ENTRYPOINT ["python", "-m", "cartidt.driver.train"]
CMD ["--help"]
