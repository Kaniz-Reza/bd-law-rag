FROM python:3.11-slim

# Install uv by copying its binary from astral's official image -- avoids a
# separate pip install step just to get the tool that installs everything else.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# Copy ONLY the dependency files first. Docker caches each instruction as a
# layer; as long as these two files don't change, this (slow) install step
# is reused from cache on every rebuild, even after editing source code.
COPY pyproject.toml uv.lock ./

# Install torch FIRST, on its own, from ONLY the PyTorch CPU-only index --
# no other index mixed in, so there's no ambiguity about which build wins.
# (Mixing --index-url with --extra-index-url, as tried before, still let
# the GPU build slip through from PyPI.) This container has no GPU, so the
# CPU build is all it needs -- several GB smaller, no nvidia-* packages.
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu

# Now install everything else normally from PyPI. torch is already
# installed and satisfies sentence-transformers' requirement, so this
# won't pull in a second (GPU) copy of it.
RUN uv pip install --system -r pyproject.toml

# Now copy the actual application code and the data the API reads at
# startup (FAISS index, chunks, chunk_ids -- see src/serving/main.py).
COPY src/ src/
COPY data/processed/ data/processed/

EXPOSE 8000

# 0.0.0.0, not 127.0.0.1: the server must accept connections from outside
# the container, not just from within it. No --reload here -- that's a
# dev-only convenience; the code is baked into the image, not meant to
# change at runtime.
CMD ["uvicorn", "src.serving.main:app", "--host", "0.0.0.0", "--port", "8000"]
