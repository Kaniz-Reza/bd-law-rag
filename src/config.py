from pathlib import Path

import yaml
from pydantic import BaseModel


class DataConfig(BaseModel):
    source: str
    raw_dir: str
    processed_dir: str
    exclude_repealed: bool
    pilot_act_ids: list[int]


class ChunkingConfig(BaseModel):
    max_tokens: int
    overlap_tokens: int


class RetrievalConfig(BaseModel):
    top_k: int
    min_score: float


class EmbeddingConfig(BaseModel):
    name: str
    candidates: list[str]


class LLMConfig(BaseModel):
    provider: str
    model: str
    max_output_tokens: int


class Config(BaseModel):
    data: DataConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig
    embedding: EmbeddingConfig
    llm: LLMConfig


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    with open(path, encoding="utf-8") as f:
        return Config(**yaml.safe_load(f))
