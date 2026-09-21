from src.config import load_config


def test_config_loads():
    cfg = load_config()
    assert cfg.chunking.overlap_tokens < cfg.chunking.max_tokens
    assert cfg.retrieval.top_k > 0
    assert cfg.embedding.name in cfg.embedding.candidates
