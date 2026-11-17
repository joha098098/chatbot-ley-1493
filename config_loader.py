import os
import yaml

DEFAULT_CONFIG = {
    "app": {
        "title": "Asistente Normativo Ley 1493 de 2011 | Artes Escénicas",
        "page_icon": "🎭",
        "layout": "wide",
    },
    "llm": {
        "mode": "local",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "temperature": 0.1,
        "max_new_tokens": 512,
        "repetition_penalty": 1.1,
        "use_4bit_quantization": True,
    },
    "embeddings": {
        "model_id": "BAAI/bge-m3",
        "device": "cpu",
        "normalize_embeddings": True,
    },
    "knowledge_base": {
        "docs_folder": "./docs_ley1493",
        "vector_store_path": "./faiss_index_ley1493",
        "chunk_size": 900,
        "chunk_overlap": 150,
    },
    "retrieval": {
        "top_k": 4,
        "score_threshold": 1.15,
    },
    "memory": {
        "max_turns": 4,
        "max_chars": 3000,
    },
    "security_and_logging": {
        "max_input_length": 1000,
        "log_file_path": "./logs/conversations_ley1493.jsonl",
    },
}

def load_config(config_path: str = "config.yaml") -> dict:
    config = {section: values.copy() for section, values in DEFAULT_CONFIG.items()}

    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        for section, values in user_cfg.items():
            if section in config and isinstance(values, dict):
                config[section].update(values)

    if os.getenv("LLM_MODE"):
        config["llm"]["mode"] = os.getenv("LLM_MODE").lower()
    if os.getenv("LLM_MODEL_ID"):
        config["llm"]["model_id"] = os.getenv("LLM_MODEL_ID")

    return config
