import os
import gc
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

def get_open_llm(llm_config: dict = None):
    llm_config = llm_config or {}
    model_id = os.getenv("LLM_MODEL_ID", llm_config.get("model_id", "meta-llama/Llama-3.1-8B-Instruct"))
    mode = os.getenv("LLM_MODE", llm_config.get("mode", "api")).lower()
    temperature = float(llm_config.get("temperature", 0.1))
    max_new_tokens = int(llm_config.get("max_new_tokens", 512))
    repetition_penalty = float(llm_config.get("repetition_penalty", 1.1))
    hf_token = os.getenv("HF_TOKEN")

    if mode == "api":
        if not hf_token:
            raise ValueError("Falta la variable de entorno HF_TOKEN. Configúrala en Secrets de Streamlit.")
        endpoint = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            huggingfacehub_api_token=hf_token
        )
        return ChatHuggingFace(llm=endpoint)

    elif mode == "local":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
        from langchain_huggingface import HuggingFacePipeline

        if not torch.cuda.is_available():
            raise RuntimeError("No se detectó GPU. Configura LLM_MODE = 'api' en Secrets de Streamlit.")

        gc.collect()
        torch.cuda.empty_cache()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            token=hf_token
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            return_full_text=False
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    else:
        raise ValueError(f"Modo no soportado: {mode}. Usa 'local' o 'api'.")
