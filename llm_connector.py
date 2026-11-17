import os
import gc
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEndpoint, ChatHuggingFace

def get_open_llm():
    # Usamos Qwen2.5-7B-Instruct por defecto (o Qwen2.5-3B-Instruct si la GPU está muy llena)
    model_id = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    mode = os.getenv("LLM_MODE", "local").lower()
    hf_token = os.getenv("HF_TOKEN")

    if mode == "api":
        if not hf_token:
            raise ValueError("Configura la variable de entorno HF_TOKEN para usar modo 'api'.")
        endpoint = HuggingFaceEndpoint(
            repo_id=model_id,
            task="text-generation",
            max_new_tokens=512,
            temperature=0.1,
            repetition_penalty=1.1,
            huggingfacehub_api_token=hf_token
        )
        return ChatHuggingFace(llm=endpoint)

    elif mode == "local":
        if not torch.cuda.is_available():
            raise RuntimeError("Activa la GPU T4 en Colab: Entorno de ejecución > Cambiar tipo de entorno > GPU T4.")
        
        # Liberar memoria caché residual de PyTorch antes de cargar el LLM
        gc.collect()
        torch.cuda.empty_cache()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            llm_int8_enable_fp32_cpu_offload=True  # Permite respaldo en CPU si falta VRAM
        )

        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            max_memory={0: "13GiB", "cpu": "12GiB"},  # Reserva espacio seguro en la GPU T4
            token=hf_token
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            repetition_penalty=1.1,
            return_full_text=False
        )
        return ChatHuggingFace(llm=HuggingFacePipeline(pipeline=pipe))

    else:
        raise ValueError(f"Modo no soportado: {mode}. Usa 'local' o 'api'.")
