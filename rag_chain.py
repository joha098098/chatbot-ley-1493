import time
from langchain_community.vectorstores import FAISS
from prompt_utils import (
    preprocess_user_input,
    build_rag_prompt_template,
    format_retrieved_docs
)
from memory_manager import BoundedConversationMemory

class Ley1493RAGPipeline:
    """
    Orquestador del flujo Retrieval-Augmented Generation (RAG):
    1. Preprocesa la consulta del usuario.
    2. Recupera los fragmentos más relevantes de FAISS filtrando por umbral de similitud.
    3. Ensambla el prompt separando explícitamente:
       - System Instructions (reglas y guardrails)
       - Conversational Memory (historial acotado)
       - Retrieved Context (artículos de la Ley 1493 y decretos)
       - User Input (pregunta del productor)
    4. Invoca el LLM abierto y retorna la respuesta junto con sus fuentes trazables.
    """

    def __init__(self, vector_store: FAISS, llm, top_k: int = 4, score_threshold: float = 1.15):
        self.vector_store = vector_store
        self.llm = llm
        self.top_k = top_k
        self.score_threshold = score_threshold
        self.prompt_template = build_rag_prompt_template()

    def retrieve_context(self, query: str) -> tuple[list, str]:
        """
        Paso de Recuperación (Retrieval):
        Busca los Top-K fragmentos en FAISS y descarta aquellos cuya distancia L2
        supere el umbral de relevancia (evitando inyectar ruido al LLM).
        """
        docs_with_scores = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        
        # Filtrar por relevancia semántica (menor distancia L2 = mayor similitud en vectores normalizados)
        relevant_docs = [
            doc for doc, score in docs_with_scores if score <= self.score_threshold
        ]
        
        formatted_context = format_retrieved_docs(relevant_docs)
        return relevant_docs, formatted_context

    def generate_answer(self, raw_user_input: str, memory: BoundedConversationMemory) -> dict:
        """
        Ejecuta el ciclo completo RAG de extremo a extremo.
        """
        start_time = time.time()

        # 1. Preprocesamiento y validación de seguridad de la entrada
        prep = preprocess_user_input(raw_user_input)
        if not prep["is_valid"]:
            return {
                "answer": f"⚠️ {prep['reason']}",
                "sources": [],
                "formatted_context": "",
                "escalated": False,
                "latency": round(time.time() - start_time, 2)
            }

        clean_query = prep["clean_text"]

        # 2. Paso de Recuperación (Retrieval) desde la base vectorial FAISS
        relevant_docs, formatted_context = self.retrieve_context(clean_query)

        # 3. Inyección estructurada en el Prompt Template (Separación clara de bloques)
        structured_messages = self.prompt_template.invoke({
            "chat_history": memory.get_history(),
            "context": formatted_context,
            "question": clean_query
        })

        # 4. Paso de Generación (Generation) con el LLM abierto
        ai_response = self.llm.invoke(structured_messages)
        answer_text = ai_response.content

        # 5. Actualizar memoria conversacional de corto plazo
        memory.add_user_message(raw_user_input)
        memory.add_ai_message(answer_text)

        # Verificar si se activó el protocolo de escalamiento por falta de evidencia
        escalated = (len(relevant_docs) == 0) or ("No encuentro respaldo específico" in answer_text)

        return {
            "answer": answer_text,
            "sources": relevant_docs,
            "formatted_context": formatted_context,
            "escalated": escalated,
            "latency": round(time.time() - start_time, 2),
            "debug_prompt": structured_messages  # Útil para auditar la separación de roles
        }
