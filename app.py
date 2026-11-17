import os
import json
import uuid
from datetime import datetime
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from llm_connector import get_open_llm
from memory_manager import BoundedConversationMemory
from rag_chain import Ley1493RAGPipeline
from logger_manager import AnonymizedConversationLogger

# 1. CONFIGURACIÓN DE PÁGINA Y ACCESIBILIDAD
st.set_page_config(
    page_title="Asistente Normativo Ley 1493 de 2011 | Artes Escénicas",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para mejorar contraste, legibilidad y distinción de roles
st.markdown("""
<style>
    /* Mejorar tamaño base de fuente y espaciado entre líneas para lectura cómoda */
    .stMarkdown p, .stMarkdown li {
        font-size: 1.02rem !important;
        line-height: 1.6 !important;
    }
    /* Etiquetas de encabezado de turno accesibles */
    .role-badge-user {
        display: inline-block;
        background-color: #1E3A8A;
        color: #FFFFFF;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .role-badge-bot {
        display: inline-block;
        background-color: #065F46;
        color: #FFFFFF;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 6px;
    }
    .timestamp-text {
        color: #6B7280;
        font-size: 0.78rem;
        margin-left: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 2. CARGA EN CACHÉ DE MOTORES RAG Y LLM
@st.cache_resource(show_spinner="Cargando base documental (Ley 1493 de 2011 y PULEP)...")
def init_vector_store():
    embeddings_cpu = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return FAISS.load_local(
        "./faiss_index_ley1493",
        embeddings_cpu,
        allow_dangerous_deserialization=True
    )

@st.cache_resource(show_spinner="Iniciando modelo de lenguaje abierto en GPU...")
def init_llm():
    os.environ["LLM_MODE"] = "local"
    os.environ["LLM_MODEL_ID"] = "Qwen/Qwen2.5-7B-Instruct"
    return get_open_llm()

def update_feedback_in_log(log_path: str, event_id: str, feedback_value: str):
    if not os.path.exists(log_path):
        return
    updated_lines = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("event_id") == event_id:
                record["metrics"]["user_feedback"] = feedback_value
            updated_lines.append(json.dumps(record, ensure_ascii=False))
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")

# 3. GESTIÓN DE ESTADO MULTITURNO
logger = AnonymizedConversationLogger(log_path="./logs/conversations_ley1493.jsonl")

if "session_hash" not in st.session_state:
    st.session_state.session_hash = logger.generate_anonymous_session_id(str(uuid.uuid4()))

if "memory" not in st.session_state:
    st.session_state.memory = BoundedConversationMemory(max_turns=4, max_chars=3000)

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# 4. BARRA LATERAL ACCESIBLE (Controles de Sesión, Guía y Privacidad)
with st.sidebar:
    st.title("🎭 Guía Ley 1493")
    st.markdown("Asistente virtual para productores y gestores de espectáculos públicos de las artes escénicas en Colombia.")
    
    st.divider()
    st.subheader("💬 Control de Conversación")
    stats = st.session_state.memory.get_stats()
    st.markdown(f"- **Turnos recordados:** {stats['active_turns']} de {stats['max_turns']}")
    st.markdown(f"- **Contexto activo:** ~{stats['approx_tokens']} tokens")
    
    if st.button("🔄 Iniciar nueva conversación", use_container_width=True, help="Borra el historial actual y reinicia el contexto"):
        st.session_state.memory.clear()
        st.session_state.ui_messages = []
        st.session_state.session_hash = logger.generate_anonymous_session_id(str(uuid.uuid4()))
        st.rerun()

    st.divider()
    with st.expander("ℹ️ Cómo hacer mejores preguntas"):
        st.markdown(
            "1. Indica si eres **persona natural o jurídica**.\n"
            "2. Menciona el **valor de las boletas** (en pesos o UVT).\n"
            "3. Especifica el **tipo de espectáculo** (teatro, danza, música en vivo, circo)."
        )

    with st.expander("📊 Métricas de Uso (Anonimizadas)"):
        st.json(logger.generate_analytics_summary())

    st.divider()
    st.info(
        "**Privacidad y Alcance:** Este sistema ofrece orientación informativa basada en la Ley 1493 de 2011 "
        "y documentos del PULEP. No reemplaza asesoría legal oficial. Los datos personales (NIT, correos o teléfonos) "
        "se eliminan automáticamente de los registros."
    )

# 5. ENCABEZADO PRINCIPAL Y PREGUNTAS SUGERIDAS (ONBOARDING)
st.title("Consulta Normativa: Artes Escénicas (Ley 1493 de 2011)")
st.caption("Resuelve dudas sobre contribución parafiscal cultural, registro en el PULEP, permisos y exclusiones de ley.")

vector_db = init_vector_store()
llm = init_llm()
rag_pipeline = Ley1493RAGPipeline(vector_store=vector_db, llm=llm, top_k=4, score_threshold=1.15)

if not st.session_state.ui_messages:
    st.markdown("### 👋 ¿Por dónde quieres empezar?")
    st.markdown("Selecciona una consulta frecuente o escribe tu propia pregunta en el cuadro inferior:")
    cols = st.columns(3)
    ejemplos = [
        "¿Cuándo debo pagar el 10% de contribución parafiscal cultural?",
        "¿Qué requisitos pide el PULEP para registrar un espectáculo?",
        "¿Un evento corporativo privado se rige por la Ley 1493?"
    ]
    for idx, ej in enumerate(ejemplos):
        if cols[idx].button(ej, use_container_width=True):
            st.session_state.pending_prompt = ej
            st.rerun()
    st.divider()

# 6. RENDERIZADO DE HISTORIAL MULTITURNO (Distinción clara Usuario vs. Chatbot)
for idx, msg in enumerate(st.session_state.ui_messages):
    is_user = msg["role"] == "user"
    avatar_icon = "🧑‍💼" if is_user else "🎭"
    
    with st.chat_message(msg["role"], avatar=avatar_icon):
        # Etiqueta explícita de rol + hora para accesibilidad visual y lectores de pantalla
        if is_user:
            st.markdown(
                f'<span class="role-badge-user">Productor / Usuario</span>'
                f'<span class="timestamp-text">{msg.get("time", "")}</span>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<span class="role-badge-bot">Asistente Ley 1493</span>'
                f'<span class="timestamp-text">{msg.get("time", "")}</span>',
                unsafe_allow_html=True
            )
            
        st.markdown(msg["content"])

        # Detalles adicionales solo para turnos del asistente
        if not is_user:
            if msg.get("sources"):
                with st.expander("📚 Consultar artículos y fuentes citadas"):
                    st.markdown(msg["sources"])
            
            event_id = msg.get("event_id")
            if event_id:
                col_fb1, col_fb2, _ = st.columns([1, 1, 10])
                if col_fb1.button("👍 Útil", key=f"up_{event_id}_{idx}"):
                    update_feedback_in_log("./logs/conversations_ley1493.jsonl", event_id, "positive")
                    st.toast("Valoración positiva guardada.")
                if col_fb2.button("👎 Mejorar", key=f"down_{event_id}_{idx}"):
                    update_feedback_in_log("./logs/conversations_ley1493.jsonl", event_id, "negative")
                    st.toast("Gracias, revisaremos este caso.")

# 7. ENTRADA DE NUEVO MENSAJE Y RESPUESTA EN VIVO
user_input = st.chat_input("Escribe tu consulta aquí (ej. ¿Qué pasa si mis boletas valen menos de 3 UVT?)...")
if "pending_prompt" in st.session_state:
    user_input = st.session_state.pop("pending_prompt")

if user_input:
    current_time = datetime.now().strftime("%H:%M")
    
    # Mostrar y guardar turno del usuario
    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(
            f'<span class="role-badge-user">Productor / Usuario</span>'
            f'<span class="timestamp-text">{current_time}</span>',
            unsafe_allow_html=True
        )
        st.markdown(user_input)
        
    st.session_state.ui_messages.append({
        "role": "user",
        "content": user_input,
        "time": current_time
    })

    # Generar y mostrar turno del chatbot
    with st.chat_message("assistant", avatar="🎭"):
        bot_time = datetime.now().strftime("%H:%M")
        st.markdown(
            f'<span class="role-badge-bot">Asistente Ley 1493</span>'
            f'<span class="timestamp-text">{bot_time}</span>',
            unsafe_allow_html=True
        )
        with st.spinner("Consultando la Ley 1493 de 2011 y sus decretos..."):
            result = rag_pipeline.generate_answer(user_input, st.session_state.memory)
            answer_text = result["answer"]
            
            st.markdown(answer_text)
            if result["sources"]:
                with st.expander("📚 Consultar artículos y fuentes citadas"):
                    st.markdown(result["formatted_context"])

    # Registrar interacción en el logger anonimizado
    event_id = logger.log_turn(
        session_id=st.session_state.session_hash,
        user_query=user_input,
        bot_response=answer_text,
        retrieved_docs=result["sources"],
        latency_sec=result["latency"],
        escalated=result["escalated"]
    )

    st.session_state.ui_messages.append({
        "role": "assistant",
        "content": answer_text,
        "time": bot_time,
        "sources": result["formatted_context"] if result["sources"] else None,
