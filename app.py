import os
import json
import uuid
from datetime import datetime
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config_loader import load_config
from llm_connector import get_open_llm
from memory_manager import BoundedConversationMemory
from rag_chain import Ley1493RAGPipeline
from logger_manager import AnonymizedConversationLogger

# Sincronizar secretos de Streamlit Cloud con variables de entorno (si existen)
for secret_key in ["HF_TOKEN", "GROQ_API_KEY", "LLM_MODE", "LLM_MODEL_ID"]:
    if secret_key in st.secrets and not os.getenv(secret_key):
        os.environ[secret_key] = str(st.secrets[secret_key])

# Cargar configuración centralizada
cfg = load_config("config.yaml")

st.set_page_config(
    page_title=cfg["app"]["title"],
    page_icon=cfg["app"]["page_icon"],
    layout=cfg["app"]["layout"],
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stMarkdown p, .stMarkdown li {
        font-size: 1.02rem !important;
        line-height: 1.6 !important;
    }
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

@st.cache_resource(show_spinner="Cargando base documental (Ley 1493 de 2011 y PULEP)...")
def init_vector_store():
    emb_cfg = cfg["embeddings"]
    kb_cfg = cfg["knowledge_base"]
    index_path = kb_cfg["vector_store_path"]

    embeddings = HuggingFaceEmbeddings(
        model_name=emb_cfg["model_id"],
        model_kwargs={"device": emb_cfg["device"]},
        encode_kwargs={"normalize_embeddings": emb_cfg["normalize_embeddings"]}
    )

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        return FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
    else:
        from knowledge_base import load_and_clean_pdfs, segment_documents
        raw_docs = load_and_clean_pdfs(docs_folder=kb_cfg["docs_folder"])
        chunks = segment_documents(
            raw_docs,
            chunk_size=kb_cfg["chunk_size"],
            chunk_overlap=kb_cfg["chunk_overlap"]
        )
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(index_path)
        return vector_store

@st.cache_resource(show_spinner="Inicializando modelo de lenguaje abierto...")
def init_llm():
    return get_open_llm(llm_config=cfg["llm"])

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

log_path = cfg["security_and_logging"]["log_file_path"]
logger = AnonymizedConversationLogger(log_path=log_path)

if "session_hash" not in st.session_state:
    st.session_state.session_hash = logger.generate_anonymous_session_id(str(uuid.uuid4()))

if "memory" not in st.session_state:
    st.session_state.memory = BoundedConversationMemory(
        max_turns=cfg["memory"]["max_turns"],
        max_chars=cfg["memory"]["max_chars"]
    )

if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = []

# --- BARRA LATERAL CON GUÍA DE USO Y AVISO DE LIMITACIONES ---
with st.sidebar:
    st.title("🎭 Guía Ley 1493")
    st.markdown("Asistente virtual para productores y gestores de espectáculos públicos de las artes escénicas en Colombia.")
    
    st.divider()
    st.subheader("💬 Control de Conversación")
    stats = st.session_state.memory.get_stats()
    st.markdown(f"- **Turnos recordados:** {stats['active_turns']} de {stats['max_turns']}")
    st.markdown(f"- **Contexto activo:** ~{stats['approx_tokens']} tokens")
    
    if st.button("🔄 Iniciar nueva conversación", use_container_width=True):
        st.session_state.memory.clear()
        st.session_state.ui_messages = []
        st.session_state.session_hash = logger.generate_anonymous_session_id(str(uuid.uuid4()))
        st.rerun()

    st.divider()
    with st.expander("📌 Capacidades y Limitaciones", expanded=True):
        st.markdown(
            "**✅ Qué puede hacer:**\n"
            "- Explicar requisitos de inscripción y registro en el **PULEP**.\n"
            "- Orientar sobre el cálculo y umbral (3 UVT) de la **Contribución Parafiscal Cultural (10 %)**.\n"
            "- Aclarar qué eventos están cubiertos o excluidos por la **Ley 1493 de 2011**.\n\n"
            "**⛔ Limitaciones:**\n"
            "- No cubre eventos deportivos, cine, ni fiestas privadas cerradas.\n"
            "- No realiza trámites ni revisa documentos personales."
        )

    with st.expander("📊 Métricas de Uso (Anonimizadas)"):
        st.json(logger.generate_analytics_summary())

    st.divider()
    st.warning(
        "**⚠️ Aviso sobre posibles errores y alcance legal:** Este chatbot utiliza Inteligencia Artificial "
        "y puede cometer errores, omisiones o imprecisiones (*alucinaciones*). La información entregada es "
        "estrictamente orientativa y **no constituye asesoría jurídica ni tributaria vinculante**. "
        "Verifica siempre las fuentes citadas o consulta directamente con el Ministerio de las Culturas y la DIAN."
    )

# --- ENCABEZADO Y GUÍA INICIAL EN PANTALLA PRINCIPAL ---
st.title("Consulta Normativa: Artes Escénicas (Ley 1493 de 2011)")
st.caption("Orientación basada en la Ley 1493 de 2011, decretos reglamentarios e instructivos oficiales del PULEP.")

# Aviso permanente visible sobre posibles errores / alucinaciones
st.info(
    "ℹ️ **Nota importante:** Este asistente de IA responde con base en documentos normativos cargados, "
    "pero puede presentar imprecisiones o interpretaciones incompletas. Verifica siempre el artículo citado "
    "en la sección desplegable de cada respuesta. No ingreses datos personales (cédula, NIT o contraseñas)."
)

vector_db = init_vector_store()
llm = init_llm()
rag_pipeline = Ley1493RAGPipeline(
    vector_store=vector_db,
    llm=llm,
    top_k=cfg["retrieval"]["top_k"],
    score_threshold=cfg["retrieval"]["score_threshold"]
)

# Panel de bienvenida: Capacidades, Limitaciones y Prompts Recomendados
if not st.session_state.ui_messages:
    col_cap, col_lim = st.columns(2)
    with col_cap:
        st.success(
            "**✅ Capacidades del Asistente**\n"
            "- Guía paso a paso para registro de productores y eventos en **PULEP**.\n"
            "- Cálculo y reglas de la **Contribución Parafiscal del 10 %** (boletas ≥ 3 UVT).\n"
            "- Citas exactas del documento y página consultada."
        )
    with col_lim:
        st.error(
            "**⛔ Limitaciones del Sistema**\n"
            "- No abarca cine, deportes, reinados ni eventos corporativos privados.\n"
            "- Si una consulta excede los PDF cargados, derivará a canales oficiales.\n"
            "- Puede generar errores de interpretación propios de modelos de lenguaje."
        )

    st.markdown("### 💡 Prompts recomendados (haz clic para probar)")
    cols = st.columns(3)
    ejemplos = [
        "Soy productor ocasional persona natural y mis boletas cuestan 4 UVT. ¿Debo pagar el 10% parafiscal?",
        "¿Qué documentos y plazos exige el PULEP para registrar una temporada de teatro?",
        "¿Una fiesta privada empresarial con música en vivo se rige por la Ley 1493 de 2011?"
    ]
    for idx, ej in enumerate(ejemplos):
        if cols[idx].button(ej, use_container_width=True):
            st.session_state.pending_prompt = ej
            st.rerun()
    st.divider()

# Renderizar historial multiturno
for idx, msg in enumerate(st.session_state.ui_messages):
    is_user = msg["role"] == "user"
    avatar_icon = "🧑‍💼" if is_user else "🎭"
    
    with st.chat_message(msg["role"], avatar=avatar_icon):
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

        if not is_user:
            if msg.get("sources"):
                with st.expander("📚 Consultar artículos y fuentes citadas"):
                    st.markdown(msg["sources"])
            
            event_id = msg.get("event_id")
            if event_id:
                col_fb1, col_fb2, _ = st.columns([1, 1, 6])
                if col_fb1.button("👍 Útil", key=f"up_{event_id}_{idx}"):
                    update_feedback_in_log(log_path, event_id, "positive")
                    st.toast("Valoración positiva guardada.")
                if col_fb2.button("👎 Mejorar", key=f"down_{event_id}_{idx}"):
                    update_feedback_in_log(log_path, event_id, "negative")
                    st.toast("Gracias, revisaremos este caso.")

# Entrada de chat
user_input = st.chat_input("Escribe tu consulta (ej. Indica tipo de productor, espectáculo y valor en UVT)...")
if "pending_prompt" in st.session_state:
    user_input = st.session_state.pop("pending_prompt")

if user_input:
    current_time = datetime.now().strftime("%H:%M")

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
        "event_id": event_id
    })
    st.rerun()
