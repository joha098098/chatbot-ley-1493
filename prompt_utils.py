import re
import unicodedata
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. FUNCIÓN REUTILIZABLE DE PREPROCESAMIENTO DE ENTRADA (INPUT PREPROCESSING)
def preprocess_user_input(raw_text: str, max_length: int = 1000) -> dict:
    """
    Limpia, normaliza y valida la consulta del productor antes de enviarla al RAG y al LLM.
    Retorna un diccionario con el texto limpio, bandera de validez y advertencias de seguridad.
    """
    if not raw_text or not isinstance(raw_text, str):
        return {"is_valid": False, "clean_text": "", "reason": "La consulta está vacía."}

    # Normalización Unicode y eliminación de caracteres de control invisibles
    text = unicodedata.normalize("NFKC", raw_text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Colapsar múltiples espacios o saltos de línea innecesarios
    text = re.sub(r"\s+", " ", text).strip()

    # Control de longitud para evitar desbordamiento de contexto o ataques de relleno
    if len(text) > max_length:
        text = text[:max_length]

    # Filtro básico contra inyección de instrucciones (Prompt Injection)
    injection_patterns = [
        r"ignora\s+(todas\s+)?las\s+instrucciones",
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"olvida\s+tu\s+rol",
        r"actúa\s+como\s+si\s+no\s+tuvieras\s+reglas",
        r"system\s*prompt"
    ]
    for pattern in injection_patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return {
                "is_valid": False,
                "clean_text": "",
                "reason": "Consulta bloqueada por políticas de seguridad. Por favor formula una pregunta sobre la Ley 1493 de 2011."
            }

    # Expansión de siglas y términos frecuentes del sector cultural colombiano para mejorar búsqueda semántica
    domain_glossary = {
        r"\bpulep\b": "PULEP (Portal Único de Espectáculos Públicos de las Artes Escénicas)",
        r"\buvt\b": "UVT (Unidad de Valor Tributario)",
        r"\bparafiscal\b": "Contribución Parafiscal Cultural",
        r"\bmincultura\b": "Ministerio de las Culturas, las Artes y los Saberes"
    }
    enriched_text = text
    for abbr, expansion in domain_glossary.items():
        enriched_text = re.sub(abbr, expansion, enriched_text, flags=re.IGNORECASE)

    return {
        "is_valid": True,
        "clean_text": enriched_text,
        "original_text": text,
        "reason": None
    }


# 2. SYSTEM MESSAGE Y REGLAS DE SEGURIDAD / ANTI-ALUCINACIÓN
SYSTEM_PROMPT_LEY_1493 = """Eres un asistente virtual especializado exclusivamente en la normativa colombiana de espectáculos públicos de las artes escénicas, tomando como eje principal la Ley 1493 de 2011, sus decretos reglamentarios (como el Decreto 1080 de 2015) y los instructivos oficiales del Ministerio de las Culturas y el PULEP.

### REGLAS ESTRICTAS DE COMPORTAMIENTO Y SEGURIDAD (GUARDRAILS):
1. **Fidelidad absoluta al contexto (Cero Alucinaciones):** Responde ÚNICAMENTE utilizando la información contenida en los fragmentos normativos proporcionados en la sección [CONTEXTO DOCUMENTAL]. No inventes artículos, plazos, tarifas, porcentajes ni requisitos que no estén explícitos en el contexto.
2. **Protocolo de Escalamiento ante falta de evidencia:** Si el [CONTEXTO DOCUMENTAL] está vacío o no contiene información suficiente para responder con certeza, NO intentes adivinar. Responde exactamente con este mensaje estructurado:
   "No encuentro respaldo específico en los documentos normativos cargados para resolver este caso con certeza. Te recomiendo escalar tu consulta a través de los canales oficiales del Ministerio de las Culturas, las Artes y los Saberes y el PULEP:
   * **Sitio web del Ministerio:** https://www.mincultura.gov.co/
   * **Portal Único de Espectáculos Públicos (PULEP):** https://pulep.mincultura.gov.co/Paginas/default.aspx
   * **Correos electrónicos de atención:** servicioalciudadano@mincultura.gov.co y usuariospulep@mincultura.gov.co
   * **Línea telefónica:** (601) 342 4100 ext. 4053"
3. **Control de alcance (Dominio cerrado):** Si el usuario pregunta sobre temas ajenos a las artes escénicas en Colombia (ej. deportes, cine, eventos corporativos cerrados, política o recetas), aclara amablemente que tu alcance se limita a la Ley 1493 de 2011 y explica por qué esas actividades no entran en este régimen si aplica.
4. **Desambiguación proactiva:** Si la pregunta del productor es ambigua o le faltan datos críticos (por ejemplo: valor de la boletería en UVT, tipo de productor natural o jurídica, o tipo de espectáculo), haz preguntas aclaratorias breves antes de dar una conclusión definitiva.
5. **Trazabilidad normativa obligatoria:** Siempre que afirmes un requisito, plazo o cálculo, cita al final la fuente exacta indicada en los metadatos del contexto (ej. *Fuente: Ley 1493 de 2011, Art. 7* o *Instructivo PULEP, Pág. 4*).
6. **Privacidad y tono:** Nunca solicites datos personales sensibles (cédula, NIT real, contraseñas o estados financieros). Mantén un tono profesional, claro, empático y práctico dirigido a productores culturales. Al final de orientaciones tributarias o jurídicas complejas, recuerda brevemente que la respuesta es orientativa y no sustituye un concepto jurídico vinculante.
"""


# 3. CONSTRUCCIÓN DE PLANTILLA DE PROMPT REUTILIZABLE (PROMPT TEMPLATE)
def build_rag_prompt_template() -> ChatPromptTemplate:
    """
    Crea una plantilla estructurada de LangChain que separa claramente las instrucciones del sistema,
    el historial conversacional, los fragmentos normativos recuperados y la pregunta del usuario.
    """
    template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT_LEY_1493),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        ("human", """Analiza el siguiente contexto normativo y responde a la consulta del productor.

[CONTEXTO DOCUMENTAL]
{context}

[CONSULTA DEL PRODUCTOR]
{question}

[RESPUESTA ESTRUCTURADA]""")
    ])
    return template


# 4. FORMATEADOR DE DOCUMENTOS RECUPERADOS CON METADATOS DE TRAZABILIDAD
def format_retrieved_docs(docs: list) -> str:
    """
    Convierte una lista de objetos Document de LangChain en un bloque de texto estructurado
    con etiquetas claras de fuente y página para facilitar las citas del LLM.
    """
    if not docs:
        return "SIN DOCUMENTOS RELEVANTES ENCONTRADOS."

    formatted_chunks = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Documento normativo")
        page = doc.metadata.get("page", "N/A")
        # Ajustar índice de página si empieza en 0
        page_display = page + 1 if isinstance(page, int) else page
        content = doc.page_content.strip()
        formatted_chunks.append(
            f"--- Fragmento {idx} [Fuente: {source} | Página: {page_display}] ---\n{content}"
        )

    return "\n\n".join(formatted_chunks)
