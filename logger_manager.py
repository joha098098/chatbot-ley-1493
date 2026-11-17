import os
import re
import json
import uuid
import hashlib
from datetime import datetime, timezone
import pandas as pd

class AnonymizedConversationLogger:
    """
    Registra interacciones del chatbot en formato estructurado JSONL sin almacenar
    identificadores personales (PII), facilitando la analítica posterior de consultas
    frecuentes, modos de fallo y brechas de conocimiento.
    """

    def __init__(self, log_path: str = "./logs/conversations_ley1493.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    @staticmethod
    def generate_anonymous_session_id(raw_session_key: str) -> str:
        """
        Genera un hash SHA-256 corto y no reversible para agrupar turnos de una misma
        sesión sin vincular IP, nombre ni identidad del usuario.
        """
        return hashlib.sha256(raw_session_key.encode("utf-8")).hexdigest()[:12]

    @staticmethod
    def scrub_pii(text: str) -> str:
        """
        Enmascara posibles identificadores personales (correos, teléfonos, NIT, cédulas)
        antes de escribir el texto en disco.
        """
        if not text:
            return ""
        
        # 1. Enmascarar correos electrónicos
        text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", text)
        # 2. Enmascarar posibles NIT con dígito de verificación (ej. 900.123.456-7 o 900123456-7)
        text = re.sub(r"\b\d{3}\.?\d{3}\.?\d{3}-\d\b", "[NIT_REDACTED]", text)
        # 3. Enmascarar números largos de identificación o teléfonos (7 a 12 dígitos seguidos)
        text = re.sub(r"\b\d{7,12}\b", "[ID_OR_PHONE_REDACTED]", text)
        return text

    @staticmethod
    def classify_topic(query: str) -> str:
        """
        Clasifica automáticamente la intención temática de la consulta para análisis agregado.
        """
        q = query.lower()
        if any(w in q for w in ["parafiscal", "10%", "uvt", "dian", "impuesto", "boletería", "boleta"]):
            return "contribucion_parafiscal_y_uvt"
        if any(w in q for w in ["pulep", "registro", "inscripción", "requisito", "permiso", "alcaldía"]):
            return "registro_y_tramites_pulep"
        if any(w in q for w in ["excluido", "privada", "corporativo", "deporte", "cine", "arte escénica"]):
            return "alcance_y_exclusiones_ley"
        if any(w in q for w in ["infraestructura", "escenario", "teatro", "recurso", "proyecto"]):
            return "infraestructura_cultural"
        return "consulta_general_normativa"

    def log_turn(
        self,
        session_id: str,
        user_query: str,
        bot_response: str,
        retrieved_docs: list,
        latency_sec: float,
        escalated: bool,
        user_feedback: str = None
    ) -> str:
        """
        Escribe un registro estructurado en formato JSONL por cada turno conversacional.
        Retorna el ID único del evento (event_id) por si se desea asociar feedback posterior.
        """
        event_id = str(uuid.uuid4())[:8]
        clean_query = self.scrub_pii(user_query)
        clean_response = self.scrub_pii(bot_response)

        # Extraer metadatos de fuentes consultadas sin duplicar texto extenso
        sources_summary = [
            {
                "document": doc.metadata.get("source", "desconocido"),
                "page": (doc.metadata.get("page", 0) + 1) if isinstance(doc.metadata.get("page"), int) else "N/A"
            }
            for doc in retrieved_docs
        ]

        # Determinar modo de fallo si hubo escalamiento o ausencia de evidencia
        failure_mode = None
        if escalated:
            failure_mode = "no_relevant_chunks_found" if len(retrieved_docs) == 0 else "insufficient_legal_evidence"

        record = {
            "event_id": event_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "session_hash": session_id,
            "topic_category": self.classify_topic(clean_query),
            "metrics": {
                "latency_sec": round(latency_sec, 3),
                "retrieved_chunks_count": len(retrieved_docs),
                "escalated": escalated,
                "failure_mode": failure_mode,
                "user_feedback": user_feedback  # 'positive', 'negative' o None
            },
            "interaction": {
                "sanitized_query": clean_query,
                "sanitized_response": clean_response,
                "sources_cited": sources_summary
            }
        }

        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        return event_id

    def generate_analytics_summary(self) -> dict:
        """
        Lee los logs acumulados y genera un resumen ejecutivo para detectar
        temas más consultados, tasa de escalamiento y latencia promedio.
        """
        if not os.path.exists(self.log_path):
            return {"status": "Sin registros aún."}

        records = []
        with open(self.log_path, mode="r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        df = pd.json_normalize(records)
        return {
            "total_consultas": len(df),
            "latencia_promedio_seg": round(df["metrics.latency_sec"].mean(), 2),
            "tasa_escalamiento_pct": round(df["metrics.escalated"].mean() * 100, 1),
            "distribucion_temas": df["topic_category"].value_counts().to_dict(),
            "modos_de_fallo": df["metrics.failure_mode"].dropna().value_counts().to_dict()
        }
