from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

class BoundedConversationMemory:
    """
    Gestor de memoria conversacional de corto plazo con ventana deslizante (Rolling Window)
    y límite máximo estricto de caracteres/tokens para evitar desbordamiento de contexto.
    """
    def __init__(self, max_turns: int = 4, max_chars: int = 3000):
        """
        Args:
            max_turns: Número máximo de pares pregunta-respuesta (turnos) a conservar.
                       Ejemplo: max_turns=4 conserva los últimos 8 mensajes (4 usuario + 4 IA).
            max_chars: Límite máximo total de caracteres permitidos en el historial activo
                       (~750 tokens), garantizando espacio suficiente para el System Prompt
                       y los fragmentos normativos del RAG.
        """
        self.max_turns = max_turns
        self.max_chars = max_chars
        self.messages: list[BaseMessage] = []

    def add_user_message(self, content: str) -> None:
        """Agrega un mensaje del usuario y aplica las reglas de recorte."""
        self.messages.append(HumanMessage(content=content))
        self._enforce_limits()

    def add_ai_message(self, content: str) -> None:
        """Agrega una respuesta del asistente y aplica las reglas de recorte."""
        self.messages.append(AIMessage(content=content))
        self._enforce_limits()

    def _enforce_limits(self) -> None:
        """
        Aplica dos niveles de control:
        1. Ventana deslizante por turnos (max_turns * 2 mensajes).
        2. Recorte por presupuesto máximo de longitud (max_chars) eliminando los mensajes más antiguos.
        """
        max_messages = self.max_turns * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

        # Verificar que la suma total de caracteres no supere el presupuesto máximo
        while self._total_chars() > self.max_chars and len(self.messages) > 1:
            # Eliminar el mensaje más antiguo (FIFO)
            self.messages.pop(0)

    def _total_chars(self) -> int:
        """Calcula la longitud acumulada de todos los mensajes en memoria."""
        return sum(len(msg.content) for msg in self.messages)

    def get_history(self) -> list[BaseMessage]:
        """Retorna la lista de mensajes lista para inyectar en MessagesPlaceholder de LangChain."""
        return self.messages

    def get_stats(self) -> dict:
        """Retorna métricas de uso de la memoria para monitoreo o visualización en la UI."""
        total_chars = self._total_chars()
        return {
            "active_messages": len(self.messages),
            "active_turns": len(self.messages) // 2,
            "max_turns": self.max_turns,
            "used_chars": total_chars,
            "max_chars": self.max_chars,
            "approx_tokens": total_chars // 4  # Estimación estándar: ~4 caracteres por token en español
        }

    def clear(self) -> None:
        """Reinicia la memoria de la sesión."""
        self.messages = []
