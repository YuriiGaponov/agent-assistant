"""src.agent.state

Модуль реализации модели состояния агента.
"""

from typing import Annotated

from pydantic import BaseModel, Field
from langgraph.graph.message import add_messages


class State(BaseModel):
    """Модель состояния агента для RAG‑сервиса Agent‑Assistant.

    Хранит всю необходимую информацию для оркестрации шагов агента:
    запрос пользователя, историю сообщений цикла (сообщения модели,
    вызовы инструментов и результаты), найденный контекст и итоговый
    обобщённый ответ.

    Поле messages использует редуктор add_messages для корректного
    накопления истории сообщений в ReAct‑цикле.
    """

    query: str
    """запрос пользователя"""

    messages: Annotated[list, add_messages] = Field(default_factory=list)
    """история сообщений ReAct‑цикла: сообщения модели, вызовы
    инструментов и результаты их выполнения"""

    answer: str = ''
    """обобщённый ответ агента, сформированный на основе контекста"""
