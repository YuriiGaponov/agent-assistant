"""src.agent.graph

Модуль реализации графа агента на базе LangGraph для RAG‑цикла
Agent‑Assistant.

Граф построен по масштабируемой ReAct‑архитектуре и оркестрирует
обработку пользовательского запроса через цикл «вызов модели → решение →
вызов инструмента(ов) → возврат к модели»:

    1. Узел вызова модели (call_model) получает запрос и решает: вызвать
       один или несколько инструментов либо сразу дать ответ пользователю.
    2. Если модель решила вызвать инструмент(ы) — управление переходит в
       узел вызова инструментов (tools), который выполняет соответствующие
       инструменты.
    3. Результаты работы инструментов возвращаются в состояние, и цикл
       повторяется снова с узла вызова модели.
    4. Если модель решает дать ответ — цикл завершается, формируется
       обобщённый ответ.

Такая архитектура позволяет легко расширять список инструментов: для
добавления нового инструмента достаточно описать его в фабрике tools.py,
граф автоматически передаст его модели через привязку инструментов.

Промпты вынесены в отдельный модуль src.agent.prompts.
"""

from typing import Literal

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage
)
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.config import Config
from src.interfaces import (
    EmbeddingModelProtocol, VectorDBManagerInterface
)
from src.logger import logger
from src.agent.state import State
from src.agent.prompts import SYSTEM_PROMPT
from src.agent.tools import make_tools


def make_call_model_node(llm: ChatOllama):
    """Создаёт узел вызова модели в графе агента.

    Узел передаёт модели историю сообщений состояния. Модель сама решает,
    вызвать ли инструмент(ы) (возвращая AIMessage с tool_calls) или сразу
    дать финальный ответ.

    Args:
        llm: Экземпляр LangChain ChatOllama, к которому привязаны инструменты.

    Returns:
        Callable[[State], dict]: Узел графа, возвращающий сообщение модели.
    """
    async def call_model_node(state: State) -> dict:
        messages = state.messages
        if not messages:
            # Первый вызов модели: инициализируем системным промптом
            # и запросом пользователя.
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=state.query),
            ]
        logger.debug('вызов модели')
        response = await llm.ainvoke(messages)
        return {'messages': [response]}

    return call_model_node


def make_tools_node(tools: list):
    """Создаёт узел вызова инструментов в графе агента.

    Обёртка над стандартным ToolNode из LangGraph, который выполняет
    инструменты, запрошенные моделью (через tool_calls), и возвращает
    результаты в виде ToolMessage.

    Args:
        tools: Список инструментов LangChain, доступных агенту.

    Returns:
        ToolNode: Узел графа для выполнения инструментов.
    """
    return ToolNode(tools)


def make_should_continue():
    """Создаёт функцию-маршрутизатор, определяющую следующий шаг цикла.

    Если последнее сообщение модели содержит вызовы инструментов
    (tool_calls) — управление переходит в узел вызова инструментов,
    иначе — в узел финального ответа.

    Returns:
        Callable[[State], str]: Функция маршрутизации состояния.
    """
    def should_continue(state: State) -> Literal['tools', 'final']:
        last_message = state.messages[-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return 'tools'
        return 'final'

    return should_continue


def make_final_node():
    """Создаёт узел финального ответа в графе агента.

    Извлекает финальный ответ из последнего сообщения модели и сохраняет
    его в поле state.answer.

    Returns:
        Callable[[State], dict]: Узел графа, возвращающий обобщённый ответ.
    """
    async def final_node(state: State) -> dict:
        last_message = state.messages[-1]
        answer = last_message.content if hasattr(last_message, 'content') else str(last_message)
        logger.info(f'сформирован ответ: {answer}')
        return {'answer': answer}

    return final_node


def build_graph(
        config: Config,
        embedding_model: EmbeddingModelProtocol,
        vector_db: VectorDBManagerInterface
):
    """Собирает и компилирует граф агента для RAG‑цикла.

    Строит ReAct‑граф на основе модели состояния State:
        START → call_model → (should_continue)
                    ├─ tools → call_model (цикл)
                    └─ final → END

    Создаёт инструменты через make_tools и привязывает их к модели
    через bind_tools, что позволяет модели самостоятельно принимать
    решение о вызове инструментов.

    Args:
        config: Конфигурация проекта.
        embedding_model: Модель эмбеддингов (EmbeddingModelProtocol).
        vector_db: Менеджер векторной БД (VectorDBManagerInterface).

    Returns:
        CompiledStateGraph: Скомпилированный граф LangGraph.
    """
    tools = make_tools(embedding_model, vector_db)
    llm = ChatOllama(model=config.MAIN_LLM.value).bind_tools(tools)

    graph = StateGraph(State)
    graph.add_node('call_model', make_call_model_node(llm))
    graph.add_node('tools', make_tools_node(tools))
    graph.add_node('final', make_final_node())

    graph.add_edge(START, 'call_model')
    graph.add_conditional_edges(
        'call_model',
        make_should_continue(),
        {'tools': 'tools', 'final': 'final'},
    )
    graph.add_edge('tools', 'call_model')
    graph.add_edge('final', END)

    compiled = graph.compile()
    logger.info('граф агента собран и скомпилирован')
    return compiled
