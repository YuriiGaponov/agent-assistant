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

Граф не зависит от реализации менеджера LLM: он получает готовый объект,
реализующий LLMManagerInterface (с уже привязанными инструментами), и
список инструментов для узла ToolNode. Такая архитектура позволяет легко
расширять список инструментов и подключать новых провайдеров LLM без
изменения графа.

Промпты вынесены в отдельный модуль src.agent.prompts.
"""

from typing import Literal

from langchain_core.messages import (
    AIMessage, HumanMessage, SystemMessage
)
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.interfaces import LLMManagerInterface
from src.logger import logger
from src.agent.state import State
from src.agent.prompts import SYSTEM_PROMPT


def make_call_model_node(llm: LLMManagerInterface):
    """Создаёт узел вызова модели в графе агента.

    Узел передаёт модели историю сообщений состояния. Модель сама решает,
    вызвать ли инструмент(ы) (возвращая AIMessage с tool_calls) или сразу
    дать финальный ответ.

    Args:
        llm: Менеджер LLM (LLMManagerInterface), к которому привязаны
            инструменты.

    Returns:
        Callable[[State], dict]: Узел графа, возвращающий сообщение модели.
    """
    async def call_model_node(state: State) -> dict:
        state.step = 'call_model'
        logger.info('узел call_model', step='call_model', state=state)
        messages = state.messages
        if not messages:
            # Первый вызов модели: инициализируем системным промптом
            # и запросом пользователя.
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=state.query),
            ]
        logger.info('вызов модели', step='call_model', messages=messages)
        response = await llm.ainvoke(messages)
        logger.info('ответ модели', step='call_model', response=response)
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
            state.step = 'tool_call'
            logger.info('переход в узел tools', state=state)
            return 'tools'
        state.step = 'final'
        logger.info('переход в узел final', state=state)
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
        state.step = 'final'
        logger.info('узел final', step='final', state=state)
        last_message = state.messages[-1]
        answer = (
            last_message.content
            if hasattr(last_message, 'content')
            else str(last_message)
        )
        logger.info('сформирован итоговый ответ', step='final', answer=answer)
        return {'answer': answer}

    return final_node


def build_graph(llm: LLMManagerInterface, tools: list):
    """Собирает и компилирует граф агента для RAG‑цикла.

    Строит ReAct‑граф на основе модели состояния State:
        START → call_model → (should_continue)
                    ├─ tools → call_model (цикл)
                    └─ final → END

    Граф не создаёт модель и не привязывает инструменты самостоятельно —
    он получает готовый менеджер LLM (LLMManagerInterface) с привязанными
    инструментами и список инструментов для узла ToolNode.

    Args:
        llm: Менеджер LLM (LLMManagerInterface) с привязанными инструментами.
        tools: Список инструментов LangChain, доступных агенту.

    Returns:
        CompiledStateGraph: Скомпилированный граф LangGraph.
    """
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
