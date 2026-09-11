"""Customer Support MAS: Supervisor/Router архітектура на LangGraph.

Архітектура (див. MAS_structure.md, Крок 2 — адаптація прикладу до
реального домену цього проєкту):

    triage_agent (супервізор, без доступу до ризикових tools)
        -> order_agent     (search_order, get_order_ship_date,
                             track_parcel, track_ukrposhta_parcel)
        -> knowledge_agent (search_knowledge — RAG по базі знань)
        -> refund_agent    (notify_accountant — РИЗИКОВИЙ tool,
                             потребує підтвердження оператора)

Handoff matrix (структурно, через ребра графа — інших шляхів немає):
    triage  -> order, knowledge, refund
    order, knowledge, refund -> triage

order_agent НЕ МОЖЕ напряму передати керування refund_agent і навпаки —
усі міжрольові переходи проходять лише через triage. Це не політика
рівня промпту, а факт топології графа: прямого ребра між спеціалістами
не існує.

Guardrails:
    - Input:  guardrails.validate_input — довжина запиту, детекція
      prompt injection (застосовується один раз, на вході в triage).
    - Tool:   allowlist — структурний (кожен вузол-спеціаліст фізично
      бачить лише свій список tools); валідація аргументів — Pydantic-
      схеми в tools_legacy.py/hitl.py; rate limiting — RunGuard
      (safety.py), застосований у triage_node (max_steps/timeout —
      обмежує кількість "стрибків" triage<->спеціаліст на один запит).
    - Output: guardrails.apply_output_guardrails — маскування PII
      (телефони) і блокування витоку секретів у respond_node.

Human-in-the-loop:
    notify_accountant виконується лише після підтвердження оператора:
    граф зупиняється через interrupt() у refund_confirm_node (той самий
    патерн, що й у hitl.py: approve/reject/edit).
"""

import operator
import sqlite3
import uuid
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from agent import SupportResponse, llm, llm_structured
from hitl import RISKY_TOOLS, TOOLS_BY_NAME, _resolve_decision, notify_accountant
from knowledge import search_knowledge
from safety import RunGuard
from tools_legacy import get_order_ship_date, search_order, track_parcel, track_ukrposhta_parcel
from observability import traced_apply_output_guardrails, traced_validate_input
from trajectory_logger import TrajectoryLogger

MAX_HANDOFFS = 6  # запобіжник від зациклення triage <-> спеціаліст

# ── Allowlist tools по агентах (tool guardrail: структурний) ─────
ORDER_TOOLS = [search_order, get_order_ship_date, track_parcel, track_ukrposhta_parcel]
KNOWLEDGE_TOOLS = [search_knowledge]
# refund_agent НЕ отримує notify_accountant як звичайний tool ReAct-агента:
# виклик проходить вручну через interrupt() у refund_confirm_node, щоб
# ризикова дія не могла виконатись без підтвердження оператора.

ORDER_SYSTEM_PROMPT = (
    'Ти — агент служби підтримки, що відповідає ЛИШЕ за статус замовлень і '
    'відстеження посилок (Нова Пошта / Укрпошта). Використовуй tools, щоб '
    'знайти реальні дані. Ніколи не вигадуй статус, дату чи номер, якого '
    'немає в результаті виклику tool. Дай стислу відповідь по суті завдання.'
)
KNOWLEDGE_SYSTEM_PROMPT = (
    'Ти — агент служби підтримки, що відповідає ЛИШЕ на довідкові питання '
    '(правила доставки, оплати, повернення, гарантії тощо) за допомогою '
    'search_knowledge. Дай стислу відповідь на основі знайдених документів.'
)

order_agent = create_agent(llm, ORDER_TOOLS, system_prompt=ORDER_SYSTEM_PROMPT)
knowledge_agent = create_agent(llm, KNOWLEDGE_TOOLS, system_prompt=KNOWLEDGE_SYSTEM_PROMPT)


# ── Structured output: рішення triage ────────────────────────────
class TriageDecision(BaseModel):
    """Рішення супервізора: кому передати запит або чи можна відповісти клієнту одразу."""

    action: Literal['route_order', 'route_knowledge', 'route_refund', 'finish'] = Field(
        description=(
            "'route_order' — питання про статус замовлення чи відстеження посилки; "
            "'route_knowledge' — загальне довідкове питання (доставка/оплата/повернення/гарантія); "
            "'route_refund' — потрібно ініціювати повернення коштів клієнту (РИЗИКОВА дія); "
            "'finish' — можна відповісти клієнту напряму (привітання, дані вже зібрані, "
            "запит поза межами компетенції служби підтримки тощо)."
        )
    )
    instructions: Optional[str] = Field(
        default=None, description="Для route_*: що саме має зробити спеціаліст (одне самодостатнє завдання)."
    )
    response: Optional[str] = Field(default=None, description="Для action='finish': фінальна відповідь клієнту.")


llm_triage = llm.with_structured_output(TriageDecision, method='function_calling')

TRIAGE_PROMPT = (
    'Ти — центральний координатор (triage) служби підтримки інтернет-магазину. '
    'Ти НЕ виконуєш дії сам і не маєш доступу до жодних tools — лише аналізуєш '
    'запит клієнта і рішучо визначаєш, хто має його обробити:\n'
    '- order_agent — статус замовлення, дата відправки, відстеження посилки;\n'
    '- knowledge_agent — довідкові питання (правила, умови, причини затримок);\n'
    '- refund_agent — клієнт просить повернути кошти / оформити повернення грошей.\n'
    'Якщо запит поза межами підтримки магазину (привітання, стороння тема) або '
    'дані вже зібрані спеціалістами і достатні для відповіді — завершуй сам '
    "(action='finish') з відповіддю клієнту українською."
)


# ── State ─────────────────────────────────────────────────────────
class MASState(TypedDict):
    input: str
    history: Annotated[List[str], operator.add]  # "agent: результат" по кожному хендофу
    route: Optional[str]
    task_instructions: Optional[str]
    pending_tool_calls: list
    response: Optional[str]
    final_response: Optional[SupportResponse]
    step_count: int
    guard: RunGuard
    logger: TrajectoryLogger


def _format_history(history: List[str]) -> str:
    return '\n'.join(history) or '(ще немає результатів від спеціалістів)'


# ── Вузол triage ──────────────────────────────────────────────────
def triage_node(state: MASState) -> dict:
    guard: RunGuard = state['guard']
    traj_logger: TrajectoryLogger = state['logger']
    step = state.get('step_count', 0)

    # Input guardrail — лише на першому вході (сирий запит клієнта).
    if step == 0:
        problem = traced_validate_input(state['input'])
        if problem:
            traj_logger.log('input_guardrail_blocked', {'reason': problem})
            return {'response': f'⚠️ Запит відхилено guardrail-ом: {problem}', 'step_count': step + 1}

    # Rate limiting — обмежує кількість стрибків triage<->спеціаліст.
    stop_reason = guard.check_step_limit(step) or guard.check_timeout()
    if stop_reason:
        traj_logger.log('triage_stopped', {'reason': stop_reason, 'step': step})
        return {
            'response': (
                f'⚠️ {stop_reason} Ось часткова відповідь на основі вже зібраної '
                f'інформації:\n{_format_history(state["history"])}'
            ),
            'step_count': step + 1,
        }

    decision = llm_triage.invoke([
        SystemMessage(content=TRIAGE_PROMPT),
        HumanMessage(content=(
            f'Запит клієнта: {state["input"]}\n\n'
            f'Результати вже виконаної роботи спеціалістів:\n{_format_history(state["history"])}'
        )),
    ])

    traj_logger.log('triage', {
        'step': step + 1,
        'action': decision.action,
        'instructions': decision.instructions,
    })

    if decision.action == 'finish':
        return {'response': decision.response or 'Не вдалося сформувати відповідь.', 'step_count': step + 1}

    route = decision.action.removeprefix('route_')
    return {'route': route, 'task_instructions': decision.instructions, 'step_count': step + 1}


def after_triage(state: MASState) -> Literal['order_agent', 'knowledge_agent', 'refund_propose', 'respond']:
    if state.get('response'):
        return 'respond'
    return {'order': 'order_agent', 'knowledge': 'knowledge_agent', 'refund': 'refund_propose'}[state['route']]


# ── Вузол order_agent ─────────────────────────────────────────────
def order_agent_node(state: MASState) -> dict:
    traj_logger: TrajectoryLogger = state['logger']
    task = state['task_instructions'] or state['input']
    result = order_agent.invoke({'messages': [HumanMessage(content=task)]})
    text = result['messages'][-1].content
    traj_logger.log('order_agent', {'task': task, 'result_preview': str(text)[:300]})
    return {'history': [f'order_agent: {text}']}


# ── Вузол knowledge_agent ─────────────────────────────────────────
def knowledge_agent_node(state: MASState) -> dict:
    traj_logger: TrajectoryLogger = state['logger']
    task = state['task_instructions'] or state['input']
    result = knowledge_agent.invoke({'messages': [HumanMessage(content=task)]})
    text = result['messages'][-1].content
    traj_logger.log('knowledge_agent', {'task': task, 'result_preview': str(text)[:300]})
    return {'history': [f'knowledge_agent: {text}']}


# ── refund_agent: propose (LLM обирає параметри) + confirm (interrupt) ──
# Аналогічно hitl.py: propose винесено окремим вузлом, щоб при resume
# після interrupt() не викликати LLM повторно.
llm_with_notify = llm.bind_tools([notify_accountant])


def refund_propose_node(state: MASState) -> dict:
    traj_logger: TrajectoryLogger = state['logger']
    task = state['task_instructions'] or state['input']
    response = llm_with_notify.invoke(
        f'Завдання: {task}\n\n'
        f'Контекст (результати попередніх кроків):\n{_format_history(state["history"])}\n\n'
        'Виклич notify_accountant з правильними параметрами (номер замовлення, сума, причина).'
    )
    tool_calls = response.tool_calls or []
    traj_logger.log('refund_propose', {'tool_calls': [tc['name'] for tc in tool_calls]})
    return {'pending_tool_calls': tool_calls}


def refund_confirm_node(state: MASState) -> dict:
    traj_logger: TrajectoryLogger = state['logger']
    tool_calls = state['pending_tool_calls']

    if not tool_calls:
        traj_logger.log('refund_confirm', {'result': 'LLM не викликав notify_accountant'})
        return {'history': ['refund_agent: не вдалося визначити параметри повернення коштів.']}

    outputs = []
    for tc in tool_calls:
        name, args = tc['name'], tc['args']
        if name not in RISKY_TOOLS:
            outputs.append(f'{name}: {TOOLS_BY_NAME[name].invoke(args)}')
            continue

        decision = interrupt({
            'action': name,
            'args': args,
            'message': f'⚠️ Підтвердіть ризикову дію перед виконанням:\nTool: {name}\nПараметри: {args}',
        })
        kind, final_args = _resolve_decision(decision, args)
        traj_logger.log('refund_confirm', {'tool': name, 'args': args, 'decision': kind})

        if kind == 'reject':
            reason = decision.get('reason', 'без причини') if isinstance(decision, dict) else 'без причини'
            outputs.append(f'{name} — ВІДХИЛЕНО оператором ({reason})')
            continue

        result = TOOLS_BY_NAME[name].invoke(final_args)
        label = 'ЗМІНЕНО оператором і виконано' if kind == 'edit' else 'ПІДТВЕРДЖЕНО і виконано'
        outputs.append(f'{name} [{label}]: {result}')

    return {'history': [f'refund_agent: ' + '; '.join(outputs)], 'pending_tool_calls': []}


# ── Вузол respond (structured output + output guardrails) ───────
def respond_node(state: MASState) -> dict:
    traj_logger: TrajectoryLogger = state['logger']
    safe_response = traced_apply_output_guardrails(state['response'])
    structured = llm_structured.invoke([
        SystemMessage(content='Перетвори відповідь агента підтримки клієнтів у структурований формат.'),
        HumanMessage(content=safe_response),
    ])
    traj_logger.log('respond', {'final_text_preview': safe_response[:300], 'resolved': structured.resolved})
    return {'final_response': structured}


# ── Граф ─────────────────────────────────────────────────────────
graph = StateGraph(MASState)
graph.add_node('triage', triage_node)
graph.add_node('order_agent', order_agent_node)
graph.add_node('knowledge_agent', knowledge_agent_node)
graph.add_node('refund_propose', refund_propose_node)
graph.add_node('refund_confirm', refund_confirm_node)
graph.add_node('respond', respond_node)

graph.add_edge(START, 'triage')
graph.add_conditional_edges('triage', after_triage, {
    'order_agent': 'order_agent',
    'knowledge_agent': 'knowledge_agent',
    'refund_propose': 'refund_propose',
    'respond': 'respond',
})
# Handoff matrix: спеціалісти повертають керування ЛИШЕ triage.
graph.add_edge('order_agent', 'triage')
graph.add_edge('knowledge_agent', 'triage')
graph.add_edge('refund_propose', 'refund_confirm')
graph.add_edge('refund_confirm', 'triage')
graph.add_edge('respond', END)

# ── Checkpointer (ОБОВ'ЯЗКОВИЙ: refund_confirm використовує interrupt()) ──
CHECKPOINT_DB = Path(__file__).parent / 'mas_state.db'
_conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False)
_serde = JsonPlusSerializer(allowed_msgpack_modules=[
    ('safety', 'RunGuard'),
    ('safety', 'LoopDetector'),
    ('trajectory_logger', 'TrajectoryLogger'),
])
checkpointer = SqliteSaver(_conn, serde=_serde)

app = graph.compile(checkpointer=checkpointer)


# ── Демонстрація ──────────────────────────────────────────────────
def _initial_state(query: str) -> dict:
    return {
        'input': query,
        'history': [],
        'route': None,
        'task_instructions': None,
        'pending_tool_calls': [],
        'response': None,
        'final_response': None,
        'step_count': 0,
        'guard': RunGuard(max_steps=MAX_HANDOFFS),
        'logger': TrajectoryLogger(),
    }


def run_query(query: str, trajectory_path: str = 'trajectory_mas.json') -> dict:
    """Прогнати один запит клієнта через MAS; якщо граф зупинився на interrupt()
    (ризикова дія refund_agent) — одразу підтвердити (approve), як демонстрація."""
    thread_id = str(uuid.uuid4())
    config = {'configurable': {'thread_id': thread_id}}
    print(f'\n{"=" * 70}\nЗапит: {query}\n{"=" * 70}')

    result = app.invoke(_initial_state(query), config)

    while result.get('__interrupt__'):
        payload = result['__interrupt__'][0].value
        print(f"\n⏸  interrupt: {payload['message']}\n▶ Демо-рішення оператора: approve")
        result = app.invoke(Command(resume={'action': 'approve'}), config)

    result['logger'].save(trajectory_path)
    final: SupportResponse = result['final_response']
    print('\nStructured response:')
    print(final.model_dump_json(indent=2, exclude_none=True))
    return result


if __name__ == '__main__':
    run_query('Який статус мого замовлення У-0387776 і коли його відправили?', 'trajectory_mas_order.json')
    run_query('Чому затримується доставка за кордон?', 'trajectory_mas_knowledge.json')
    run_query(
        'Поверніть, будь ласка, 450 грн за замовлення У-0387776, товар прийшов пошкодженим.',
        'trajectory_mas_refund.json',
    )
