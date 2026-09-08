"""Customer Support MAS: той самий кейс (див. mas_langgraph.py, MAS_structure.md
Крок 2), реалізований у CrewAI — hierarchical crew з manager_agent, для
порівняння двох моделей оркестрації (граф з явними ребрами vs делегування
менеджера).

Агенти (той самий домен, ті самі tools, що й у mas_langgraph.py):
    triage_manager (manager_agent, без tools)
        -> order_agent     (search_order, get_order_ship_date,
                             track_parcel, track_ukrposhta_parcel)
        -> knowledge_agent (search_knowledge — RAG по базі знань)
        -> refund_agent    (propose_refund — БЕЗПЕЧНИЙ tool; реальний
                             notify_accountant агенту не видається, див. нижче)

Ключові структурні відмінності від LangGraph-версії (mas_langgraph.py):

    1. Handoff matrix. У LangGraph межі дозволених переходів — це ребра
       графа: order_agent фізично не може передати керування refund_agent,
       бо такого ребра просто не існує. У CrewAI hierarchical crew такого
       топологічного гаранта немає: маршрутизація — це рішення LLM-менеджера
       (triage_manager), кероване промптом і `allow_delegation`. Тут усі три
       спеціалісти мають allow_delegation=False (не можуть делегувати нікому),
       а delegation-інструмент отримує лише manager_agent — тому спеціаліст
       не може напряму передати керування іншому спеціалісту, але сам факт
       "маршрут A, а не B" — це вибір менеджера, а не структурна неможливість.

    2. Rate limiting. RunGuard (safety.py) рахує спільний бюджет кроків для
       ВСЬОГО ланцюжка triage<->спеціаліст. У CrewAI немає єдиного лічильника
       на весь crew — найближчий аналог, `max_iter`/`max_rpm`, працює
       ПООДИНОКО на кожного агента. Тут узято ті самі числа (safety.MAX_STEPS/
       TIMEOUT_SECONDS) для порівнюваності, але це м'якший, локальний ліміт.

    3. Human-in-the-loop. LangGraph зупиняє ВИКОНАННЯ ВСЕРЕДИНІ вузла через
       interrupt() і продовжує через Command(resume=...) — потенційно з
       іншого процесу, за thread_id. У CrewAI найближчий нативний механізм,
       Task(human_input=True), блокує crew.kickoff() на реальному input() у
       терміналі й не має аналога resume-за-ідентифікатором — тому демо-запуск
       не можна прогнати неінтерактивно. Замість цього ризикову дію винесено
       за межі допуску refund_agent: агент отримує лише propose_refund
       (БЕЗПЕЧНИЙ tool, що готує, але не виконує повернення коштів), а реальний
       notify_accountant виконується виключно через confirm_and_notify() —
       окрему функцію, яку викликає оператор ПІСЛЯ crew.kickoff(), той самий
       контракт approve/reject/edit, що й у hitl.py/mas_langgraph.py
       (_resolve_decision перевикористано звідти напряму).

Guardrails (ті самі механізми, що й у mas_langgraph.py):
    - Input:  guardrails.validate_input — застосовується в run_query() перед
      crew.kickoff() (тут немає вузла triage, який робив би це сам).
    - Tool:   allowlist — структурний, через `tools=` кожного Agent;
      валідація аргументів — ті самі Pydantic-схеми з tools_legacy.py/hitl.py
      (tools перевикористані без змін, CrewAI підтримує LangChain tools);
      rate limiting — max_iter/max_rpm на кожного агента (див. пункт 2 вище).
    - Output: guardrails.apply_output_guardrails — застосовується в
      run_query() до фінальної відповіді crew.
"""

import os

from crewai import Agent, Crew, Process, Task
from crewai.tools.base_tool import Tool as CrewTool
from langchain_core.tools import tool

from agent import SupportResponse
from guardrails import apply_output_guardrails, validate_input
from hitl import NotifyAccountantInput, _resolve_decision, notify_accountant
from knowledge import search_knowledge
from safety import MAX_STEPS, TIMEOUT_SECONDS
from tools_legacy import get_order_ship_date, search_order, track_parcel, track_ukrposhta_parcel
from trajectory_logger import TrajectoryLogger

LLM_MODEL = os.getenv('LLM_MODEL', 'gpt-4.1')


def _as_crew_tools(*langchain_tools) -> list[CrewTool]:
    """CrewAI (>=1.x) не приймає LangChain StructuredTool напряму в Agent(tools=...) —
    конвертує в crewai.tools.Tool через from_langchain, зберігаючи Pydantic
    args_schema (ту саму валідацію аргументів з tools_legacy.py/hitl.py)."""
    return [CrewTool.from_langchain(t) for t in langchain_tools]


# ── Allowlist tools по агентах (tool guardrail: структурний) ─────
ORDER_TOOLS = _as_crew_tools(search_order, get_order_ship_date, track_parcel, track_ukrposhta_parcel)
KNOWLEDGE_TOOLS = _as_crew_tools(search_knowledge)


# ── Безпечний tool для refund_agent: лише готує, не виконує ─────
# На відміну від mas_langgraph.py (де сам виклик notify_accountant
# гейтиться через interrupt() усередині графа), тут гейт — структурний:
# refund_agent фізично не має доступу до notify_accountant, лише до
# цього безпечного "чернетка" tool-а. Реальний виклик — confirm_and_notify().
@tool(args_schema=NotifyAccountantInput)
def propose_refund(order_number: str, amount: float, reason: str) -> str:
    """Підготувати пропозицію повернення коштів для перевірки оператором.

    РИЗИКОВУ дію (реальне надсилання повідомлення бухгалтеру) цей tool
    НЕ виконує — лише формує пропозицію з параметрами. Виконання —
    через confirm_and_notify(), після підтвердження оператором.

    Args:
        order_number: Номер замовлення (наприклад, "У-0387776").
        amount: Сума до повернення клієнту, грн.
        reason: Причина повернення коштів.

    Returns:
        Текстовий опис пропозиції (не підтвердженої і не виконаної).
    """
    return (
        f'Пропозиція повернення коштів (потребує підтвердження оператора): '
        f'{amount:.2f} грн за замовленням {order_number}. Причина: {reason}'
    )


# ── Агенти ────────────────────────────────────────────────────────
triage_manager = Agent(
    role='Triage Manager служби підтримки інтернет-магазину',
    goal=(
        'Проаналізувати запит клієнта, визначити його тип і делегувати '
        'правильному спеціалісту, а тоді сформувати фінальну відповідь '
        'клієнту українською мовою.'
    ),
    backstory=(
        'Ти координатор служби підтримки. Сам не виконуєш доменну роботу і '
        'не маєш жодних tools — лише вирішуєш, кому передати запит:\n'
        '- order_agent — статус замовлення, дата відправки, відстеження посилки '
        '(Нова Пошта / Укрпошта);\n'
        '- knowledge_agent — довідкові питання (правила доставки, оплати, '
        'повернення, гарантії, причини затримок);\n'
        '- refund_agent — клієнт просить повернути кошти / оформити повернення '
        'грошей (refund_agent лише готує пропозицію — фінальне виконання поза '
        'межами цього crew, потребує підтвердження оператора).\n'
        'Якщо запит поза межами підтримки магазину — відповідай сам, ввічливо '
        'поясни, що це поза компетенцією служби підтримки.'
    ),
    llm=LLM_MODEL,
    allow_delegation=True,
    max_iter=MAX_STEPS,
    max_execution_time=TIMEOUT_SECONDS,
    verbose=True,
)

order_agent = Agent(
    role='Order & Delivery Specialist',
    goal='Дізнатись статус замовлення, дату відправки чи стан посилки клієнта.',
    backstory=(
        'Ти агент служби підтримки, що відповідає ЛИШЕ за статус замовлень і '
        'відстеження посилок. Використовуй tools, щоб знайти реальні дані. '
        'Ніколи не вигадуй статус, дату чи номер, якого немає в результаті '
        'виклику tool.'
    ),
    tools=ORDER_TOOLS,
    llm=LLM_MODEL,
    allow_delegation=False,
    max_iter=MAX_STEPS,
    max_execution_time=TIMEOUT_SECONDS,
    verbose=True,
)

knowledge_agent = Agent(
    role='Knowledge Base Specialist',
    goal='Відповісти на довідкові питання клієнта про правила магазину.',
    backstory=(
        'Ти агент служби підтримки, що відповідає ЛИШЕ на довідкові питання '
        '(правила доставки, оплати, повернення, гарантії тощо) за допомогою '
        'search_knowledge. Дай стислу відповідь на основі знайдених документів.'
    ),
    tools=KNOWLEDGE_TOOLS,
    llm=LLM_MODEL,
    allow_delegation=False,
    max_iter=MAX_STEPS,
    max_execution_time=TIMEOUT_SECONDS,
    verbose=True,
)

refund_agent = Agent(
    role='Refund Specialist',
    goal='Підготувати коректну пропозицію повернення коштів клієнту.',
    backstory=(
        'Ти агент служби підтримки, що обробляє запити на повернення коштів. '
        'Ти НЕ можеш самостійно виконати повернення — лише підготувати '
        'пропозицію (propose_refund) з номером замовлення, сумою і причиною. '
        'Реальне виконання завжди підтверджує оператор окремо.'
    ),
    tools=_as_crew_tools(propose_refund),
    llm=LLM_MODEL,
    allow_delegation=False,
    max_iter=MAX_STEPS,
    max_execution_time=TIMEOUT_SECONDS,
    verbose=True,
)


# ── Крок confirm: реальний notify_accountant — ПОЗА межами crew ──
def confirm_and_notify(order_number: str, amount: float, reason: str, decision: dict) -> str:
    """Підтвердити (approve/reject/edit) і за потреби виконати notify_accountant.

    Аналог refund_confirm_node (mas_langgraph.py) / confirm_and_execute
    (hitl.py): той самий контракт рішення оператора (_resolve_decision
    перевикористано напряму звідти), але викликається окремо від crew,
    бо CrewAI не має нативного interrupt()/Command(resume=...) для паузи
    ВСЕРЕДИНІ виконання агента.
    """
    original_args = {'order_number': order_number, 'amount': amount, 'reason': reason}
    kind, final_args = _resolve_decision(decision, original_args)

    if kind == 'reject':
        reason_txt = decision.get('reason', 'без причини') if isinstance(decision, dict) else 'без причини'
        return f'notify_accountant — ВІДХИЛЕНО оператором ({reason_txt})'

    result = notify_accountant.invoke(final_args)
    label = 'ЗМІНЕНО оператором і виконано' if kind == 'edit' else 'ПІДТВЕРДЖЕНО і виконано'
    return f'notify_accountant [{label}]: {result}'


# ── Crew (будується заново на кожен запит, щоб callback бачив свій logger) ──
def _build_crew(traj_logger: TrajectoryLogger) -> Crew:
    support_task = Task(
        description=(
            'Оброби звернення клієнта служби підтримки: {query}\n'
            '1. Визнач тип запиту: order / knowledge / refund.\n'
            '2. Делегуй роботу правильному спеціалісту (order_agent, '
            'knowledge_agent або refund_agent).\n'
            '3. Поверни клієнту фінальну відповідь українською мовою на основі '
            'результату спеціаліста. Для refund — чітко зазнач, що це '
            'пропозиція, яка потребує підтвердження оператора.'
        ),
        expected_output=(
            'Коротка, зрозуміла відповідь клієнту українською мовою на основі '
            'реальних даних від спеціаліста (без вигаданих фактів).'
        ),
        # agent НЕ задаємо: якщо прив'язати задачу до конкретного агента,
        # CrewAI (crew.py:_update_manager_tools) звузить список коворкерів
        # для delegate_work_to_coworker лише до цього агента — менеджер
        # взагалі не побачить order_agent/knowledge_agent/refund_agent.
        # Без agent крew сам призначає виконавцем manager_agent і будує
        # delegation tool з повним self.agents.
        output_pydantic=SupportResponse,
        callback=lambda output: traj_logger.log('task_completed', {'preview': str(output)[:300]}),
    )
    return Crew(
        agents=[order_agent, knowledge_agent, refund_agent],
        tasks=[support_task],
        process=Process.hierarchical,
        manager_agent=triage_manager,
        verbose=True,
    )


# ── Демонстрація ──────────────────────────────────────────────────
def run_query(query: str, trajectory_path: str = 'trajectory_mas_crewai.json') -> None:
    traj_logger = TrajectoryLogger()
    print(f'\n{"=" * 70}\nЗапит: {query}\n{"=" * 70}')

    problem = validate_input(query)
    if problem:
        traj_logger.log('input_guardrail_blocked', {'reason': problem})
        print(f'⚠️ Запит відхилено guardrail-ом: {problem}')
        traj_logger.save(trajectory_path)
        return

    crew = _build_crew(traj_logger)
    result = crew.kickoff(inputs={'query': query})

    structured: SupportResponse | None = getattr(result, 'pydantic', None)
    raw_answer = structured.answer if structured else str(result)
    safe_answer = apply_output_guardrails(raw_answer)
    if structured:
        structured.answer = safe_answer

    traj_logger.log('respond', {'final_text_preview': safe_answer[:300]})
    traj_logger.save(trajectory_path)

    print('\nStructured response:')
    if structured:
        print(structured.model_dump_json(indent=2, exclude_none=True))
    else:
        print(safe_answer)


if __name__ == '__main__':
    run_query('Який статус мого замовлення У-0387776 і коли його відправили?', 'trajectory_mas_crewai_order.json')
    run_query('Чому затримується доставка за кордон?', 'trajectory_mas_crewai_knowledge.json')
    run_query(
        'Поверніть, будь ласка, 450 грн за замовлення У-0387776, товар прийшов пошкодженим.',
        'trajectory_mas_crewai_refund.json',
    )

    # HITL: підтвердження ризикової дії оператором ПІСЛЯ crew.kickoff()
    # (crew.kickoff() вище лише підготував пропозицію через propose_refund,
    # гроші ще не рухались) — демо-рішення: approve, як у mas_langgraph.py.
    print(f'\n{"=" * 70}\nHITL: підтвердження notify_accountant оператором\n{"=" * 70}')
    print(confirm_and_notify('У-0387776', 450.0, 'товар пошкоджений', {'action': 'approve'}))
