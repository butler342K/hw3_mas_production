# ── Крок 1: Архітектура Customer Support MAS ─────────────────────
#
# Загальна ідея:
#   Побудувати мультиагентну систему підтримки клієнтів із центральним
#   координатором (supervisor), який аналізує запит і вирішує,
#   якому спеціалісту передати обробку.
#
# Патерн координації:
#   Supervisor / Router pattern
#
# Агенти:
#   1) triage_agent
#      - Центральний координатор системи
#      - Аналізує запит користувача
#      - Визначає тип звернення: billing / technical / general
#      - Не має доступу до ризикових інструментів
#      - Може передавати керування billing_agent або tech_agent
#
#   2) billing_agent
#      - Обробляє запити, пов’язані з оплатою, поверненням коштів,
#        статусом транзакцій, рахунками
#      - Має доступ до check_order_status, check_payment, process_refund
#      - Повертає керування triage_agent, якщо запит не належить до billing
#
#   3) tech_agent
#      - Обробляє технічні проблеми
#      - Має доступ до check_order_status та search_faq
#      - Повертає керування triage_agent, якщо запит не є технічним
#
# MCP Tools:
#   - check_order_status(order_id: str) -> str
#   - check_payment(payment_id: str) -> str
#   - process_refund(order_id: str, reason: str) -> str   # ризиковий tool
#   - search_faq(query: str) -> str
#
# Resource:
#   - support://info
#
# Handoff matrix:
#   triage  -> billing, tech
#   billing -> triage
#   tech    -> triage
#
# Важливе обмеження безпеки:
#   billing НЕ МОЖЕ напряму передавати керування tech
#   tech НЕ МОЖЕ напряму передавати керування billing
#   Усі міжрольові переходи проходять через triage
#
# Human-in-the-loop:
#   process_refund потребує підтвердження людиною
#
# Guardrails:
#   Input guardrails:
#       - виявлення prompt injection
#       - обмеження довжини запиту
#
#   Tool guardrails:
#       - allowlist інструментів для кожного агента
#       - перевірка аргументів
#       - rate limiting
#
#   Output guardrails:
#       - маскування PII
#       - блокування небажаного виводу

# ── Крок 2: Адаптація структури до проєкту shop_support ──────────
#
# Той самий Supervisor/Router патерн з Кроку 1, спроєктований під
# реальний домен і tools цього проєкту (замість billing/tech-заглушок).
# Це САМЕ СТРУКТУРА/дизайн — граф і код агентів пишемо наступним кроком.
#
# Стан реалізації на момент цього кроку:
#   - mcp_server.py         — реалізовано (MCP-обгортка над tools)
#   - tools_legacy.py       — перевикористано без змін (з hw1)
#   - knowledge.py          — перевикористано без змін (RAG, з hw2)
#   - hitl.py               — перевикористано без змін (HITL-патерн
#                              interrupt()/approve/reject/edit, з hw2)
#   - safety.py             — перевикористано без змін (RunGuard, з hw1)
#   - trajectory_logger.py  — перевикористано без змін (з hw1)
#   - агенти/граф (triage/order/knowledge/refund) — ще не реалізовано,
#     будуємо наступним кроком за цією структурою
#
# Агенти:
#   1) triage_agent
#      - Центральний координатор системи
#      - Аналізує запит клієнта
#      - Визначає тип звернення: order / knowledge / refund
#      - Не має доступу до жодних tools
#      - Може передавати керування order_agent, knowledge_agent або refund_agent
#
#   2) order_agent
#      - Обробляє запити про статус замовлення, дату відправки й
#        відстеження посилки (Нова Пошта / Укрпошта)
#      - Має доступ до search_order, get_order_ship_date, track_parcel,
#        track_ukrposhta_parcel
#      - Повертає керування triage_agent, якщо запит не про замовлення/посилку
#
#   3) knowledge_agent
#      - Обробляє довідкові запитання (доставка, оплата, повернення,
#        гарантія, митні процедури)
#      - Має доступ до search_knowledge (RAG над базою знань)
#      - Повертає керування triage_agent, якщо запит не довідковий
#
#   4) refund_agent
#      - Обробляє запити на повернення коштів клієнту
#      - Має доступ до notify_accountant — РИЗИКОВИЙ tool (рухає гроші
#        компанії)
#      - Повертає керування triage_agent після виконання чи відхилення дії
#
# MCP Tools (вже реалізовано в mcp_server.py, сервер "shop_support"):
#   - search_order(query: str) -> str
#   - get_order_ship_date(query: str) -> str
#   - track_parcel(ttn: str) -> str
#   - track_ukrposhta_parcel(barcode: str) -> str
#   - search_knowledge(query: str) -> str
#   notify_accountant свідомо НЕ винесено в MCP: ризикова дія має
#   виконуватись лише всередині графа під контролем interrupt(), а не як
#   звичайний виклик tool ззовні.
#
# Resource:
#   - support://info
#
# Handoff matrix:
#   triage  -> order, knowledge, refund
#   order, knowledge, refund -> triage
#
# Важливе обмеження безпеки:
#   order_agent НЕ МОЖЕ напряму передавати керування refund_agent
#   refund_agent НЕ МОЖЕ напряму передавати керування order_agent чи knowledge_agent
#   Усі міжрольові переходи проходять лише через triage
#
# Human-in-the-loop:
#   notify_accountant потребує підтвердження людиною (approve/reject/edit)
#   перед виконанням — той самий патерн interrupt(), що вже є в hitl.py
#
# Guardrails:
#   Input guardrails:
#       - виявлення prompt injection
#       - обмеження довжини запиту
#
#   Tool guardrails:
#       - allowlist інструментів для кожного агента (order_agent і
#         knowledge_agent фізично не отримують доступу до чужих tools)
#       - перевірка аргументів — вже є, Pydantic-схеми в tools_legacy.py/hitl.py
#       - rate limiting — вже є, RunGuard з safety.py (max_steps/timeout),
#         застосувати до кількості хендофів triage<->спеціаліст
#
#   Output guardrails:
#       - маскування PII (телефони клієнтів)
#       - блокування небажаного виводу (витік секретів/ключів)
