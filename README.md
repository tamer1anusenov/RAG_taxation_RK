# RAG Taxation RK

Обработка русского текста Налогового кодекса Республики Казахстан для RAG.

## Структура проекта

```text
.
├── data/
│   ├── raw/
│   │   └── taxiation_doc.txt                 # исходный извлечённый текст
│   ├── parsed/
│   │   ├── taxiation_chunks.jsonl            # иерархические статьи/пункты/подпункты
│   │   └── taxiation_table_candidates.json   # кандидаты табличных фрагментов
│   └── chunks/
│       ├── taxiation_parents.jsonl           # целые статьи для контекста генерации
│       ├── taxiation_children.jsonl          # поисковые child-чанки
│       └── taxiation_parent_child_manifest.json
│   └── indexes/
│       ├── bm25_index.pkl                      # keyword/BM25-индекс
│       └── index_manifest.json
├── src/
│   ├── parse_tax_code.py                     # парсинг структуры документа
│   ├── build_parent_child.py                 # построение parent-child чанков
│   ├── index_chunks.py                       # embedding и индексация
│   ├── hybrid_search.py                      # dense + BM25 + RRF + reranking
│   ├── api.py                                # FastAPI API
│   └── telegram_bot.py                       # Telegram long-polling adapter
├── frontend/
│   └── index.html                            # минимальный интерфейс
├── eval/
│   ├── chunks.json                             # слепок chunks из Qdrant для evaluation
│   ├── golden.json                             # golden questions с relevant_ids
│   └── metrics.json                            # метрики retrieval evaluation
├── config/
│   ├── system_prompt.txt                    # строгие правила ответа
│   ├── response_schema.json                  # JSON Schema ответа
│   └── llm_config.json                       # temperature и параметры генерации
├── requirements.txt                          # Python-зависимости RAG-хранилища
├── README.md
└── .gitignore
```

## Шаг 4. Хранилище

Используются два поисковых слоя:

- Qdrant — векторная БД для семантического поиска;
- `rank-bm25` — keyword/BM25-поиск по номерам статей, терминам и точным фразам.

`qdrant-client` подключает Python-код к Qdrant, а `fastembed` генерирует эмбеддинги.

В production Qdrant должен быть внешним сервисом. Укажите его URL через
`QDRANT_URL`; локальное хранилище `QDRANT_LOCAL_PATH` предназначено только для
разработки и не используется на Vercel. Локально при наличии каталога
`qdrant_storage/` он выбирается автоматически, поэтому отдельная переменная
`QDRANT_LOCAL_PATH` не обязательна.

## Шаг 5. Эмбеддинги и индексация

Индексатор использует `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` — компактную multilingual-модель с поддержкой русского текста. Для этой модели специальные префиксы не требуются; при смене на E5-модель можно задать `EMBEDDING_PREFIX="passage: "`, а для запросов использовать `query: `.

Полная индексация child-чанков в Qdrant и BM25:

```bash
python3 -m pip install -r requirements.txt
python3 src/index_chunks.py
```

Переменные окружения:

- `QDRANT_URL` — URL Qdrant, по умолчанию `http://localhost:6333`;
- `QDRANT_COLLECTION` — имя коллекции, по умолчанию `taxation_children`;
- `EMBEDDING_MODEL` — embedding-модель, по умолчанию `intfloat/multilingual-e5-small`;
- `EMBEDDING_BATCH_SIZE` — размер batch, по умолчанию `32`;
- `QDRANT_LOCAL_PATH` — локальный режим Qdrant без сервера, например `qdrant_storage`.

Для отдельной проверки BM25 без Qdrant:

```bash
python3 src/index_chunks.py --bm25-only
```

Результаты сохраняются в `data/indexes/`:

- `bm25_index.pkl` — индекс точного keyword/BM25-поиска;
- `index_manifest.json` — параметры и статистика индексации.

## Шаг 6. Гибридный поиск

Гибридный поиск объединяет dense-результаты Qdrant и sparse/BM25-результаты через Reciprocal Rank Fusion (RRF). После этого только top-10 кандидатов проходят через лёгкий cross-encoder reranker `Xenova/ms-marco-MiniLM-L-6-v2`.

```bash
QDRANT_LOCAL_PATH=qdrant_storage python3 src/hybrid_search.py "налоговая ставка для малого бизнеса"
```

Для Docker-Qdrant:

```bash
python3 src/hybrid_search.py "налоговая ставка для малого бизнеса"
```

Настройки ограничения нагрузки:

- `--dense-limit 30` — максимум dense-кандидатов;
- `--bm25-limit 30` — максимум BM25-кандидатов;
- `--rerank-limit 10` — сколько RRF-кандидатов rerank-ить;
- `--output-limit 5` — сколько результатов вернуть.

Cross-encoder можно заменить через `RERANKER_MODEL`, но для русского языка более крупные модели потребуют больше памяти.

## Шаг 7. Строгий ответ и JSON-формат

Конфигурация ответа находится в `config/`:

- `system_prompt.txt` — отвечает только на основании `FOUND_CONTEXT`, требует точные номера статей/пунктов и дословные цитаты;
- `response_schema.json` — запрещает лишние поля и задаёт поля `answer`, `citations`, `sources`, `confidence`;
- `llm_config.json` — параметры совместимости генерации и честный отказ при отсутствии контекста; temperature не отправляется, потому что текущая модель поддерживает только значение по умолчанию.

При отсутствии достаточного контекста модель должна вернуть:

```json
{
  "answer": "В найденном контексте нет достаточной информации для ответа на этот вопрос.",
  "citations": [],
  "sources": [],
  "confidence": 0.0
}
```

В обычном ответе каждая существенная часть `answer` должна иметь элемент в `citations` с точными `article`, `point`, `subpoint`, `quote` и `source_id`. Поле `quote` должно быть дословной цитатой из найденного контекста.

## Шаг 8. API и минимальный frontend

FastAPI оборачивает retrieval и генерацию в эндпоинт `POST /api/ask`. Каждый запрос логируется в JSONL: вопрос, найденные source ID и оценки, финальный ответ и время по стадиям. Логи ограничены `5 MB` на файл и двумя backup-файлами через `RotatingFileHandler`; локально они сохраняются в `logs/api.jsonl`, а на Vercel — во временном `/tmp/api.jsonl`.

Запуск API локально из корня проекта:

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Для Vercel функция находится в `api/index.py`, а готовый frontend — в
`frontend/`. Файловая система serverless-функции read-only, поэтому логи при
развёртывании пишутся в `/tmp`. Qdrant должен быть доступен по внешнему
`QDRANT_URL`; локальный `qdrant_storage` в репозиторий не добавляется.

Открыть локальный frontend: `http://localhost:8000/`.

## Telegram-бот

Telegram-бот использует long polling и отправляет вопросы в локальный FastAPI.
Токен хранится только в `.env` и не должен добавляться в Git:

```dotenv
TELEGRAM_BOT_TOKEN=your_bot_token
RAG_API_URL=http://127.0.0.1:8000
```

В одном терминале запустите API, во втором — бота:

```bash
QDRANT_LOCAL_PATH=qdrant_storage uvicorn src.api:app --host 127.0.0.1 --port 8000
python3 -m src.telegram_bot
```

Команды `/start` и `/help` показывают подсказку; любое другое текстовое
сообщение обрабатывается как вопрос к RAG. Во время обработки бот сначала
показывает статус ожидания, а в итоговом сообщении выводит время dense-поиска,
BM25, RRF, reranking, генерации и общее время.

## Слепок чанков для evaluation

Экспорт всех точек из Qdrant через paginated `scroll` в формате `{"id": "...", "text": "..."}`:

```bash
python3 -m src.export_qdrant_chunks
```

Результат сохраняется в `eval/chunks.json`. Размер страницы можно изменить через
`--limit`; локальный `qdrant_storage/` определяется автоматически, а для
облачного Qdrant используются `QDRANT_URL` и `QDRANT_COLLECTION`.

Для генерации небольшого стабильного golden sample используются 50 случайных
chunks и фиксированный seed `42`. Для каждого chunk выполняется один LLM-запрос,
который генерирует два живых пользовательских вопроса, полностью покрываемых
этим chunk. Модель по умолчанию — `gpt-4o-mini`:

```bash
python3 -m src.generate_golden
```

Команда делает 50 LLM-запросов и сохраняет 100 записей в `eval/golden.json`.
Каждая запись имеет формат `{"question": "...", "relevant_ids": ["id_чанка"]}`;
оба вопроса одного chunk получают его Qdrant ID из `eval/chunks.json`.
Можно изменить параметры через `--seed`, `--sample-size`, `--model` и `--output`.

Для оценки dense retrieval по golden-вопросам:

```bash
python3 -m src.evaluate_retrieval
```

Скрипт выполняет поиск top-5 в той же Qdrant-коллекции и сохраняет в
`eval/metrics.json` средние `precision_at_k`, `recall_at_k`, `hit_rate_at_k` и
`mrr`, а также результаты по каждому вопросу. По умолчанию `k=5`; ID сравниваются
напрямую с Qdrant point IDs из `relevant_ids`.

Пример API-запроса:

```bash
curl -X POST http://localhost:8000/api/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Какая налоговая ставка применяется?"}'
```

Для генерации ответа API ожидает `OPENAI_API_KEY`; совместимый endpoint можно задать через `OPENAI_BASE_URL`, модель — через `LLM_MODEL`. Эти параметры можно хранить в локальном `.env` (он добавлен в `.gitignore`):

```dotenv
OPENAI_API_KEY=your_alem_api_key
OPENAI_BASE_URL=https://llm.alem.ai/v1
LLM_MODEL=kazllm
LLM_RESPONSE_FORMAT=none
```

`none` выбран для совместимости с Alem: строгий JSON обеспечивается system prompt и дополнительно проверяется API через `normalize_answer()`. Если конкретный endpoint поддерживает JSON-формат, можно заменить значение на `json_object` или `json_schema`.

До генерации выполняются dense + BM25 + RRF + reranking, а parent-статья добавляется в `FOUND_CONTEXT`.

## Запуск обработки данных

Из корня проекта:

```bash
python3 src/parse_tax_code.py
python3 src/build_parent_child.py
```

Индексация выполнялась локально через embedded Qdrant:

bash
QDRANT_LOCAL_PATH=qdrant_storage python3 src/index_chunks.py


Для будущего запуска через Docker Qdrant:

bash
python3 src/index_chunks.py


При этом должен быть доступен Qdrant на:

text
http://localhost:6333


Использованная модель:

sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

Причина выбора:

- поддерживается установленной версией fastembed;
- компактная модель;
- поддерживает русский язык;
- модель успешно загрузилась и отработала на текущей машине.


Скрипты используют пути относительно корня проекта, поэтому их можно запускать из любой рабочей директории.

## Стратегия чанкинга

- child-чанк: пункт статьи;
- подпункты включаются в соответствующий пункт;
- статья без пунктов становится одним child-чанком;
- parent-чанк: полная статья целиком;
- связь между child и parent хранится в `metadata.parent_id`.
