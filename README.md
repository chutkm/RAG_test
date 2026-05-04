# RAG Query Script (Single Document QA)

## 📌 Описание

Проект реализует простой **RAG (Retrieval-Augmented Generation)** пайплайн для ответа на вопросы по одному документу.

Скрипт:

* принимает документ (TXT / PDF / DOCX)
* извлекает релевантные фрагменты
* формирует ответ **только на основе найденного контекста**

---

## ⚙️ Архитектура решения

Pipeline состоит из следующих этапов:

1. **Загрузка документа**

   * Поддержка: `.txt`, `.pdf`, `.docx`

2. **Chunking**

   * Разбиение текста на перекрывающиеся фрагменты

3. **Embedding**

   * Модель: `intfloat/multilingual-e5-large`
   * Используется через Hugging Face API

4. **Retrieval (FAISS)**

   * Индекс: `IndexFlatIP`
   * Метрика: cosine similarity (через нормализацию)
   * Фильтрация по threshold

5. **Reranking**

   * reranker через embedding similarity

6. **LLM генерация**

   * Модель: `deepseek-ai/DeepSeek-V4-Flash`
   * Ответ формируется строго по контексту

---

## 📁 Структура проекта

```
RAG_test/
│
├── rag_query.py
├── config.py
├── requirements.txt
├── .env
├── test.txt
└── README.md
```

---

## 🔧 Конфигурация

### `config.py`

```python
EMBEDDING_CONFIG = {
    "model": "intfloat/multilingual-e5-large",
    "provider": "hf-inference",
    "batch_size": 16,
}

RETRIEVER_CONFIG = {
    "top_k": 10,
    "score_threshold": 0.5,
}

RERANKER_CONFIG = {
    "enabled": True,
    "model": "BAAI/bge-reranker-large",
    "provider": "hf-inference",
    "top_k": 5,
}

LLM_CONFIG = {
    "model": "deepseek-ai/DeepSeek-V4-Flash",
    "provider": "novita",
    "max_tokens": 512,
}
```

---

## 🔑 Настройка API

Создать файл `.env`:

```
HF_TOKEN=your_huggingface_token
```

---

## ▶️ Запуск

```bash
python rag_query.py --document <путь_к_файлу> --question "<вопрос>"
```

---

## 🧪 Быстрый тест (рекомендуется)

В проекте уже есть файл `test.txt`, который можно использовать для проверки работы.

### Пример запуска:

```bash
python rag_query.py --document test.txt --question "О чем этот документ?"
```

### Дополнительные тесты:

```bash
python rag_query.py --document test.txt --question "Какие функции у NovaAI?"
python rag_query.py --document test.txt --question "Какие ключевые факты упоминаются?"
```

👉 Эти команды позволяют быстро проверить:

* что retrieval работает
* что модель отвечает по контексту
* что выводятся источники

---

## 💡 Примеры

### Пример 1

```bash
python rag_query.py --document document.txt --question "Какая цель программы?"
```

---

### Пример 2

```bash
python rag_query.py --document document.pdf --question "Сколько длится программа?"
```

---

### Пример 3 (нет ответа)

```bash
python rag_query.py --document document.txt --question "Кто автор Марса?"
```

```
Ответ не найден в документе
```

---

## 📦 Зависимости

Установить:

```bash
pip install -r requirements.txt
```

---

## ⚠️ Ограничения

* Используется HF API → возможны задержки и timeout

---

## ✅ Итог

Реализован RAG pipeline:

* retrieval через FAISS
* embeddings через HF API
* генерация через LLM
* поддержка TXT / PDF / DOCX
* вывод источников ответа
* наличие тестового файла для быстрой проверки

---
