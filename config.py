"""Единый конфигурационный файл"""
# Параметры для обработки данных
CHUNK_SIZE = 1024
CHUNK_OVERLAP = 150
EMBEDDING_BATCH_SIZE = 100  # Отправляем по 100 чанков за один API-запрос
FAISS_DIMENSION = 1536  # Размерность векторов для модели эмбеддингов
K_FINAL_CHUNKS = 7  # Количество наиболее релевантных чанков для поиска

# Асинхронная обработка вопросов
ASYNC_MODE = True  # Обработка вопросов в асинхронном (True) либо синхронном (False) режиме
MAX_WORKERS = 10  # Число параллельных обработчиков

# Настройки Rerank:
USE_RERANKER = True
RETRIEVAL_K_FOR_RERANK = (
    30  # Сколько чанков изначально достаем из FAISS для переранжирования
)

# Параметры для режима разработки
USE_LOCAL_RAG_FILES = (
    False  # Использовать RAG-артефакты, которые уже сохранены в директории
)
SAVE_RAG_FILES = False
LOGGING_TOKEN_USAGE = False  # Логгировать использование токенов
LOGGING_TIME_USAGE = False  # Логгировать использование времени
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNKS_PATH = "corpus_chunks.pkl"
