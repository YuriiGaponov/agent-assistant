"""src.RAG.__init__

Пакет RAG (Retrieval-Augmented Generation) для Agent‑Assistant.
Обеспечивает обработку файлов внутренней документации:
добавление, удаление, чтение и разделение на чанки.

Состав:
- chunking_service: разбивка текста на смысловые чанки
- doc_dir_service: модуль логики взаимодействия с директорией внутренней
  документации
- doc_file_service: чтение файлов, метаданных
- doc_read_service: исервисов чтения внутренней документации,
  создание структуры данных прочитанных документов
- embedding_service: сервис создания эмбеддингов
- schemas: схемы данных, используемые при обработке документов
- init_vector_collection: инициализация векторной коллекции при старте
"""

from src.RAG.embedding_service import embedding_model
from src.RAG.init_vector_collection import init_vector_collection
from src.RAG.schemas import Document
from src.RAG.search_service import semantic_search
from src.RAG.doc_dir_service import doc_dir_exist


__all__ = ['embedding_model', 'init_vector_collection', 'Document',
           'semantic_search', 'doc_dir_exist']
