"""src.models.__init__

Пакет модей сущностей для хранения в БД.
"""

from src.models.state import StateModel
from src.models.user import UserModel

__all__ = ['StateModel', 'UserModel']
