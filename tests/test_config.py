"""tests.test_config

Тесты для модуля src.config.
"""

import pytest

from src.config import get_config, Config


class TestGetConfig:
    """Набор тестов для функции get_config."""

    def test_returns_config_instance(self):
        """get_config возвращает экземпляр Config."""
        config = get_config()
        assert isinstance(config, Config)

    def test_returns_singleton_by_lru_cache(self):
        """get_config возвращает один и тот же объект благодаря lru_cache."""
        config_1 = get_config()
        config_2 = get_config()
        assert config_1 is config_2

    def test_default_values(self):
        """Проверка значений по умолчанию."""
        config = get_config()
        assert config.TITLE == 'Agent-Assistant'
        assert config.DESCRIPTION == (
            'Проект AI‑агента, для работы с внутренней документацией.'
        )
        assert config.VERSION == '0.1_MVP'
        assert config.DEBUG is False

    def test_debug_false_by_default(self):
        """DEBUG по умолчанию равен False."""
        config = get_config()
        assert config.DEBUG is False

    def test_override_debug_via_env(self, monkeypatch):
        """DEBUG переопределяется через переменную окружения."""
        monkeypatch.setenv('DEBUG', 'true')
        # Сбрасываем кэш, чтобы get_config создал новый экземпляр
        get_config.cache_clear()
        config = get_config()
        assert config.DEBUG is True

    def test_override_title_via_env(self, monkeypatch):
        """TITLE переопределяется через переменную окружения."""
        monkeypatch.setenv('TITLE', 'Custom Title')
        get_config.cache_clear()
        config = get_config()
        assert config.TITLE == 'Custom Title'

    def test_extra_env_ignored(self, monkeypatch):
        """Лишние переменные окружения игнорируются (extra='ignore')."""
        monkeypatch.setenv('UNKNOWN_VAR', 'should_be_ignored')
        get_config.cache_clear()
        config = get_config()
        # Проверяем, что неизвестное поле не добавилось
        with pytest.raises(AttributeError):
            _ = config.UNKNOWN_VAR

    def test_cache_clear_creates_new_instance(self):
        """После cache_clear get_config возвращает новый объект."""
        config_1 = get_config()
        get_config.cache_clear()
        config_2 = get_config()
        assert config_1 is not config_2
