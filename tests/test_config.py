"""tests.test_config

Тесты для модуля src.config.
"""

import os
from pathlib import Path

import pytest

from src.config import get_config, Config, LogLevel, BASE_DIR, ENV_MODE


class TestLogLevel:
    """Набор тестов для перечисления LogLevel."""

    def test_enum_values(self):
        """LogLevel содержит все допустимые уровни."""
        assert LogLevel.DEBUG.value == 'DEBUG'
        assert LogLevel.INFO.value == 'INFO'
        assert LogLevel.WARNING.value == 'WARNING'
        assert LogLevel.ERROR.value == 'ERROR'
        assert LogLevel.CRITICAL.value == 'CRITICAL'

    def test_enum_inherits_str(self):
        """LogLevel наследуется от str."""
        assert isinstance(LogLevel.DEBUG, str)
        assert issubclass(LogLevel, str)

    def test_enum_members_are_strings(self):
        """Значения LogLevel можно использовать как строки."""
        assert LogLevel.INFO == 'INFO'
        assert LogLevel.DEBUG == 'DEBUG'


class TestModuleConstants:
    """Набор тестов для модульных констант BASE_DIR и ENV_MODE."""

    def test_base_dir_is_parent_of_src(self):
        """BASE_DIR указывает на корень проекта (родитель src)."""
        expected = Path(__file__).resolve().parent.parent
        assert BASE_DIR == expected

    def test_env_mode_defaults_to_development(self):
        """ENV_MODE по умолчанию равен 'development'."""
        assert ENV_MODE == os.getenv('ENVIRONMENT', 'development')


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


class TestLogLevelConfig:
    """Набор тестов для LOG_LEVEL в Config."""

    def test_default_log_level_is_info(self):
        """По умолчанию LOG_LEVEL равен INFO."""
        config = get_config()
        assert config.LOG_LEVEL == LogLevel.INFO

    def test_log_level_is_loglevel_enum(self):
        """LOG_LEVEL является экземпляром LogLevel."""
        config = get_config()
        assert isinstance(config.LOG_LEVEL, LogLevel)

    def test_log_level_from_env(self, monkeypatch):
        """LOG_LEVEL читается из переменной окружения."""
        monkeypatch.setenv('LOG_LEVEL', 'WARNING')
        get_config.cache_clear()
        config = get_config()
        assert config.LOG_LEVEL == LogLevel.WARNING

    def test_log_level_from_env_critical(self, monkeypatch):
        """LOG_LEVEL читается из .env как CRITICAL."""
        monkeypatch.setenv('LOG_LEVEL', 'CRITICAL')
        get_config.cache_clear()
        config = get_config()
        assert config.LOG_LEVEL == LogLevel.CRITICAL


class TestLogDir:
    """Набор тестов для LOG_DIR в Config."""

    def test_default_log_dir_is_base_dir_logs(self):
        """По умолчанию LOG_DIR указывает на BASE_DIR / 'logs'."""
        config = get_config()
        assert config.LOG_DIR == BASE_DIR / 'logs'

    def test_log_dir_is_path_instance(self):
        """LOG_DIR является экземпляром Path."""
        config = get_config()
        assert isinstance(config.LOG_DIR, Path)

    def test_log_dir_from_env(self, monkeypatch):
        """LOG_DIR переопределяется через переменную окружения."""
        monkeypatch.setenv('LOG_DIR', '/custom/log/path')
        get_config.cache_clear()
        config = get_config()
        assert config.LOG_DIR == Path('/custom/log/path')
