# tests/test_exceptions.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Exception Hierarchy Unit Test Suite
#  Çalıştırma: pytest tests/test_exceptions.py -v
# ═══════════════════════════════════════════════════════════

import pytest

from exceptions import (
    AgentError,
    ToolExecutionError,
    ToolTimeoutError,
    SecurityViolationError,
    LLMConnectionError,
    ConfigurationError,
    FileOperationError,
    ValidationError,
)


# ═══════════════════════════════════════════════════════════
#  1. HATA HİYERARŞİSİ TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestErrorHierarchy:
    """Hata sınıfı miras zincirini doğrular."""

    def test_agent_error_is_exception(self):
        """AgentError, Exception'dan türemeli."""
        assert issubclass(AgentError, Exception)

    def test_tool_execution_error_inherits_agent(self):
        """ToolExecutionError → AgentError."""
        assert issubclass(ToolExecutionError, AgentError)

    def test_tool_timeout_error_inherits_tool_execution(self):
        """ToolTimeoutError → ToolExecutionError → AgentError."""
        assert issubclass(ToolTimeoutError, ToolExecutionError)
        assert issubclass(ToolTimeoutError, AgentError)

    def test_security_violation_inherits_agent(self):
        """SecurityViolationError → AgentError."""
        assert issubclass(SecurityViolationError, AgentError)

    def test_llm_connection_inherits_agent(self):
        """LLMConnectionError → AgentError."""
        assert issubclass(LLMConnectionError, AgentError)

    def test_configuration_error_inherits_agent(self):
        """ConfigurationError → AgentError."""
        assert issubclass(ConfigurationError, AgentError)

    def test_file_operation_inherits_agent(self):
        """FileOperationError → AgentError."""
        assert issubclass(FileOperationError, AgentError)

    def test_validation_error_inherits_agent(self):
        """ValidationError → AgentError."""
        assert issubclass(ValidationError, AgentError)


# ═══════════════════════════════════════════════════════════
#  2. KULLANICI MESAJI TESTLERİ — user_message()
# ═══════════════════════════════════════════════════════════

class TestUserMessages:
    """user_message() çıktısını doğrular."""

    def test_basic_message(self):
        """Temel hata mesajı ❌ ile başlamalı."""
        e = AgentError("Test hatası")
        msg = e.user_message()
        assert "❌" in msg
        assert "Test hatası" in msg

    def test_message_with_details(self):
        """Detaylı hata mesajı 📋 içermeli."""
        e = AgentError("Hata", details="Ek detay")
        msg = e.user_message()
        assert "📋" in msg
        assert "Ek detay" in msg

    def test_message_with_suggestion(self):
        """Önerili hata mesajı 💡 içermeli."""
        e = AgentError("Hata", suggestion="Çözüm önerisi")
        msg = e.user_message()
        assert "💡" in msg
        assert "Çözüm önerisi" in msg

    def test_message_with_all_fields(self):
        """Tüm alanlar dolu mesaj tam formatlanmalı."""
        e = AgentError("Ana hata", details="Detay", suggestion="Öneri")
        msg = e.user_message()
        assert "❌" in msg
        assert "📋" in msg
        assert "💡" in msg

    def test_none_details_omitted(self):
        """details=None ise 📋 satırı olmamalı."""
        e = AgentError("Hata")
        msg = e.user_message()
        assert "📋" not in msg

    def test_none_suggestion_omitted(self):
        """suggestion=None ise 💡 satırı olmamalı."""
        e = AgentError("Hata")
        msg = e.user_message()
        assert "💡" not in msg


# ═══════════════════════════════════════════════════════════
#  3. TOOL ÇIKTISI TESTLERİ — tool_output()
# ═══════════════════════════════════════════════════════════

class TestToolOutput:
    """tool_output() çıktısını doğrular (LLM'e iletilen format)."""

    def test_format_includes_class_name(self):
        """tool_output() sınıf adını içermeli."""
        e = ToolExecutionError("PYTHON", "Hata oluştu")
        out = e.tool_output()
        assert "[ToolExecutionError]" in out

    def test_format_includes_message(self):
        """tool_output() hata mesajını içermeli."""
        e = ToolExecutionError("BASH", "Komut başarısız")
        out = e.tool_output()
        assert "Komut başarısız" in out

    def test_format_includes_details(self):
        """tool_output() detayları da içermeli."""
        e = AgentError("Hata", details="Ek bilgi")
        out = e.tool_output()
        assert "Ek bilgi" in out

    def test_no_details_no_pipe(self):
        """details yoksa | ayırıcı olmamalı."""
        e = AgentError("Hata")
        out = e.tool_output()
        assert " | " not in out


# ═══════════════════════════════════════════════════════════
#  4. ÖZEL HATA SINIFI TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestSpecificErrors:
    """Her hata sınıfının özel alanlarını doğrular."""

    def test_tool_execution_error_has_tool_name(self):
        """ToolExecutionError tool_name alanı içermeli."""
        e = ToolExecutionError("PYTHON", "Hata")
        assert e.tool_name == "PYTHON"

    def test_tool_timeout_has_seconds(self):
        """ToolTimeoutError timeout_seconds alanı içermeli."""
        e = ToolTimeoutError("BASH", 180)
        assert e.timeout_seconds == 180

    def test_tool_timeout_has_default_suggestion(self):
        """ToolTimeoutError varsayılan öneri içermeli."""
        e = ToolTimeoutError("PYTHON", 60)
        assert e.suggestion is not None
        assert "timeout" in e.suggestion.lower() or "bölün" in e.suggestion

    def test_security_violation_has_type(self):
        """SecurityViolationError violation_type alanı içermeli."""
        e = SecurityViolationError("Blok", violation_type="path_traversal")
        assert e.violation_type == "path_traversal"

    def test_security_violation_default_suggestion(self):
        """SecurityViolationError varsayılan öneri içermeli."""
        e = SecurityViolationError("Blok")
        assert e.suggestion is not None

    def test_llm_connection_has_model(self):
        """LLMConnectionError model alanı içermeli."""
        e = LLMConnectionError("qwen2.5:7b", "Bağlantı reddedildi")
        assert e.model == "qwen2.5:7b"

    def test_llm_connection_default_suggestion(self):
        """LLMConnectionError Ollama çözüm önerisi içermeli."""
        e = LLMConnectionError("model", "Hata")
        msg = e.user_message()
        assert "ollama" in msg.lower()

    def test_configuration_error_message(self):
        """ConfigurationError yapılandırma mesajı içermeli."""
        e = ConfigurationError("Geçersiz alan")
        assert "Yapılandırma" in str(e)

    def test_file_operation_has_fields(self):
        """FileOperationError operation ve path alanları içermeli."""
        e = FileOperationError("okuma", "data.csv", "Bulunamadı")
        assert e.operation == "okuma"
        assert e.path == "data.csv"

    def test_validation_error_has_field(self):
        """ValidationError field alanı içermeli."""
        e = ValidationError("query", "Boş olamaz")
        assert e.field == "query"


# ═══════════════════════════════════════════════════════════
#  5. CATCH SEMATİĞİ TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestCatchSemantics:
    """except bloklarında doğru yakalanmayı doğrular."""

    def test_timeout_caught_by_tool_execution(self):
        """ToolTimeoutError, ToolExecutionError except'iyle yakalanabilmeli."""
        with pytest.raises(ToolExecutionError):
            raise ToolTimeoutError("PYTHON", 180)

    def test_timeout_caught_by_agent_error(self):
        """ToolTimeoutError, AgentError except'iyle yakalanabilmeli."""
        with pytest.raises(AgentError):
            raise ToolTimeoutError("BASH", 60)

    def test_security_caught_by_agent_error(self):
        """SecurityViolationError, AgentError except'iyle yakalanabilmeli."""
        with pytest.raises(AgentError):
            raise SecurityViolationError("Test")

    def test_all_errors_are_exceptions(self):
        """Tüm hata sınıfları Exception ile yakalanabilmeli."""
        errors = [
            AgentError("a"),
            ToolExecutionError("T", "b"),
            ToolTimeoutError("T", 1),
            SecurityViolationError("c"),
            LLMConnectionError("m", "d"),
            ConfigurationError("e"),
            FileOperationError("o", "p", "f"),
            ValidationError("f", "g"),
        ]
        for err in errors:
            assert isinstance(err, Exception), f"{type(err).__name__} is not Exception"
