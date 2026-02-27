# exceptions.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Özel Hata Sınıfları (Exception Hierarchy)
# ═══════════════════════════════════════════════════════════
#
#  Tüm agent'a özel hatalar AgentError'dan türetilir.
#  Her hata sınıfı, kullanıcıya gösterilecek anlamlı Türkçe
#  mesajlar üretebilir.
#
#  Hiyerarşi:
#    AgentError
#    ├── ToolExecutionError
#    │   └── ToolTimeoutError
#    ├── SecurityViolationError
#    ├── LLMConnectionError
#    ├── ConfigurationError
#    ├── FileOperationError
#    └── ValidationError
# ═══════════════════════════════════════════════════════════


class AgentError(Exception):
    """Tüm agent hatalarının temel sınıfı.

    Attributes:
        details:    Hatanın teknik detayları (log için).
        suggestion: Kullanıcıya gösterilecek çözüm önerisi.
    """

    def __init__(self, message: str, *, details: str | None = None, suggestion: str | None = None):
        self.details = details
        self.suggestion = suggestion
        super().__init__(message)

    def user_message(self) -> str:
        """Kullanıcıya gösterilecek biçimlendirilmiş Türkçe mesaj."""
        parts = [f"❌ {self}"]
        if self.details:
            parts.append(f"   📋 Detay: {self.details}")
        if self.suggestion:
            parts.append(f"   💡 Öneri: {self.suggestion}")
        return "\n".join(parts)

    def tool_output(self) -> str:
        """LLM'e iletilecek kısa hata çıktısı (TOOL_OUTPUT formatında)."""
        tag = type(self).__name__
        msg = str(self)
        if self.details:
            msg += f" | {self.details}"
        return f"[{tag}] {msg}"


# ─────────────────────────────────────────────
#  Tool Çalıştırma Hataları
# ─────────────────────────────────────────────

class ToolExecutionError(AgentError):
    """Bir tool çalıştırılırken oluşan hata.

    Örnek: Python kodu çalışırken beklenmeyen hata, bash komutu başarısız.
    """

    def __init__(self, tool_name: str, message: str, **kwargs):
        self.tool_name = tool_name
        super().__init__(
            f"{tool_name} çalıştırma hatası: {message}",
            **kwargs,
        )


class ToolTimeoutError(ToolExecutionError):
    """Tool çalışması zaman aşımına uğradı.

    Örnek: Python kodu 180 saniyeden uzun sürdü.
    """

    def __init__(self, tool_name: str, timeout_seconds: int, **kwargs):
        self.timeout_seconds = timeout_seconds
        kwargs.setdefault(
            "suggestion",
            f"Kodunuzu daha küçük parçalara bölün veya timeout süresini artırın "
            f"(--timeout {timeout_seconds * 2}).",
        )
        super().__init__(
            tool_name,
            f"{timeout_seconds} saniye zaman aşımı süresini aştı",
            **kwargs,
        )


# ─────────────────────────────────────────────
#  Güvenlik Hataları
# ─────────────────────────────────────────────

class SecurityViolationError(AgentError):
    """Güvenlik politikası ihlali.

    Örnek: Tehlikeli bash komutu, path traversal girişimi.
    """

    def __init__(self, message: str, violation_type: str = "genel", **kwargs):
        self.violation_type = violation_type
        kwargs.setdefault("suggestion", "Güvenli komutlar ve relative path'ler kullanın.")
        super().__init__(
            f"Güvenlik ihlali ({violation_type}): {message}",
            **kwargs,
        )


# ─────────────────────────────────────────────
#  LLM Bağlantı Hataları
# ─────────────────────────────────────────────

class LLMConnectionError(AgentError):
    """LLM (Ollama) ile iletişim hatası.

    Örnek: Ollama servisi çalışmıyor, model bulunamadı.
    """

    def __init__(self, model: str, message: str, **kwargs):
        self.model = model
        kwargs.setdefault(
            "suggestion",
            f"Ollama servisinin çalıştığından emin olun: `ollama serve`\n"
            f"   Model yüklü mü kontrol edin: `ollama list`",
        )
        super().__init__(
            f"LLM bağlantı hatası (model={model}): {message}",
            **kwargs,
        )


# ─────────────────────────────────────────────
#  Yapılandırma Hataları
# ─────────────────────────────────────────────

class ConfigurationError(AgentError):
    """Yapılandırma dosyası veya parametre hatası.

    Örnek: config.yaml geçersiz, zorunlu alan eksik.
    """

    def __init__(self, message: str, **kwargs):
        kwargs.setdefault("suggestion", "config.yaml dosyasını kontrol edin.")
        super().__init__(f"Yapılandırma hatası: {message}", **kwargs)


# ─────────────────────────────────────────────
#  Dosya İşlem Hataları
# ─────────────────────────────────────────────

class FileOperationError(AgentError):
    """Dosya okuma/yazma hatası.

    Örnek: Dosya bulunamadı, yazma izni yok, geçersiz format.
    """

    def __init__(self, operation: str, path: str, message: str, **kwargs):
        self.operation = operation
        self.path = path
        super().__init__(
            f"Dosya {operation} hatası ({path}): {message}",
            **kwargs,
        )


# ─────────────────────────────────────────────
#  Doğrulama Hataları
# ─────────────────────────────────────────────

class ValidationError(AgentError):
    """Girdi doğrulama hatası.

    Örnek: Boş sorgu, geçersiz URL formatı, eksik alan.
    """

    def __init__(self, field: str, message: str, **kwargs):
        self.field = field
        super().__init__(
            f"Doğrulama hatası ({field}): {message}",
            **kwargs,
        )
