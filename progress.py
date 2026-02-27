# progress.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Terminal İlerleme Göstergesi (Spinner)
#  Uzun süren işlemler için animasyonlu geri bildirim.
# ═══════════════════════════════════════════════════════════

import itertools
import sys
import threading
import time


class Spinner:
    """Uzun süren işlemler için animasyonlu terminal spinner'ı.

    Context manager olarak kullanılır:

        with Spinner("🧠 LLM düşünüyor"):
            result = llm_chat(model, messages)
        # Çıktı: ✓ 🧠 LLM düşünüyor (3.2s)

    Attributes:
        message:  Spinner yanında gösterilecek mesaj.
        _frames:  Braille animasyon kareleri.
        _delay:   Kareler arası bekleme süresi (saniye).
    """

    _frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
    _delay = 0.1

    def __init__(self, message: str = "Çalışıyor"):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time: float = 0.0
        self._elapsed: float = 0.0
        self._success: bool = True
        self._is_tty: bool = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()

    # ── Context Manager ──

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._success = exc_type is None
        self.stop()
        return False  # exception'ları yutma, aynen fırlat

    # ── Public API ──

    def start(self) -> None:
        """Spinner animasyonunu başlat."""
        self._stop_event.clear()
        self._start_time = time.time()
        if self._is_tty:
            self._thread = threading.Thread(target=self._animate, daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Spinner animasyonunu durdur ve sonuç satırını yazdır."""
        self._stop_event.set()
        self._elapsed = time.time() - self._start_time
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._print_result()

    @property
    def elapsed(self) -> float:
        """Son çalışmanın toplam süresi (saniye)."""
        return self._elapsed

    def update(self, message: str) -> None:
        """Spinner mesajını güncelleyerek değiştir."""
        self.message = message

    # ── Internal ──

    def _animate(self) -> None:
        """Arka plan thread'inde braille animasyonunu çalıştır."""
        cycle = itertools.cycle(self._frames)
        while not self._stop_event.is_set():
            frame = next(cycle)
            elapsed = time.time() - self._start_time
            text = f"\r{frame} {self.message}... ({elapsed:.1f}s)"
            sys.stdout.write(text)
            sys.stdout.flush()
            self._stop_event.wait(self._delay)

    def _print_result(self) -> None:
        """Sonuç satırını yazdır: ✓ başarı veya ✗ hata."""
        if self._is_tty:
            # Önceki satırı temizle
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
        indicator = "✓" if self._success else "✗"
        elapsed_str = f"({self._elapsed:.1f}s)"
        print(f"{indicator} {self.message} {elapsed_str}")


# ── Kısayol fonksiyonları ──

def spin(message: str = "Çalışıyor") -> Spinner:
    """Spinner oluşturmak için kısayol.

    Kullanım:
        with spin("İşlem yapılıyor"):
            do_something()
    """
    return Spinner(message)
