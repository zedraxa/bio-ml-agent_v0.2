# tests/test_progress.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Progress (Spinner) Unit Test Suite
#  Çalıştırma: pytest tests/test_progress.py -v
# ═══════════════════════════════════════════════════════════

import io
import sys
import time
import threading

import pytest

from progress import Spinner, spin


# ═══════════════════════════════════════════════════════════
#  1. TEMEL SPINNER TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestSpinnerBasic:
    """Spinner'ın temel davranışlarını doğrular."""

    def test_context_manager_starts_and_stops(self):
        """Spinner context manager ile başlayıp durmalı."""
        s = Spinner("Test")
        s._is_tty = False  # Animasyon devre dışı bırak
        with s:
            time.sleep(0.05)
        assert s.elapsed > 0

    def test_elapsed_time_tracked(self):
        """Geçen süre doğru ölçülmeli."""
        with Spinner("Bekle") as s:
            s._is_tty = False
            time.sleep(0.2)
        assert s.elapsed >= 0.15

    def test_success_flag_on_normal_exit(self):
        """Hatasız çıkışta _success=True olmalı."""
        s = Spinner("Test")
        s._is_tty = False
        with s:
            pass
        assert s._success is True

    def test_success_flag_on_exception(self):
        """Hata durumunda _success=False olmalı."""
        s = Spinner("Test")
        s._is_tty = False
        with pytest.raises(ValueError):
            with s:
                raise ValueError("test")
        assert s._success is False

    def test_exception_propagation(self):
        """Exception context manager'dan geçmeli (yutulmamalı)."""
        with pytest.raises(RuntimeError, match="test error"):
            with Spinner("Test") as s:
                s._is_tty = False
                raise RuntimeError("test error")

    def test_message_attribute(self):
        """Mesaj doğru saklanmalı."""
        s = Spinner("İşlem yapılıyor")
        assert s.message == "İşlem yapılıyor"

    def test_update_message(self):
        """update() mesajı değiştirmeli."""
        s = Spinner("Eski mesaj")
        s.update("Yeni mesaj")
        assert s.message == "Yeni mesaj"


# ═══════════════════════════════════════════════════════════
#  2. TTY TESPİT TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestTTYDetection:
    """TTY olmayan ortamda animasyon devre dışı olmalı."""

    def test_non_tty_no_thread(self):
        """TTY olmayan ortamda thread başlatılmamalı."""
        s = Spinner("Test")
        s._is_tty = False
        with s:
            assert s._thread is None

    def test_non_tty_still_tracks_elapsed(self):
        """TTY olmasa bile süre takibi çalışmalı."""
        s = Spinner("Test")
        s._is_tty = False
        with s:
            time.sleep(0.1)
        assert s.elapsed >= 0.05


# ═══════════════════════════════════════════════════════════
#  3. SONUÇ GÖSTERGESİ TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestResultIndicator:
    """_print_result() doğru gösterge sembolünü kullanmalı."""

    def test_success_indicator(self, capsys):
        """Başarılı çıkışta ✓ gösterilmeli."""
        s = Spinner("Test işlem")
        s._is_tty = False
        with s:
            pass
        captured = capsys.readouterr()
        assert "✓" in captured.out
        assert "Test işlem" in captured.out

    def test_error_indicator(self, capsys):
        """Hatalı çıkışta ✗ gösterilmeli."""
        s = Spinner("Test işlem")
        s._is_tty = False
        try:
            with s:
                raise ValueError("hata")
        except ValueError:
            pass
        captured = capsys.readouterr()
        assert "✗" in captured.out
        assert "Test işlem" in captured.out

    def test_elapsed_time_in_output(self, capsys):
        """Çıktıda geçen süre gösterilmeli."""
        s = Spinner("Süre test")
        s._is_tty = False
        with s:
            time.sleep(0.1)
        captured = capsys.readouterr()
        # (X.Xs) formatında süre olmalı
        assert "s)" in captured.out


# ═══════════════════════════════════════════════════════════
#  4. TTY ANİMASYON TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestTTYAnimation:
    """TTY ortamında animasyonun çalıştığını doğrular."""

    def test_thread_starts_for_tty(self):
        """TTY ortamında thread başlatılmalı."""
        s = Spinner("Test")
        s._is_tty = True
        s.start()
        assert s._thread is not None
        assert s._thread.is_alive()
        s._success = True
        s.stop()

    def test_thread_stops_cleanly(self):
        """stop() sonrası thread durmalı."""
        s = Spinner("Test")
        s._is_tty = True
        s.start()
        s._success = True
        s.stop()
        assert s._thread is None

    def test_frames_are_braille(self):
        """Spinner kareleri braille karakterleri olmalı."""
        for frame in Spinner._frames:
            # Braille Patterns Unicode block: U+2800 to U+28FF
            assert 0x2800 <= ord(frame) <= 0x28FF


# ═══════════════════════════════════════════════════════════
#  5. KISAYOL FONKSİYON TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestSpinShortcut:
    """spin() kısayol fonksiyonunu doğrular."""

    def test_returns_spinner_instance(self):
        """spin() Spinner döndürmeli."""
        result = spin("Test")
        assert isinstance(result, Spinner)

    def test_spin_message_passed(self):
        """spin() mesajı aktarmalı."""
        result = spin("Özel mesaj")
        assert result.message == "Özel mesaj"

    def test_spin_as_context_manager(self):
        """spin() context manager olarak çalışmalı."""
        with spin("Test") as s:
            s._is_tty = False
            time.sleep(0.05)
        assert s.elapsed > 0
