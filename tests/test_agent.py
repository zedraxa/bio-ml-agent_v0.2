# tests/test_agent.py
# ═══════════════════════════════════════════════════════════
#  Bio-ML Agent — Kapsamlı Unit Test Suite
#  Çalıştırma: pytest tests/ -v
# ═══════════════════════════════════════════════════════════

import json
import os
import re
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from agent import (
    is_dangerous_bash,
    safe_relpath,
    sanitize_content,
    normalize_user_message,
    extract_tool,
    current_project,
    generate_session_id,
    save_conversation,
    load_conversation,
    list_conversations,
    delete_conversation,
    run_python,
    run_bash,
    read_file,
    write_file,
    append_todo,
    setup_logger,
    DEFAULT_PROJECT,
)

from exceptions import (
    AgentError,
    ToolExecutionError,
    ToolTimeoutError,
    SecurityViolationError,
    FileOperationError,
    ValidationError,
)


# ═══════════════════════════════════════════════════════════
#  1. GÜVENLİK TESTLERİ — is_dangerous_bash()
# ═══════════════════════════════════════════════════════════

class TestIsDangerousBash:
    """Tehlikeli bash komutlarının engellenmesini test eder."""

    # ── Engellenmesi gereken komutlar ──

    def test_blocks_rm_rf_root(self):
        """rm -rf / engellenmeli."""
        assert is_dangerous_bash("rm -rf /") is not None

    def test_blocks_rm_rf_with_sudo(self):
        """sudo rm -rf / engellenmeli."""
        assert is_dangerous_bash("sudo rm -rf /") is not None

    def test_blocks_fork_bomb(self):
        """Fork bomb engellenmeli."""
        assert is_dangerous_bash(":(){ :|:& };:") is not None

    def test_blocks_dd_devzero(self):
        """dd if=/dev/zero engellenmeli."""
        assert is_dangerous_bash("dd if=/dev/zero of=/dev/sda") is not None

    def test_blocks_mkfs(self):
        """mkfs komutu engellenmeli."""
        assert is_dangerous_bash("mkfs.ext4 /dev/sda1") is not None

    def test_blocks_shutdown(self):
        """shutdown komutu engellenmeli."""
        assert is_dangerous_bash("shutdown -h now") is not None

    def test_blocks_reboot(self):
        """reboot komutu engellenmeli."""
        assert is_dangerous_bash("reboot") is not None

    def test_blocks_kill_pid1(self):
        """kill -9 1 (init process) engellenmeli."""
        assert is_dangerous_bash("kill -9 1") is not None

    # ── İzin verilmesi gereken komutlar ──

    def test_allows_ls(self):
        """ls komutu güvenli."""
        assert is_dangerous_bash("ls -la") is None

    def test_allows_cat(self):
        """cat komutu güvenli."""
        assert is_dangerous_bash("cat somefile.txt") is None

    def test_allows_mkdir(self):
        """mkdir komutu güvenli."""
        assert is_dangerous_bash("mkdir -p data/raw") is None

    def test_allows_pip_install(self):
        """pip install komutu güvenli."""
        assert is_dangerous_bash("pip install pandas") is None

    def test_allows_python_run(self):
        """python script çalıştırma güvenli."""
        assert is_dangerous_bash("python train.py") is None

    def test_allows_echo(self):
        """echo komutu güvenli."""
        assert is_dangerous_bash("echo 'hello world'") is None

    def test_allows_rm_single_file(self):
        """Tek dosya silme izin verilmeli (rm -rf / olmadan)."""
        assert is_dangerous_bash("rm temp.txt") is None

    def test_allows_grep(self):
        """grep komutu güvenli."""
        assert is_dangerous_bash("grep -r 'pattern' src/") is None

    def test_allows_wget(self):
        """wget komutu güvenli."""
        assert is_dangerous_bash("wget https://example.com/data.csv") is None

    def test_empty_command(self):
        """Boş komut güvenli."""
        assert is_dangerous_bash("") is None

    def test_whitespace_command(self):
        """Sadece boşluk içeren komut güvenli."""
        assert is_dangerous_bash("   ") is None


# ═══════════════════════════════════════════════════════════
#  2. GÜVENLİK TESTLERİ — safe_relpath()
# ═══════════════════════════════════════════════════════════

class TestSafeRelpath:
    """Path güvenlik kontrollerini test eder."""

    def test_blocks_absolute_path(self):
        """Absolute path SecurityViolationError fırlatmalı."""
        with pytest.raises(SecurityViolationError, match="Absolute path"):
            safe_relpath("/etc/passwd")

    def test_blocks_absolute_home(self):
        """Home dizini absolute path SecurityViolationError fırlatmalı."""
        with pytest.raises(SecurityViolationError, match="Absolute path"):
            safe_relpath("/home/user/file.txt")

    def test_blocks_traversal_double_dot(self):
        """.. ile path traversal SecurityViolationError fırlatmalı."""
        with pytest.raises(SecurityViolationError, match="Path traversal"):
            safe_relpath("../../etc/passwd")

    def test_blocks_traversal_single_level(self):
        """Tek seviye traversal engellenmeli."""
        with pytest.raises(SecurityViolationError, match="Path traversal"):
            safe_relpath("../secret.txt")

    def test_allows_relative_path(self):
        """Normal relative path izin verilmeli."""
        result = safe_relpath("data/raw/file.csv")
        assert result == "data/raw/file.csv"

    def test_allows_simple_filename(self):
        """Basit dosya adı izin verilmeli."""
        result = safe_relpath("train.py")
        assert result == "train.py"

    def test_allows_nested_relative(self):
        """İç içe relative path izin verilmeli."""
        result = safe_relpath("src/models/baseline.py")
        assert result == "src/models/baseline.py"

    def test_normalizes_path(self):
        """Path normalleştirilmeli (fazla / kaldırılmalı)."""
        result = safe_relpath("data//raw///file.csv")
        assert ".." not in result


# ═══════════════════════════════════════════════════════════
#  3. İÇERİK TEMİZLEME — sanitize_content()
# ═══════════════════════════════════════════════════════════

class TestSanitizeContent:
    """Code fence temizleme işlemini test eder."""

    def test_removes_python_fences(self):
        """```python ... ``` code fence'larını kaldırmalı."""
        content = "```python\nprint('hello')\n```"
        result = sanitize_content(content)
        assert "```" not in result
        assert "print('hello')" in result

    def test_removes_plain_fences(self):
        """``` ... ``` code fence'larını kaldırmalı."""
        content = "```\nsome code\n```"
        result = sanitize_content(content)
        assert "```" not in result
        assert "some code" in result

    def test_preserves_content_without_fences(self):
        """Fence olmayan içerik değiştirilmemeli."""
        content = "normal content\nwith multiple lines"
        result = sanitize_content(content)
        assert "normal content" in result
        assert "with multiple lines" in result

    def test_removes_bash_fences(self):
        """```bash ... ``` code fence'larını kaldırmalı."""
        content = "```bash\nls -la\n```"
        result = sanitize_content(content)
        assert "```" not in result
        assert "ls -la" in result

    def test_empty_content(self):
        """Boş içerik hata vermemeli."""
        result = sanitize_content("")
        assert result == ""

    def test_strips_leading_newlines(self):
        """Baştaki boş satırlar kaldırılmalı."""
        content = "\n\n\nsome content"
        result = sanitize_content(content)
        assert result.startswith("some content")


# ═══════════════════════════════════════════════════════════
#  4. MESAJ NORMALİZASYONU — normalize_user_message()
# ═══════════════════════════════════════════════════════════

class TestNormalizeUserMessage:
    """Kullanıcı mesajı normalleştirmesini test eder."""

    def test_normalizes_crlf(self):
        """Windows satır sonlarını normalleştirmeli."""
        result = normalize_user_message("hello\r\nworld")
        assert "\r" not in result

    def test_strips_whitespace_from_lines(self):
        """Her satırdan boşluklar kaldırılmalı."""
        result = normalize_user_message("  hello  \n  world  ")
        lines = result.split("\n")
        assert lines[0] == "hello"
        assert lines[1] == "world"

    def test_removes_empty_lines(self):
        """Boş satırlar kaldırılmalı."""
        result = normalize_user_message("hello\n\n\nworld")
        lines = result.split("\n")
        assert len(lines) == 2

    def test_simple_message(self):
        """Basit mesaj değişmemeli."""
        result = normalize_user_message("hello world")
        assert result == "hello world"

    def test_empty_message(self):
        """Boş mesaj hata vermemeli."""
        result = normalize_user_message("")
        assert result == ""


# ═══════════════════════════════════════════════════════════
#  5. TOOL ÇIKARMA — extract_tool()
# ═══════════════════════════════════════════════════════════

class TestExtractTool:
    """LLM çıktısından tool ayrıştırmayı test eder."""

    def test_extracts_python_tool(self):
        """<PYTHON>...</PYTHON> bloğunu ayrıştırmalı."""
        text = "İşte kod:\n<PYTHON>print('hello')</PYTHON>"
        tool, payload, outside = extract_tool(text)
        assert tool == "PYTHON"
        assert "print('hello')" in payload

    def test_extracts_bash_tool(self):
        """<BASH>...</BASH> bloğunu ayrıştırmalı."""
        text = "<BASH>ls -la</BASH>"
        tool, payload, outside = extract_tool(text)
        assert tool == "BASH"
        assert "ls -la" in payload

    def test_extracts_web_search_tool(self):
        """<WEB_SEARCH>...</WEB_SEARCH> bloğunu ayrıştırmalı."""
        text = "<WEB_SEARCH>python pandas tutorial</WEB_SEARCH>"
        tool, payload, outside = extract_tool(text)
        assert tool == "WEB_SEARCH"
        assert "python pandas tutorial" in payload

    def test_extracts_write_file_tool(self):
        """<WRITE_FILE>...</WRITE_FILE> bloğunu ayrıştırmalı."""
        text = "<WRITE_FILE>path: test.py\n---\nprint('test')</WRITE_FILE>"
        tool, payload, outside = extract_tool(text)
        assert tool == "WRITE_FILE"
        assert "path:" in payload

    def test_extracts_read_file_tool(self):
        """<READ_FILE>...</READ_FILE> bloğunu ayrıştırmalı."""
        text = "<READ_FILE>data/raw/file.csv</READ_FILE>"
        tool, payload, outside = extract_tool(text)
        assert tool == "READ_FILE"

    def test_extracts_todo_tool(self):
        """<TODO>...</TODO> bloğunu ayrıştırmalı."""
        text = "<TODO>Model karşılaştırma yap</TODO>"
        tool, payload, outside = extract_tool(text)
        assert tool == "TODO"

    def test_no_tool_returns_none(self):
        """Tool olmayan metin None döndürmeli."""
        text = "Bu bir normal yanıttır."
        tool, payload, outside = extract_tool(text)
        assert tool is None
        assert payload is None

    def test_empty_text(self):
        """Boş metin hata vermemeli."""
        tool, payload, outside = extract_tool("")
        assert tool is None

    def test_none_text(self):
        """None metin hata vermemeli."""
        tool, payload, outside = extract_tool(None)
        assert tool is None

    def test_case_insensitive(self):
        """Büyük/küçük harf duyarsız olmalı."""
        text = "<python>print('hello')</python>"
        tool, payload, outside = extract_tool(text)
        assert tool == "PYTHON"

    def test_outside_text_captured(self):
        """Tool dışındaki metin 'outside' olarak döndürülmeli."""
        text = "Önceki metin\n<BASH>ls</BASH>\nSonraki metin"
        tool, payload, outside = extract_tool(text)
        assert tool == "BASH"
        assert "Önceki metin" in outside or "Sonraki metin" in outside


# ═══════════════════════════════════════════════════════════
#  6. PROJE YÖNETİMİ — current_project()
# ═══════════════════════════════════════════════════════════

class TestCurrentProject:
    """Aktif proje belirleme işlemini test eder."""

    def test_default_project(self):
        """Env değişkeni yoksa varsayılan proje adı dönmeli."""
        os.environ.pop("AGENT_PROJECT", None)
        assert current_project() == DEFAULT_PROJECT

    def test_custom_project(self):
        """Env değişkeni ayarlanmışsa o değer dönmeli."""
        os.environ["AGENT_PROJECT"] = "wine_quality"
        assert current_project() == "wine_quality"
        os.environ.pop("AGENT_PROJECT", None)


# ═══════════════════════════════════════════════════════════
#  7. OTURUM KİMLİĞİ — generate_session_id()
# ═══════════════════════════════════════════════════════════

class TestGenerateSessionId:
    """Oturum ID üretimini test eder."""

    def test_returns_string(self):
        """String döndürmeli."""
        sid = generate_session_id()
        assert isinstance(sid, str)

    def test_contains_timestamp(self):
        """Zaman damgası içermeli (YYYYMMDD formatında)."""
        sid = generate_session_id()
        # İlk 8 karakter tarih olmalı
        date_part = sid[:8]
        assert date_part.isdigit()

    def test_unique_ids(self):
        """Her çağrıda benzersiz ID üretmeli."""
        ids = {generate_session_id() for _ in range(10)}
        assert len(ids) == 10

    def test_format_structure(self):
        """YYYYMMDD_HHMMSS_xxxxxxxx formatında olmalı."""
        sid = generate_session_id()
        parts = sid.split("_")
        assert len(parts) == 3
        assert len(parts[0]) == 8   # YYYYMMDD
        assert len(parts[1]) == 6   # HHMMSS
        assert len(parts[2]) == 8   # hex


# ═══════════════════════════════════════════════════════════
#  8. KONUŞMA GEÇMİŞİ — save/load/list/delete
# ═══════════════════════════════════════════════════════════

class TestConversationHistory:
    """Konuşma geçmişi CRUD işlemlerini test eder."""

    # ── save_conversation ──

    def test_save_creates_file(self, tmp_history_dir, sample_messages):
        """Kayıt dosya oluşturmalı."""
        sid = "test_session_001"
        path = save_conversation(tmp_history_dir, sid, sample_messages)
        assert path.exists()
        assert path.suffix == ".json"

    def test_save_content_valid_json(self, tmp_history_dir, sample_messages):
        """Kaydedilen dosya geçerli JSON olmalı."""
        sid = "test_session_002"
        path = save_conversation(tmp_history_dir, sid, sample_messages)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "messages" in data
        assert "session_id" in data
        assert data["session_id"] == sid

    def test_save_preserves_messages(self, tmp_history_dir, sample_messages):
        """Mesajlar doğru kaydedilmeli."""
        sid = "test_session_003"
        save_conversation(tmp_history_dir, sid, sample_messages)
        data = json.loads((tmp_history_dir / f"{sid}.json").read_text(encoding="utf-8"))
        assert len(data["messages"]) == len(sample_messages)

    def test_save_includes_metadata(self, tmp_history_dir, sample_messages):
        """Metadata kaydedilmeli."""
        sid = "test_session_004"
        meta = {"created_at": "2026-02-20T12:00:00", "custom_key": "custom_val"}
        save_conversation(tmp_history_dir, sid, sample_messages, metadata=meta)
        data = json.loads((tmp_history_dir / f"{sid}.json").read_text(encoding="utf-8"))
        assert data["created_at"] == "2026-02-20T12:00:00"
        assert "metadata" in data

    def test_save_extracts_summary(self, tmp_history_dir, sample_messages):
        """İlk kullanıcı mesajından özet çıkarılmalı."""
        sid = "test_session_005"
        save_conversation(tmp_history_dir, sid, sample_messages)
        data = json.loads((tmp_history_dir / f"{sid}.json").read_text(encoding="utf-8"))
        assert "Merhaba" in data["summary"]

    def test_save_overwrites_existing(self, tmp_history_dir, sample_messages):
        """Aynı session_id ile tekrar kayıt üzerine yazmalı."""
        sid = "test_overwrite"
        save_conversation(tmp_history_dir, sid, sample_messages)
        new_msgs = sample_messages + [{"role": "user", "content": "Ek mesaj"}]
        save_conversation(tmp_history_dir, sid, new_msgs)
        data = json.loads((tmp_history_dir / f"{sid}.json").read_text(encoding="utf-8"))
        assert data["message_count"] == len(new_msgs)

    # ── load_conversation ──

    def test_load_returns_messages(self, tmp_history_dir, sample_messages):
        """Yükleme mesajları döndürmeli."""
        sid = "test_load_001"
        save_conversation(tmp_history_dir, sid, sample_messages)
        messages, metadata = load_conversation(tmp_history_dir, sid)
        assert len(messages) == len(sample_messages)

    def test_load_returns_metadata(self, tmp_history_dir, sample_messages):
        """Yükleme metadata döndürmeli."""
        sid = "test_load_002"
        save_conversation(tmp_history_dir, sid, sample_messages)
        messages, metadata = load_conversation(tmp_history_dir, sid)
        assert "session_id" in metadata
        assert "created_at" in metadata

    def test_load_nonexistent_raises(self, tmp_history_dir):
        """Olmayan oturum FileNotFoundError fırlatmalı."""
        with pytest.raises(FileNotFoundError):
            load_conversation(tmp_history_dir, "nonexistent_session")

    def test_save_load_roundtrip(self, tmp_history_dir, sample_messages):
        """Kaydet-yükle döngüsü veri kaybetmemeli."""
        sid = "test_roundtrip"
        save_conversation(tmp_history_dir, sid, sample_messages)
        loaded_msgs, _ = load_conversation(tmp_history_dir, sid)
        for orig, loaded in zip(sample_messages, loaded_msgs):
            assert orig["role"] == loaded["role"]
            assert orig["content"] == loaded["content"]

    # ── list_conversations ──

    def test_list_empty_directory(self, tmp_history_dir):
        """Boş klasör boş liste döndürmeli."""
        result = list_conversations(tmp_history_dir)
        assert result == []

    def test_list_returns_sessions(self, tmp_history_dir, sample_messages):
        """Kayıtlı oturumları listele."""
        for i in range(3):
            save_conversation(tmp_history_dir, f"session_{i:03d}", sample_messages)
        result = list_conversations(tmp_history_dir)
        assert len(result) == 3

    def test_list_respects_limit(self, tmp_history_dir, sample_messages):
        """Limit parametresine uymalı."""
        for i in range(10):
            save_conversation(tmp_history_dir, f"session_{i:03d}", sample_messages)
        result = list_conversations(tmp_history_dir, limit=5)
        assert len(result) == 5

    def test_list_contains_required_fields(self, tmp_history_dir, sample_messages):
        """Her oturum gerekli alanları içermeli."""
        save_conversation(tmp_history_dir, "session_fields", sample_messages)
        result = list_conversations(tmp_history_dir)
        assert len(result) == 1
        session = result[0]
        assert "session_id" in session
        assert "created_at" in session
        assert "message_count" in session
        assert "summary" in session

    # ── delete_conversation ──

    def test_delete_existing(self, tmp_history_dir, sample_messages):
        """Var olan oturumu silme True döndürmeli."""
        sid = "session_to_delete"
        save_conversation(tmp_history_dir, sid, sample_messages)
        assert delete_conversation(tmp_history_dir, sid) is True
        assert not (tmp_history_dir / f"{sid}.json").exists()

    def test_delete_nonexistent(self, tmp_history_dir):
        """Olmayan oturumu silme False döndürmeli."""
        assert delete_conversation(tmp_history_dir, "no_such_session") is False

    def test_delete_then_load_fails(self, tmp_history_dir, sample_messages):
        """Silindikten sonra yükleme hata vermeli."""
        sid = "session_delete_load"
        save_conversation(tmp_history_dir, sid, sample_messages)
        delete_conversation(tmp_history_dir, sid)
        with pytest.raises(FileNotFoundError):
            load_conversation(tmp_history_dir, sid)


# ═══════════════════════════════════════════════════════════
#  9. DOSYA İŞLEMLERİ — read_file / write_file / append_todo
# ═══════════════════════════════════════════════════════════

class TestFileOperations:
    """Dosya okuma/yazma işlemlerini test eder."""

    # ── read_file ──

    def test_read_existing_file(self, tmp_workspace):
        """Var olan dosyayı okumalı."""
        target = tmp_workspace / "test.txt"
        target.write_text("hello world", encoding="utf-8")
        result = read_file("test.txt", tmp_workspace)
        assert "hello world" in result

    def test_read_nonexistent_file(self, tmp_workspace):
        """Olmayan dosya FileOperationError fırlatmalı."""
        with pytest.raises(FileOperationError, match="bulunamadı"):
            read_file("nonexistent.txt", tmp_workspace)

    def test_read_directory_returns_error(self, tmp_workspace):
        """Klasör verilince FileOperationError fırlatmalı."""
        subdir = tmp_workspace / "subdir"
        subdir.mkdir()
        with pytest.raises(FileOperationError, match="klasör"):
            read_file("subdir", tmp_workspace)

    def test_read_truncates_large_file(self, tmp_workspace):
        """20KB'den büyük dosya kırpılmalı."""
        target = tmp_workspace / "large.txt"
        target.write_text("x" * 25000, encoding="utf-8")
        result = read_file("large.txt", tmp_workspace)
        assert "[TRUNCATED]" in result

    def test_read_blocks_absolute_path(self, tmp_workspace):
        """Absolute path engellenmeli."""
        with pytest.raises(SecurityViolationError):
            read_file("/etc/passwd", tmp_workspace)

    def test_read_blocks_traversal(self, tmp_workspace):
        """Path traversal engellenmeli."""
        with pytest.raises(SecurityViolationError):
            read_file("../../etc/passwd", tmp_workspace)

    # ── write_file ──

    def test_write_creates_file(self, tmp_workspace):
        """Dosya oluşturulmalı."""
        os.environ["AGENT_PROJECT"] = "test_proj"
        payload = "path: test_proj/output.txt\n---\nHello World"
        result = write_file(payload, tmp_workspace)
        assert "[OK]" in result
        assert (tmp_workspace / "test_proj" / "output.txt").exists()

    def test_write_creates_directories(self, tmp_workspace):
        """Ara klasörler otomatik oluşturulmalı."""
        os.environ["AGENT_PROJECT"] = "test_proj"
        payload = "path: test_proj/deep/nested/dir/file.txt\n---\ncontent"
        result = write_file(payload, tmp_workspace)
        assert "[OK]" in result
        assert (tmp_workspace / "test_proj" / "deep" / "nested" / "dir" / "file.txt").exists()

    def test_write_invalid_format_no_separator(self, tmp_workspace):
        """--- ayırıcı yoksa ValidationError fırlatmalı."""
        payload = "path: somefile.txt\ncontent without separator"
        with pytest.raises(ValidationError, match="ayırıcı"):
            write_file(payload, tmp_workspace)

    def test_write_invalid_format_no_path(self, tmp_workspace):
        """path: satırı yoksa ValidationError fırlatmalı."""
        payload = "no path here\n---\ncontent"
        with pytest.raises(ValidationError, match="path"):
            write_file(payload, tmp_workspace)

    def test_write_sanitizes_content(self, tmp_workspace):
        """Code fence'lar temizlenmeli."""
        os.environ["AGENT_PROJECT"] = "test_proj"
        payload = "path: test_proj/code.py\n---\n```python\nprint('hello')\n```"
        write_file(payload, tmp_workspace)
        content = (tmp_workspace / "test_proj" / "code.py").read_text(encoding="utf-8")
        assert "```" not in content
        assert "print('hello')" in content

    # ── append_todo ──

    def test_todo_creates_file(self, tmp_workspace):
        """TODO dosyası oluşturulmalı."""
        os.environ["AGENT_PROJECT"] = "test_proj"
        result = append_todo("Model karşılaştırma yap", tmp_workspace)
        assert "[OK]" in result
        todo_path = tmp_workspace / "test_proj" / "todo.md"
        assert todo_path.exists()

    def test_todo_appends_content(self, tmp_workspace):
        """İçerik eklenmiş olmalı."""
        os.environ["AGENT_PROJECT"] = "test_proj"
        append_todo("Görev 1", tmp_workspace)
        append_todo("Görev 2", tmp_workspace)
        content = (tmp_workspace / "test_proj" / "todo.md").read_text(encoding="utf-8")
        assert "Görev 1" in content
        assert "Görev 2" in content

    def test_todo_includes_timestamp(self, tmp_workspace):
        """Zaman damgası eklenmiş olmalı."""
        os.environ["AGENT_PROJECT"] = "test_proj"
        append_todo("Zaman testi", tmp_workspace)
        content = (tmp_workspace / "test_proj" / "todo.md").read_text(encoding="utf-8")
        # YYYY-MM-DD formatında tarih olmalı
        assert re.search(r"\d{4}-\d{2}-\d{2}", content)

    def test_todo_empty_payload(self, tmp_workspace):
        """Boş payload ValidationError fırlatmalı."""
        with pytest.raises(ValidationError, match="boş"):
            append_todo("", tmp_workspace)


# ═══════════════════════════════════════════════════════════
#  10. KOD ÇALIŞTIRMA — run_python / run_bash
# ═══════════════════════════════════════════════════════════

class TestCodeExecution:
    """Python ve Bash kod çalıştırmayı test eder."""

    # ── run_python ──

    def test_python_simple_output(self, tmp_workspace):
        """Basit Python kodu çalıştırılmalı."""
        result = run_python("print('Merhaba Dünya')", tmp_workspace)
        assert "Merhaba Dünya" in result

    def test_python_math(self, tmp_workspace):
        """Matematiksel işlem doğru sonuç vermeli."""
        result = run_python("print(2 + 3)", tmp_workspace)
        assert "5" in result

    def test_python_error_captured(self, tmp_workspace):
        """Python hatası yakalanmalı."""
        result = run_python("raise ValueError('test error')", tmp_workspace)
        assert "ValueError" in result or "test error" in result

    def test_python_timeout(self, tmp_workspace):
        """Timeout aşılınca ToolTimeoutError fırlatmalı."""
        with pytest.raises(ToolTimeoutError):
            run_python("import time; time.sleep(10)", tmp_workspace, timeout_s=2)

    def test_python_imports_work(self, tmp_workspace):
        """Standart kütüphane import'ları çalışmalı."""
        result = run_python("import json; print(json.dumps({'key': 'val'}))", tmp_workspace)
        assert "key" in result

    def test_python_multiline(self, tmp_workspace):
        """Çok satırlı kod çalışmalı."""
        code = """
x = 10
y = 20
print(x + y)
"""
        result = run_python(code, tmp_workspace)
        assert "30" in result

    def test_python_cleanup_tmp_file(self, tmp_workspace):
        """Geçici _tmp_run.py dosyası temizlenmeli."""
        run_python("print('test')", tmp_workspace)
        assert not (tmp_workspace / "_tmp_run.py").exists()

    # ── run_bash ──

    def test_bash_echo(self, tmp_workspace):
        """echo komutu çalışmalı."""
        result = run_bash("echo 'Merhaba'", tmp_workspace)
        assert "Merhaba" in result

    def test_bash_pwd(self, tmp_workspace):
        """pwd workspace dizinini döndürmeli."""
        result = run_bash("pwd", tmp_workspace)
        assert str(tmp_workspace) in result

    def test_bash_dangerous_blocked(self, tmp_workspace):
        """Tehlikeli komut SecurityViolationError fırlatmalı."""
        with pytest.raises(SecurityViolationError):
            run_bash("rm -rf /", tmp_workspace)

    def test_bash_timeout(self, tmp_workspace):
        """Timeout aşılınca ToolTimeoutError fırlatmalı."""
        with pytest.raises(ToolTimeoutError):
            run_bash("sleep 10", tmp_workspace, timeout_s=2)

    def test_bash_pipe(self, tmp_workspace):
        """Pipe kullanımı çalışmalı."""
        result = run_bash("echo 'hello world' | wc -w", tmp_workspace)
        assert "2" in result

    def test_bash_exit_code_nonzero(self, tmp_workspace):
        """Başarısız komut çıktı üretmeli."""
        result = run_bash("ls nonexistent_dir_xyz 2>&1", tmp_workspace)
        # Hata mesajı ya da exit code bilgisi olmalı
        assert len(result) > 0


# ═══════════════════════════════════════════════════════════
#  11. LOGLAMA — setup_logger
# ═══════════════════════════════════════════════════════════

class TestLogging:
    """Loglama sistemini test eder."""

    def test_logger_creates_log_file(self, tmp_path):
        """Log dosyası oluşturulmalı."""
        log_dir = tmp_path / "test_logs"
        logger = setup_logger(log_dir, "DEBUG")
        logger.info("Test log mesajı")

        # Handler'ları flush et
        for handler in logger.handlers:
            handler.flush()

        log_file = log_dir / "agent.log"
        assert log_file.exists()

        # Temizleme: handler'ları kaldır
        logger.handlers.clear()

    def test_logger_creates_directory(self, tmp_path):
        """Log klasörü otomatik oluşturulmalı."""
        log_dir = tmp_path / "nested" / "log" / "dir"
        logger = setup_logger(log_dir, "INFO")
        assert log_dir.exists()
        logger.handlers.clear()

    def test_logger_returns_logger(self, tmp_path):
        """Logger nesnesi döndürülmeli."""
        import logging
        log_dir = tmp_path / "logger_test"
        logger = setup_logger(log_dir, "INFO")
        assert isinstance(logger, logging.Logger)
        logger.handlers.clear()


# ═══════════════════════════════════════════════════════════
#  12. EDGE CASE & ENTEGRASYON TESTLERİ
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:
    """Sınır durumlarını ve entegrasyon senaryolarını test eder."""

    def test_turkish_characters_in_content(self, tmp_workspace):
        """Türkçe karakterler doğru işlenmeli."""
        os.environ["AGENT_PROJECT"] = "test_proj"
        payload = "path: test_proj/turkce.txt\n---\nÇöğüşı merhaba dünya"
        result = write_file(payload, tmp_workspace)
        assert "[OK]" in result
        content = (tmp_workspace / "test_proj" / "turkce.txt").read_text(encoding="utf-8")
        assert "Çöğüşı" in content

    def test_unicode_in_messages(self, tmp_history_dir):
        """Unicode karakterler konuşma geçmişinde korunmalı."""
        messages = [
            {"role": "user", "content": "🧬 Protein yapısı analiz et 🔬"},
            {"role": "assistant", "content": "Tabii! İşte analiz 📊"},
        ]
        sid = "unicode_test"
        save_conversation(tmp_history_dir, sid, messages)
        loaded, _ = load_conversation(tmp_history_dir, sid)
        assert "🧬" in loaded[0]["content"]
        assert "📊" in loaded[1]["content"]

    def test_large_conversation_save_load(self, tmp_history_dir):
        """Büyük konuşma kaydedilip yüklenebilmeli."""
        messages = [
            {"role": "user" if i % 2 == 0 else "assistant",
             "content": f"Mesaj #{i}: " + "x" * 500}
            for i in range(100)
        ]
        sid = "large_conversation"
        save_conversation(tmp_history_dir, sid, messages)
        loaded, _ = load_conversation(tmp_history_dir, sid)
        assert len(loaded) == 100

    def test_concurrent_session_ids(self):
        """Hızlı art arda üretilen ID'ler benzersiz olmalı."""
        ids = [generate_session_id() for _ in range(50)]
        assert len(set(ids)) == 50

    def test_python_writes_to_workspace(self, tmp_workspace):
        """Python kodu workspace'e dosya yazabilmeli."""
        code = """
with open('test_output.txt', 'w') as f:
    f.write('Python tarafından yazıldı')
print('Dosya yazıldı')
"""
        result = run_python(code, tmp_workspace)
        assert "Dosya yazıldı" in result
        # run_python appends current_project() to workspace, so check under project dir
        proj = current_project()
        expected_path = tmp_workspace / proj / "test_output.txt"
        assert expected_path.exists(), f"File not found at {expected_path}"

    def test_extract_tool_with_multiline_payload(self):
        """Çok satırlı tool payload'ı doğru ayrıştırılmalı."""
        text = """<PYTHON>
import pandas as pd
df = pd.DataFrame({'a': [1,2,3]})
print(df.shape)
</PYTHON>"""
        tool, payload, outside = extract_tool(text)
        assert tool == "PYTHON"
        assert "import pandas" in payload
        assert "print(df.shape)" in payload
