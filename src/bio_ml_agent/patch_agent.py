import re

with open('agent.py', 'r') as f:
    text = f.read()

text = text.replace(
'''def extract_tool(text: str) -> Tuple[Optional[str], Optional[str], str]:
    m = TOOL_RE.search(text or "")
    if not m:
        return None, None, text or ""
    tool = m.group(1).upper()
    payload = m.group(2)
    outside = TOOL_RE.sub("", text).strip()
    return tool, payload, outside''',
'''def extract_tools(text: str) -> Tuple[List[Tuple[str, str]], str]:
    tools = []
    for m in TOOL_RE.finditer(text or ""):
        tools.append((m.group(1).upper(), m.group(2)))
    outside = TOOL_RE.sub("", text or "").strip()
    return tools, outside'''
)

start_idx = text.find('            tool, payload, outside = extract_tool(assistant)')
end_idx = text.find('            # Her tool adımından sonra otomatik kaydet', start_idx)

if start_idx != -1 and end_idx != -1:
    old_loop = text[start_idx:end_idx]
    
    new_loop = '''            tools_to_run, outside = extract_tools(assistant)

            if not tools_to_run:
                py_m = FENCED_PY_RE.search(assistant)
                bash_m = FENCED_BASH_RE.search(assistant)
                if py_m and (not bash_m or len(py_m.group(1)) >= len(bash_m.group(1))):
                    tools_to_run = [("PYTHON", py_m.group(1))]
                    outside = FENCED_PY_RE.sub("", assistant).strip()
                    log.info("🔧 Fenced code block'tan PYTHON tool algılandı")
                elif bash_m:
                    tools_to_run = [("BASH", bash_m.group(1))]
                    outside = FENCED_BASH_RE.sub("", assistant).strip()
                    log.info("🔧 Fenced code block'tan BASH tool algılandı")
                else:
                    log.info("💬 Agent düz metin yanıtı verdi (tool yok) | adım=%d", step + 1)
                    print("\\n🤖 Agent:\\n", assistant)
                    messages.append({"role": "assistant", "content": assistant})
                    # Asistan cevabından sonra otomatik kaydet
                    save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                    break

            if outside:
                log.warning("⚠️ Tool bloğu dışında metin vardı | dış_metin_uzunluk=%d", len(outside))
                print("\\n⚠️ Uyarı: Tool bloğu dışında metin vardı; yine de tool çalıştırılıyor.\\n")

            messages.append({"role": "assistant", "content": assistant})
            
            all_outputs = []
            break_loop = False

            for tool, payload in tools_to_run:
                log.info("🔧 Tool algılandı: %s | payload_uzunluk=%d", tool, len(payload or ""))
                try:
                    if tool == "PYTHON":
                        # PYTHON kodlarını projenin kendi klasöründe çalıştır
                        py_cwd = cfg.workspace / project
                        py_cwd.mkdir(parents=True, exist_ok=True)
                        with Spinner("🐍 Python çalıştırılıyor"):
                            out = run_python(payload, py_cwd, timeout_s=cfg.timeout)
                    elif tool == "BASH":
                        # BASH komutlarını projenin kendi klasöründe çalıştır
                        bash_cwd = cfg.workspace / project
                        bash_cwd.mkdir(parents=True, exist_ok=True)
                        with Spinner("💻 Bash çalıştırılıyor"):
                            out = run_bash(payload, bash_cwd, timeout_s=cfg.timeout)
                    elif tool == "WEB_SEARCH":
                        if not allow_web and not _cfg().security.allow_web_search:
                            out = "[BLOCKED] WEB_SEARCH is disabled. To enable for this request, include: ALLOW_WEB_SEARCH"
                        else:
                            with Spinner("🌐 Web'de aranıyor"):
                                out = web_search(payload)
                    elif tool == "WEB_OPEN":
                        with Spinner("📖 Sayfa okunuyor"):
                            out = web_open(payload)
                    elif tool == "READ_FILE":
                        out = read_file(payload, cfg.workspace)
                    elif tool == "WRITE_FILE":
                        out = write_file(payload, cfg.workspace)
                    elif tool == "TODO":
                        out = append_todo(payload, cfg.workspace)
                    elif tool == "RAG_SEARCH":
                        with Spinner("🔍 RAG'da aranıyor"):
                            results = rag.search(payload)
                            if not results:
                                out = "[RAG_SEARCH] Sonuç bulunamadı."
                            else:
                                out = "[RAG_SEARCH] Bulunan metinler:\\n\\n"
                                for i, r in enumerate(results, 1):
                                    out += f"--- Kaynak: {r['source']} (Mesafe: {r['distance']:.4f}) ---\\n"
                                    out += f"{r['document']}\\n\\n"
                    elif pm.get(tool):
                        with Spinner(f"🔌 {tool} çalıştırılıyor"):
                            out = pm.execute(tool, payload, cfg.workspace)
                    else:
                        out = f"[ERROR] Unknown tool: {tool}"

                except LLMConnectionError as e:
                    log.error("🧠 LLM bağlantı hatası | %s", e)
                    print(f"\\n{e.user_message()}")
                    print("\\n⏳ 5 saniye sonra tekrar denenecek...\\n")
                    time.sleep(5)
                    try:
                        with Spinner("🧠 LLM tekrar deneniyor"):
                            assistant = llm_chat(cfg.model, messages)
                        messages.append({"role": "assistant", "content": assistant})
                        save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                    except LLMConnectionError as e2:
                        log.error("🧠 LLM tekrar deneme başarısız | %s", e2)
                        print(f"\\n{e2.user_message()}")
                        print("\\n⚠️ LLM'e bağlanılamıyor. Lütfen Ollama servisini kontrol edin.\\n")
                        save_conversation(cfg.history_dir, session_id, messages, session_metadata)
                    break_loop = True
                    break

                except SecurityViolationError as e:
                    log.warning("🔒 Güvenlik ihlali | %s", e)
                    print(f"\\n{e.user_message()}")
                    out = e.tool_output()

                except ToolTimeoutError as e:
                    log.error("⏰ Zaman aşımı | %s", e)
                    print(f"\\n{e.user_message()}")
                    out = e.tool_output()

                except (ToolExecutionError, FileOperationError, ValidationError) as e:
                    log.error("🛠️ Tool hatası | %s", e)
                    print(f"\\n{e.user_message()}")
                    out = e.tool_output()

                except AgentError as e:
                    log.error("❌ Agent hatası | %s", e)
                    print(f"\\n{e.user_message()}")
                    out = e.tool_output()

                except Exception as e:
                    log.error("💥 Beklenmeyen hata | tool=%s | %s", tool, e, exc_info=True)
                    print(f"\\n❌ Beklenmeyen hata: {e}")
                    print(f"   💡 Öneri: Bu hatayı /logs komutuyla inceleyebilirsiniz.\\n")
                    out = f"[UNEXPECTED_ERROR] {type(e).__name__}: {e}"

                if tool in {"WEB_SEARCH", "WEB_OPEN"} and not out.startswith("["):
                    autosave_web_outputs(cfg, tool, out)

                log.info("🛠️ Tool tamamlandı | tool=%s | çıktı_uzunluk=%d", tool, len(out))
                print(f"\\n🛠️ {tool} output:\\n{out}\\n")
                all_outputs.append((tool, out))

            if break_loop:
                break
            
            user_msg = ""
            for t, o in all_outputs:
                user_msg += f"TOOL_OUTPUT ({t}):\\n{o}\\n\\n"
            user_msg += "Continue. If done, answer normally (no tool)."
            
            messages.append({
                "role": "user",
                "content": user_msg
            })

'''
    text = text.replace(old_loop, new_loop)
    with open('agent.py', 'w') as f:
        f.write(text)
    print("agent.py loop patched using substring replacement")
else:
    print("Could not find start or end index.")
