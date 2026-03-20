import time
import json
import logging
from typing import Dict, Any, List
from threading import Lock
from pathlib import Path

log = logging.getLogger("bio_ml_agent")

class SessionTelemetry:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.llm_calls: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
    def record_llm_call(self, model: str, latency_ms: float, prompt_tokens: int, completion_tokens: int):
        self.llm_calls.append({
            "model": model,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "timestamp": time.time()
        })
        
    def record_tool_call(self, tool_name: str, latency_ms: float, success: bool):
        self.tool_calls.append({
            "tool_name": tool_name,
            "latency_ms": latency_ms,
            "success": success,
            "timestamp": time.time()
        })
        
    def get_stats(self) -> Dict[str, Any]:
        total_prompt_tokens = sum(call.get("prompt_tokens", 0) for call in self.llm_calls)
        total_completion_tokens = sum(call.get("completion_tokens", 0) for call in self.llm_calls)
        total_llm_latency = float(sum(call.get("latency_ms", 0.0) for call in self.llm_calls))
        total_tool_latency = float(sum(call.get("latency_ms", 0.0) for call in self.tool_calls))
        
        return {
            "session_id": self.session_id,
            "duration_s": round(float(time.time() - self.start_time), 2),
            "total_llm_calls": len(self.llm_calls),
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "avg_llm_latency_ms": round(float(total_llm_latency) / max(len(self.llm_calls), 1), 2),
            "total_tool_calls": len(self.tool_calls),
            "successful_tool_calls": sum(1 for call in self.tool_calls if call.get("success")),
            "avg_tool_latency_ms": round(float(total_tool_latency) / max(len(self.tool_calls), 1), 2)
        }

class TelemetryManager:
    _instance = None
    _lock = Lock()
    _sessions: Dict[str, SessionTelemetry] = {}
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(TelemetryManager, cls).__new__(cls)
        return cls._instance
        
    def get_session(self, session_id: str) -> SessionTelemetry:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionTelemetry(session_id)
            return self._sessions[session_id]
            
    def remove_session(self, session_id: str):
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

# Global instance for easy access
telemetry = TelemetryManager()
