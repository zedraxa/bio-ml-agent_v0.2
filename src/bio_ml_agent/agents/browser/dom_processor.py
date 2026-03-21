import logging
import json
from typing import List, Dict, Any, Optional

log = logging.getLogger("browser.dom_processor")

class DOMPruner:
    """
    DOMPruner: Dev devasa DOM ağaçlarını LLM'lerin işleyebileceği boyuta indirir.
    - Gereksiz script, style ve boş tag'leri temizler.
    - Sadece interaktif (button, input, a) veya içerik dolu (h1-h6, p) node'ları korur.
    - Her node'un (x, y, w, h) bilgisini saklar.
    """
    
    @staticmethod
    def prune(raw_dom: Dict[str, Any]) -> Dict[str, Any]:
        """Ham DOM verisini temizler ve hiyerarşiyi korur."""
        def _process(node):
            if not isinstance(node, dict): return None
            
            tag = node.get("tagName", "").lower()
            if tag in ["script", "style", "meta", "link", "svg"]:
                return None
            
            # Önemli özellikler
            pruned: Dict[str, Any] = {
                "tag": tag,
                "id": node.get("id"),
                "text": node.get("innerText", "")[:100],
                "rect": node.get("rect"), # (x, y, width, height)
                "children": []
            }
            
            children_list: List[Dict[str, Any]] = []
            for child in node.get("children", []):
                 processed_child = _process(child)
                 if processed_child:
                     children_list.append(processed_child)
            
            pruned["children"] = children_list
            return pruned

        return _process(raw_dom) or {}

class VisualAligner:
    """
    VisualAligner: Pruned DOM içindeki elemanları screenshot üzerindeki 
    koordinatları ile eşleştirir.
    """
    @staticmethod
    def get_element_at(pruned_dom: Dict[str, Any], x: int, y: int) -> Optional[Dict[str, Any]]:
        """Belirli bir koordinattaki en derin (deepest) elemanı bulur."""
        best_match = None
        
        def _search(node):
            nonlocal best_match
            rect = node.get("rect")
            if not rect: return
            
            nx, ny, nw, nh = rect.get('x', 0), rect.get('y', 0), rect.get('width', 0), rect.get('height', 0)
            if nx <= x <= nx + nw and ny <= y <= ny + nh:
                best_match = node
                for child in node.get("children", []):
                    _search(child)
        
        _search(pruned_dom)
        return best_match
