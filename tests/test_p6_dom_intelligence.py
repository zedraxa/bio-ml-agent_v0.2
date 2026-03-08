"""
Phase 15 (P6) — DOM Intelligence Mock-Based Verification
LocatorResolver ve ActionValidator sınıflarını test eder.
"""

import sys
import logging
from unittest.mock import MagicMock

# Proje kökünü ayarla
from pathlib import Path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

logging.basicConfig(level=logging.INFO, stream=sys.stdout)

PASS = 0
FAIL = 0

def ok(name: str, cond: bool, detail: str = ""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ✅ {name}", flush=True)
    else:
        FAIL += 1
        print(f"  ❌ {name} — {detail}", flush=True)

def test_locator_resolver():
    print("\n=== Test 1: LocatorResolver ===", flush=True)
    from ultra_agent.runtime.browser.dom_intelligence import LocatorResolver

    mock_page = MagicMock()
    
    # Mock locators
    role_loc = MagicMock()
    role_loc.count.return_value = 1
    
    text_loc = MagicMock()
    text_loc.count.return_value = 1
    
    css_loc = MagicMock()

    # Mock page methods
    def mock_get_by_role(role, name=None, exact=False):
        if role == "button" and name == "Submit": return role_loc
        empty = MagicMock(); empty.count.return_value = 0
        return empty
        
    def mock_get_by_text(text, exact=False):
        if text == "Click Me": return text_loc
        empty = MagicMock(); empty.count.return_value = 0
        return empty

    mock_page.get_by_role = mock_get_by_role
    mock_page.get_by_text = mock_get_by_text
    mock_page.locator.return_value.first = css_loc

    perception = {
        "interactive": [
            {"bio_id": "bio-1", "pw_role": "button", "pw_name": "Submit", "text": "Submit"},
            {"bio_id": "bio-2", "pw_role": "div", "pw_name": "", "text": "Click Me"},
            {"bio_id": "bio-3", "pw_role": "", "pw_name": "", "text": ""},
        ]
    }

    resolver = LocatorResolver(mock_page, perception)

    # 1. Semantic resolve (Role + Name)
    loc1 = resolver.resolve("bio-1")
    ok("Role + Name matched", loc1 == role_loc.first)

    # 2. Text resolve
    loc2 = resolver.resolve("bio-2")
    ok("Text matched", loc2 == text_loc.first)

    # 3. Fallback CSS resolve
    loc3 = resolver.resolve("bio-3")
    ok("Fallback CSS matched", loc3 == css_loc)

    # 4. Unknown bio_id
    loc4 = resolver.resolve("unknown-99")
    ok("Unknown uses fallback", loc4 == css_loc)


def test_action_validator():
    print("\n=== Test 2: ActionValidator ===", flush=True)
    from ultra_agent.runtime.browser.dom_intelligence import ActionValidator

    # Senaryo 1: Geçerli butona tıklama
    loc_valid = MagicMock()
    loc_valid.count.return_value = 1
    loc_valid.first.is_visible.return_value = True
    loc_valid.first.is_disabled.return_value = False
    
    val1 = ActionValidator(loc_valid, "click")
    is_valid, reason = val1.validate()
    ok("Valid click accepted", is_valid is True)

    # Senaryo 2: Görünmez buton
    loc_hidden = MagicMock()
    loc_hidden.count.return_value = 1
    loc_hidden.first.is_visible.return_value = False
    
    val2 = ActionValidator(loc_hidden, "click")
    is_valid, reason = val2.validate()
    ok("Hidden element rejected", is_valid is False and "görünür değil" in reason)

    # Senaryo 3: Disabled input fill
    loc_disabled = MagicMock()
    loc_disabled.count.return_value = 1
    loc_disabled.first.is_visible.return_value = True
    loc_disabled.first.is_disabled.return_value = True
    
    val3 = ActionValidator(loc_disabled, "fill")
    is_valid, reason = val3.validate()
    ok("Disabled element rejected", is_valid is False and "kullanılamaz durumda" in reason)

    # Senaryo 4: Yalnızca okuma (extract_text) - görünürlük aranmaz (count > 0 yeter)
    val4 = ActionValidator(loc_hidden, "extract_text")
    is_valid, reason = val4.validate()
    ok("extract_text doesn't strictly check visibility", is_valid is True)


if __name__ == "__main__":
    print("=" * 60)
    print("  Phase 15 (P6) — DOM Intelligence Mock Verification")
    print("=" * 60)

    test_locator_resolver()
    test_action_validator()

    print(f"\n{'=' * 60}")
    print(f"  Sonuç: {PASS} geçti, {FAIL} başarısız")
    print(f"{'=' * 60}")

    sys.exit(0 if FAIL == 0 else 1)
