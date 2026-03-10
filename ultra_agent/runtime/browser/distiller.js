(() => {
    // Playwright locator resolver'ı (P6) için semantik element çıkarımı (Shadow DOM + iframe desteği)
    const interactiveTags = ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A'];
    const interactiveRoles = ['button', 'checkbox', 'combobox', 'gridcell', 'link', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'option', 'radio', 'searchbox', 'slider', 'spinbutton', 'switch', 'tab', 'textbox', 'treeitem'];

    const oldOverlay = document.getElementById('bio-ml-overlay');
    if (oldOverlay) oldOverlay.remove();

    const overlay = document.createElement('div');
    overlay.id = 'bio-ml-overlay';
    Object.assign(overlay.style, {
        position: 'absolute', top: '0', left: '0',
        width: '100vw', height: '100vh', pointerEvents: 'none',
        zIndex: '2147483647'
    });
    document.body.appendChild(overlay);

    function isVisible(el) {
        if (!el.offsetParent && el.tagName !== 'BODY') {
            // Shadow DOM içindeki elementlerin offsetParent'ı olmayabilir, bu yüzden boundingBox kontrolü şart
            const rect = el.getBoundingClientRect();
            if (rect.width === 0 || rect.height === 0) return false;
        }
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return false;
        return true;
    }

    function isDisabled(el) {
        return el.disabled || el.getAttribute('aria-disabled') === 'true' || el.classList.contains('disabled');
    }

    function isInteractive(el) {
        if (isDisabled(el)) return false;
        if (interactiveTags.includes(el.tagName)) return true;
        const role = el.getAttribute('role');
        if (role && interactiveRoles.includes(role.toLowerCase())) return true;
        if (el.hasAttribute('onclick') || el.getAttribute('tabindex') === "0") return true;
        return false;
    }

    function getAccessibleName(el) {
        let name = el.getAttribute('aria-label');
        if (name) return name.trim();
        if (el.id) {
            const labelEl = document.querySelector(`label[for="${el.id}"]`);
            if (labelEl) return labelEl.innerText.trim();
        }
        const parentLabel = el.closest('label');
        if (parentLabel && parentLabel.innerText) return parentLabel.innerText.trim();
        name = el.placeholder || el.title || el.getAttribute('alt') || el.getAttribute('name');
        if (name) return name.trim();
        return (el.innerText || '').trim().split('\n')[0].substring(0, 80);
    }

    function getSemanticRole(el) {
        let explicitRole = el.getAttribute('role');
        if (explicitRole) return explicitRole.toLowerCase();
        const tag = el.tagName.toLowerCase();
        if (tag === 'a') return 'link';
        if (tag === 'button') return 'button';
        if (tag === 'textarea') return 'textbox';
        if (tag === 'select') return 'combobox';
        if (tag === 'input') {
            const type = el.type.toLowerCase();
            if (['button', 'submit', 'reset'].includes(type)) return 'button';
            if (type === 'checkbox') return 'checkbox';
            if (type === 'radio') return 'radio';
            if (['text', 'email', 'password', 'search', 'url', 'number', 'tel'].includes(type)) return 'textbox';
        }
        return tag;
    }

    let bioCount = 0;
    const elements = [];

    // Rekürsif tarama fonksiyonu (Shadow DOM ve iframe desteği)
    function collectInteractiveElements(root, offsetTop = 0, offsetLeft = 0) {
        const walker = document.createTreeWalker(root, NodeFilter.SHOW_ELEMENT, null, false);
        let node = walker.nextNode();
        while (node) {
            // Shadow DOM kontrolü
            if (node.shadowRoot) {
                collectInteractiveElements(node.shadowRoot, offsetTop, offsetLeft);
            }

            // Iframe kontrolü (sadece erişilebilirse)
            if (node.tagName === 'IFRAME') {
                try {
                    const rect = node.getBoundingClientRect();
                    collectInteractiveElements(node.contentDocument.body, offsetTop + rect.top, offsetLeft + rect.left);
                } catch (e) {
                    // Cross-origin iframe'lere erişilemez, normaldir.
                }
            }

            if (isVisible(node) && isInteractive(node)) {
                const bioId = `bio-${++bioCount}`;
                node.setAttribute('data-bio-id', bioId);

                const rect = node.getBoundingClientRect();
                const tag = document.createElement('div');
                tag.innerText = bioId;
                Object.assign(tag.style, {
                    position: 'absolute',
                    top: `${rect.top + window.scrollY + offsetTop}px`,
                    left: `${rect.left + window.scrollX + offsetLeft}px`,
                    backgroundColor: 'rgba(255, 0, 0, 0.8)', color: 'white', padding: '1px 3px',
                    borderRadius: '2px', fontSize: '9px', fontWeight: 'bold', zIndex: '2147483647',
                    pointerEvents: 'none', lineHeight: '10px'
                });
                overlay.appendChild(tag);

                elements.push({
                    bio_id: bioId,
                    pw_role: getSemanticRole(node),
                    pw_name: getAccessibleName(node),
                    tag: node.tagName.toLowerCase(),
                    text: (node.innerText || '').trim().substring(0, 100),
                    css_selector: `${node.tagName.toLowerCase()}[data-bio-id="${bioId}"]`,
                    type: node.type || '',
                    is_disabled: isDisabled(node)
                });
            }
            node = walker.nextNode();
        }
    }

    collectInteractiveElements(document.body);

    return {
        page: { title: document.title, url: window.location.href },
        interactive: elements
    };
})();
