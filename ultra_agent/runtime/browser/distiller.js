(() => {
    // Playwright locator resolver'ı (P6) için semantic element çıkarımı

    // Hedeflediğimiz etkileşimli roller ve taglar
    const interactiveTags = ['BUTTON', 'INPUT', 'SELECT', 'TEXTAREA', 'A'];
    // Semantic HTML rollerine daha uygun bir liste
    const interactiveRoles = ['button', 'checkbox', 'combobox', 'gridcell', 'link', 'menuitem', 'menuitemcheckbox', 'menuitemradio', 'option', 'radio', 'searchbox', 'slider', 'spinbutton', 'switch', 'tab', 'textbox', 'treeitem'];

    // Mevcut overlay'i temizle
    const oldOverlay = document.getElementById('bio-ml-overlay');
    if (oldOverlay) oldOverlay.remove();

    const overlay = document.createElement('div');
    overlay.id = 'bio-ml-overlay';
    // Overlay stilleri
    Object.assign(overlay.style, {
        position: 'absolute', top: '0', left: '0',
        width: '100vw', height: '100vh', pointerEvents: 'none',
        zIndex: '2147483647'
    });
    document.body.appendChild(overlay);

    function isVisible(el) {
        if (!el.offsetParent && el.tagName !== 'BODY') return false;
        const style = window.getComputedStyle(el);
        if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;

        // P6: Bounding rect kontrolleri daha sağlam
        const rect = el.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return false;

        // Ekran dışında mı? (isteğe bağlı ama overlay için faydalı, click için PW kendi bekler)
        return true;
    }

    function isDisabled(el) {
        return el.disabled || el.getAttribute('aria-disabled') === 'true' || el.classList.contains('disabled');
    }

    function isInteractive(el) {
        if (isDisabled(el)) return false; // Disabled elementleri geç

        if (interactiveTags.includes(el.tagName)) return true;

        const role = el.getAttribute('role');
        if (role && interactiveRoles.includes(role.toLowerCase())) return true;

        // P6: Sadece onClick yetmez, cursor:pointer da bir hint ama çok div var
        if (el.hasAttribute('onclick') || el.getAttribute('tabindex') === "0") return true;

        return false;
    }

    // Playwright uyumlu get_by_role('role', name='label') için label çıkarımı
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

    // Playwright Role uydurma
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
            if (type === 'search') return 'searchbox';
        }
        return tag; // fallback
    }

    let bioCount = 0;
    const elements = [];
    const allElements = document.querySelectorAll('*');

    for (const el of allElements) {
        if (isVisible(el) && isInteractive(el)) {
            const bioId = `bio-${++bioCount}`;
            el.setAttribute('data-bio-id', bioId);

            // Overlay çizimi
            const rect = el.getBoundingClientRect();
            const tag = document.createElement('div');
            tag.innerText = bioId;
            Object.assign(tag.style, {
                position: 'absolute', top: `${rect.top + window.scrollY}px`, left: `${rect.left + window.scrollX}px`,
                backgroundColor: 'rgba(255, 0, 0, 0.8)', color: 'white', padding: '1px 3px',
                borderRadius: '2px', fontSize: '9px', fontWeight: 'bold', zIndex: '2147483647',
                pointerEvents: 'none', lineHeight: '10px'
            });
            overlay.appendChild(tag);

            // Resolver'a gidecek zengin veri payload'ı
            elements.push({
                bio_id: bioId,
                pw_role: getSemanticRole(el),
                pw_name: getAccessibleName(el),
                tag: el.tagName.toLowerCase(),
                text: (el.innerText || '').trim().substring(0, 100),
                css_selector: `${el.tagName.toLowerCase()}[data-bio-id="${bioId}"]`,
                type: el.type || '',
                is_disabled: isDisabled(el)
            });
        }
    }

    return {
        page: { title: document.title, url: window.location.href },
        interactive: elements
    };
})();
