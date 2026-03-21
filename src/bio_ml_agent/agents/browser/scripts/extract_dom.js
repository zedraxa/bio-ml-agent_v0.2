/**
 * Advanced DOM Extractor for BrowserScout (Hardened).
 * Features:
 * - Recursive Shadow DOM traversal.
 * - Iframe and Frame support (placeholder for cross-origin iframes, handles same-origin).
 * - Bounding box extraction with scroll offsets.
 */
(() => {
    function getElementTree(element) {
        if (!element) return null;
        
        let root = element;
        // If element is a host for shadow DOM, we might want to traverse its shadowRoot
        // but for the tree structure, we usually want both the shadow nodes and regular children.
        
        const rect = element.getBoundingClientRect();
        if (rect.width === 0 || rect.height === 0) return null;

        const info = {
            tagName: element.tagName,
            id: element.id,
            className: element.className,
            innerText: (element.children.length === 0) ? (element.innerText || "").trim() : "",
            rect: {
                x: rect.x + window.scrollX,
                y: rect.y + window.scrollY,
                width: rect.width,
                height: rect.height
            },
            children: []
        };

        // 1. Traverse regular children
        for (let child of element.children) {
            const childInfo = getElementTree(child);
            if (childInfo) info.children.push(childInfo);
        }

        // 2. Traverse Shadow DOM
        if (element.shadowRoot) {
            for (let child of element.shadowRoot.children) {
                const childInfo = getElementTree(child);
                if (childInfo) {
                    if (!childInfo.metadata) childInfo.metadata = {};
                    childInfo.metadata.isShadow = true;
                    info.children.push(childInfo);
                }
            }
        }

        // 3. Traverse Iframes (Same-origin only)
        if (element.tagName === 'IFRAME' || element.tagName === 'FRAME') {
            try {
                const frameDoc = element.contentDocument || element.contentWindow.document;
                if (frameDoc && frameDoc.body) {
                    const frameTree = getElementTree(frameDoc.body);
                    if (frameTree) {
                        frameTree.metadata = { isFrame: true };
                        info.children.push(frameTree);
                    }
                }
            } catch (e) {
                // Ignore cross-origin errors
                info.metadata = { crossOriginFrame: true };
            }
        }

        return info;
    }

    return getElementTree(document.body);
})();
