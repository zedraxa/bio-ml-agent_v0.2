/**
 * Advanced DOM Extractor for BrowserScout.
 * Captures tagName, ID, classes, innerText, and bounding boxes.
 */
(() => {
    function getElementTree(element) {
        if (!element) return null;
        const rect = element.getBoundingClientRect();
        
        // Skip invisible elements
        if (rect.width === 0 || rect.height === 0) return null;

        const info = {
            tagName: element.tagName,
            id: element.id,
            className: element.className,
            innerText: (element.children.length === 0) ? element.innerText : "",
            rect: {
                x: rect.x + window.scrollX,
                y: rect.y + window.scrollY,
                width: rect.width,
                height: rect.height
            },
            children: []
        };

        // Limited recursion depth for performance
        for (let child of element.children) {
            const childInfo = getElementTree(child);
            if (childInfo) info.children.push(childInfo);
        }

        return info;
    }

    return getElementTree(document.body);
})();
