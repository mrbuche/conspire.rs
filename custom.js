(() => {
    const urls = {
        "rust.html": "https://docs.rs/conspire",
        "rust_latest.html": "https://mrbuche.github.io/conspire.rs/latest/",
    };
    const rewrite = (root) =>
        Object.entries(urls).forEach(([page, url]) =>
            root.querySelectorAll(`a[href$="${page}"]`).forEach((a) => {
                if (a.getAttribute("href").split("/").pop() !== page) return;
                a.href = url;
                a.target = "_blank";
            }),
        );
    const sidebar = (frame) => {
        const doc = frame.contentDocument;
        if (!doc) return;
        rewrite(doc);
        new MutationObserver(() => rewrite(doc)).observe(doc, {
            childList: true,
            subtree: true,
        });
    };
    document.addEventListener("DOMContentLoaded", () => {
        rewrite(document);
        document.querySelectorAll("iframe.sidebar-iframe-outer").forEach((frame) => {
            frame.addEventListener("load", () => sidebar(frame));
            if (frame.contentDocument?.readyState === "complete") sidebar(frame);
        });
    });
})();
