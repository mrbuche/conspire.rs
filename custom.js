(() => {
    const url = location.pathname.includes("/latest/")
        ? "https://mrbuche.github.io/conspire.rs/latest"
        : "https://docs.rs/conspire";
    const rewrite = (root) =>
        root.querySelectorAll('a[href$="rust.html"]').forEach((a) => {
            a.href = url;
            a.target = "_blank";
        });
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
