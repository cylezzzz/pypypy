(function () {
  async function inject() {
    const nodes = document.querySelectorAll("[data-include]");
    await Promise.all([...nodes].map(async (el) => {
      const name = el.getAttribute("data-include");
      const resp = await fetch(`partials/${name}.html`, { cache: "no-store" });
      el.outerHTML = await resp.text();
    }));
    highlightActive();
  }

  function highlightActive() {
    const file = (location.pathname.split("/").pop() || "index.html").toLowerCase();
    const link = document.querySelector(`a[href="${file}"]`);
    if (link) {
      link.classList.add("active");
      link.setAttribute("aria-current", "page");
    }
  }

  document.addEventListener("DOMContentLoaded", inject);
})();
