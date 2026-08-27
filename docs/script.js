document.documentElement.classList.add("js");

const menuButton = document.querySelector(".menu-toggle");
const menu = document.querySelector(".nav-links");

function closeMenu() {
  if (!menuButton || !menu) return;
  menuButton.setAttribute("aria-expanded", "false");
  menu.classList.remove("is-open");
}

if (menuButton && menu) {
  menuButton.hidden = false;
  menuButton.addEventListener("click", () => {
    const nextOpen = menuButton.getAttribute("aria-expanded") !== "true";
    menuButton.setAttribute("aria-expanded", String(nextOpen));
    menu.classList.toggle("is-open", nextOpen);
  });

  menu.addEventListener("click", (event) => {
    if (event.target.closest("a")) closeMenu();
  });

  document.addEventListener("click", (event) => {
    if (!menu.contains(event.target) && !menuButton.contains(event.target)) closeMenu();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && menuButton.getAttribute("aria-expanded") === "true") {
      closeMenu();
      menuButton.focus();
    }
  });
}

const citationCopyButton = document.querySelector(".citation-copy");
const citationBibtex = document.querySelector("#citation-bibtex");

function copyWithTextarea(text) {
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  textarea.style.position = "fixed";
  textarea.style.opacity = "0";
  document.body.appendChild(textarea);
  textarea.select();

  try {
    return document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }
}

if (citationCopyButton && citationBibtex) {
  citationCopyButton.addEventListener("click", async () => {
    const citation = citationBibtex.textContent.trim();
    let copied = false;

    if (navigator.clipboard?.writeText) {
      try {
        await navigator.clipboard.writeText(citation);
        copied = true;
      } catch {
        copied = false;
      }
    }

    if (!copied) {
      try {
        copied = copyWithTextarea(citation);
      } catch {
        copied = false;
      }
    }

    if (!copied) {
      citationCopyButton.textContent = "Select text";
      return;
    }

    citationCopyButton.textContent = "Copied";
    setTimeout(() => {
      citationCopyButton.textContent = "Copy";
    }, 1800);
  });
}

const articleToc = document.querySelector(".article-toc");
const articleTocNav = articleToc?.querySelector("nav");
const tocOccluders = document.querySelectorAll("#streamwam-method-figure, .benchmark, .latency-figure");

if (articleToc && articleTocNav && tocOccluders.length) {
  let tocUpdatePending = false;

  function updateTocVisibility() {
    const tocRect = articleTocNav.getBoundingClientRect();
    const isObscured = Array.from(tocOccluders).some((element) => {
      const rect = element.getBoundingClientRect();
      return rect.left < tocRect.right && rect.right > tocRect.left && rect.top < tocRect.bottom && rect.bottom > tocRect.top;
    });

    articleToc.classList.toggle("is-obscured", isObscured);
    articleTocNav.toggleAttribute("inert", isObscured);
    articleTocNav.setAttribute("aria-hidden", String(isObscured));
    tocUpdatePending = false;
  }

  function scheduleTocVisibilityUpdate() {
    if (tocUpdatePending) return;
    tocUpdatePending = true;
    requestAnimationFrame(updateTocVisibility);
  }

  window.addEventListener("scroll", scheduleTocVisibilityUpdate, { passive: true });
  window.addEventListener("resize", scheduleTocVisibilityUpdate);
  updateTocVisibility();
  articleToc.classList.add("is-ready");
}

document.querySelectorAll(".rollout-demo-card").forEach((card) => {
  const video = card.querySelector(".real-robot-video");
  const source = video?.querySelector("source[data-src]");
  const loadButton = card.querySelector(".real-robot-load");

  if (!video || !source || !loadButton) return;

  loadButton.addEventListener("click", () => {
    if (!source.src) {
      source.src = source.dataset.src;
      video.load();
    }

    loadButton.hidden = true;
    card.classList.add("is-loaded");
    const playback = video.play();
    if (playback?.catch) playback.catch(() => {});
  });
});
