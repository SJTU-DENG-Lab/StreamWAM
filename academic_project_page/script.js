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

document.querySelectorAll("[data-tabs]").forEach((tabsRoot) => {
  const tabList = tabsRoot.querySelector(".tab-list");
  const tabs = [...tabsRoot.querySelectorAll("[data-tab]")];
  const panels = [...tabsRoot.querySelectorAll("[data-panel]")];

  tabList.setAttribute("role", "tablist");
  tabs.forEach((tab, index) => {
    tab.setAttribute("role", "tab");
    tab.setAttribute("aria-controls", `panel-${tab.dataset.tab}`);
    tab.setAttribute("aria-selected", String(index === 0));
  });
  panels.forEach((panel) => {
    panel.setAttribute("role", "tabpanel");
    panel.setAttribute("aria-labelledby", `tab-${panel.dataset.panel}`);
  });

  function activateTab(selectedTab, moveFocus = true) {
    tabs.forEach((tab) => {
      const selected = tab === selectedTab;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
    });
    panels.forEach((panel) => {
      panel.hidden = panel.dataset.panel !== selectedTab.dataset.tab;
    });
    if (moveFocus) selectedTab.focus();
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", (event) => {
      event.preventDefault();
      activateTab(tab, false);
    });
    tab.addEventListener("keydown", (event) => {
      let targetIndex;
      if (event.key === "ArrowRight") targetIndex = (index + 1) % tabs.length;
      if (event.key === "ArrowLeft") targetIndex = (index - 1 + tabs.length) % tabs.length;
      if (event.key === "Home") targetIndex = 0;
      if (event.key === "End") targetIndex = tabs.length - 1;
      if (targetIndex === undefined) return;
      event.preventDefault();
      activateTab(tabs[targetIndex]);
    });
  });

  activateTab(tabs.find((tab) => tab.getAttribute("aria-selected") === "true") || tabs[0], false);
});
