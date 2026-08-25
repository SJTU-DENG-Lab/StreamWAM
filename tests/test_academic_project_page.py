from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_ROOT = REPO_ROOT / "academic_project_page"
INDEX_PATH = PAGE_ROOT / "index.html"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pages.yml"


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[tuple[str, dict[str, str]]] = []
        self.ids: set[str] = set()
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = {name: value or "" for name, value in attrs}
        self.tags.append(tag)
        self.attributes.append((tag, normalized))
        if element_id := normalized.get("id"):
            self.ids.add(element_id)

    def handle_data(self, data: str) -> None:
        if stripped := data.strip():
            self.text_parts.append(stripped)


def parse_page() -> tuple[PageParser, str]:
    html = INDEX_PATH.read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(html)
    return parser, html


def test_page_exposes_the_research_preview_and_available_artifacts() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)

    assert parser.tags.count("header") == 1
    assert parser.tags.count("main") == 1
    assert parser.tags.count("footer") == 1
    assert {"overview", "method", "results", "gallery", "resources"} <= parser.ids
    assert "Think ahead. Act now." in visible_text
    assert "Streaming World-Action Models for Robotic Manipulation" in visible_text
    assert "action-conditioned" in visible_text.lower()
    assert "Coming Soon" in visible_text

    links = {
        attrs.get("href")
        for tag, attrs in parser.attributes
        if tag == "a" and attrs.get("href")
    }
    assert "https://github.com/SJTU-DENG-Lab/StreamWAM" in links
    assert "https://huggingface.co/SJTU-DENG-Lab/StreamWAM" in links
    assert "#" not in links
    assert "RTC-AC" not in html
    assert "AC-StreamWAM" not in html
    assert "Ours" not in html


def test_page_publishes_the_current_benchmark_results() -> None:
    parser, _ = parse_page()
    visible_text = " ".join(parser.text_parts)

    expected_fragments = (
        "98.20%",
        "41.0 ms",
        "11.76 s",
        "96.60",
        "98.80",
        "97.40",
        "100.00",
        "75.35%",
        "136.76 ms",
        "87.2",
        "88.8",
        "87.6",
        "50 target tasks",
        "100 rollout episodes per task",
    )
    for fragment in expected_fragments:
        assert fragment in visible_text


def test_every_local_page_reference_resolves() -> None:
    parser, _ = parse_page()
    referenced_files: set[Path] = set()

    for _, attrs in parser.attributes:
        for name in ("href", "src", "poster"):
            value = attrs.get(name, "")
            if not value or value.startswith(("http://", "https://", "mailto:", "data:")):
                continue
            if value.startswith("#"):
                assert value[1:] in parser.ids
                continue
            parsed = urlparse(value)
            referenced_files.add(PAGE_ROOT / parsed.path)

    assert referenced_files
    missing = sorted(str(path.relative_to(REPO_ROOT)) for path in referenced_files if not path.is_file())
    assert missing == []


def test_pages_workflow_deploys_only_the_project_page() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "contents: read" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "actions/configure-pages@v5" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert "path: ./academic_project_page" in workflow
    assert "path: ./" not in {line.strip() for line in workflow.splitlines()}
