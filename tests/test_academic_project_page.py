from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_ROOT = REPO_ROOT / "academic_project_page"
INDEX_PATH = PAGE_ROOT / "index.html"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pages.yml"
ARTICLE_SECTION_IDS = (
    "act-wam",
    "act-async",
    "act-streamwam",
    "experiments",
    "discussion",
)


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.attributes: list[tuple[str, dict[str, str]]] = []
        self.ids: set[str] = set()
        self.text_parts: list[str] = []
        self.article_depth = 0
        self.article_paragraph_count = 0
        self.pre_experiment_paragraph_count = 0
        self.before_experiments = True
        self.section_stack: list[str] = []
        self.act_heading_tags: list[str] = []
        self.act_body_paragraph_count = 0
        self.act_body_paragraph_text: list[str] | None = None
        self.current_h3_id: str | None = None
        self.current_h3_text: list[str] = []
        self.h3_text_by_id: dict[str, str] = {}
        self.hero_title_text: list[str] | None = None
        self.hero_title = ""
        self.caption_text: list[str] | None = None
        self.captions: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = {name: value or "" for name, value in attrs}
        self.tags.append(tag)
        self.attributes.append((tag, normalized))
        if element_id := normalized.get("id"):
            self.ids.add(element_id)
        if tag == "article":
            self.article_depth += 1
        if tag == "section":
            section_id = normalized.get("id", "")
            self.section_stack.append(section_id)
            if section_id == "experiments":
                self.before_experiments = False
        if tag == "p" and self.article_depth:
            self.article_paragraph_count += 1
            if self.before_experiments:
                self.pre_experiment_paragraph_count += 1
            classes = normalized.get("class", "").split()
            if any(section_id.startswith("act-") for section_id in self.section_stack):
                if "act-label" not in classes and "act-opening" not in classes:
                    self.act_body_paragraph_text = []
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"} and any(
            section_id.startswith("act-") for section_id in self.section_stack
        ):
            self.act_heading_tags.append(tag)
        if tag == "h3" and normalized.get("id"):
            self.current_h3_id = normalized["id"]
            self.current_h3_text = []
        if tag == "h1" and normalized.get("id") == "hero-title":
            self.hero_title_text = []
        if tag == "caption":
            self.caption_text = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "article":
            self.article_depth -= 1
        if tag == "p" and self.act_body_paragraph_text is not None:
            if " ".join(self.act_body_paragraph_text).strip():
                self.act_body_paragraph_count += 1
            self.act_body_paragraph_text = None
        if tag == "h3" and self.current_h3_id is not None:
            self.h3_text_by_id[self.current_h3_id] = " ".join(self.current_h3_text).strip()
            self.current_h3_id = None
            self.current_h3_text = []
        if tag == "h1" and self.hero_title_text is not None:
            self.hero_title = " ".join(self.hero_title_text).strip()
            self.hero_title_text = None
        if tag == "caption" and self.caption_text is not None:
            self.captions.append(" ".join(self.caption_text).strip())
            self.caption_text = None
        if tag == "section":
            self.section_stack.pop()

    def handle_data(self, data: str) -> None:
        if self.act_body_paragraph_text is not None:
            self.act_body_paragraph_text.append(data)
        if self.current_h3_id is not None:
            self.current_h3_text.append(data)
        if self.hero_title_text is not None:
            self.hero_title_text.append(data)
        if self.caption_text is not None:
            self.caption_text.append(data)
        if stripped := data.strip():
            self.text_parts.append(stripped)


def parse_page() -> tuple[PageParser, str]:
    html = INDEX_PATH.read_text(encoding="utf-8")
    parser = PageParser()
    parser.feed(html)
    return parser, html


class BenchmarkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.current_benchmark: str | None = None
        self.current_cell: list[str] | None = None
        self.current_row: list[str] | None = None
        self.rows: dict[str, list[list[str]]] = {}
        self.section_text: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = {name: value or "" for name, value in attrs}
        if tag == "section" and normalized.get("id", "").startswith("benchmark-"):
            self.current_benchmark = normalized["id"]
            self.rows[self.current_benchmark] = []
            self.section_text[self.current_benchmark] = []
        elif self.current_benchmark and tag == "tr":
            self.current_row = []
        elif self.current_row is not None and tag in {"th", "td"}:
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_benchmark and data.strip():
            self.section_text[self.current_benchmark].append(data.strip())
        if self.current_cell is not None and data.strip():
            self.current_cell.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.current_cell is not None and self.current_row is not None:
            self.current_row.append(" ".join(self.current_cell))
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None and self.current_benchmark:
            self.rows[self.current_benchmark].append(self.current_row)
            self.current_row = None
        elif tag == "section" and self.current_benchmark:
            self.current_benchmark = None


def parse_benchmarks() -> BenchmarkParser:
    parser = BenchmarkParser()
    parser.feed(INDEX_PATH.read_text(encoding="utf-8"))
    return parser


def test_page_exposes_the_research_preview_and_available_artifacts() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)

    assert parser.tags.count("header") == 1
    assert parser.tags.count("main") == 1
    assert parser.tags.count("footer") == 1
    assert {
        "overview",
        "act-wam",
        "act-async",
        "act-streamwam",
        "experiments",
        "discussion",
        "resources",
    } <= parser.ids
    assert parser.hero_title == "StreamWAM: Streaming World-Action Models for Robotic Manipulation"
    assert "Think ahead. Act now." not in visible_text
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
        "190.17 ms",
        "81.21 ms",
        "47.09 ms",
        "110.22 s",
        "102.59 s",
        "77.48 s",
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


def test_no_javascript_navigation_remains_usable() -> None:
    parser, html = parse_page()
    menu_buttons = [attrs for tag, attrs in parser.attributes if tag == "button" and "menu-toggle" in attrs.get("class", "")]
    assert len(menu_buttons) == 1
    assert "hidden" in menu_buttons[0]
    assert "data-tabs" not in html
    assert "data-panel" not in html


def test_small_dim_text_meets_wcag_aa_contrast() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    colors = dict(re.findall(r"--([\w-]+):\s*(#[0-9a-fA-F]{6})", css))

    def luminance(hex_color: str) -> float:
        channels = [int(hex_color[index : index + 2], 16) / 255 for index in (1, 3, 5)]
        linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
        return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]

    for foreground_name, background_name in (
        ("dim", "bg"),
        ("dim", "paper"),
        ("orange", "orange-soft"),
        ("teal-deep", "teal-soft"),
        ("teal-deep", "results-bg"),
        ("teal-deep", "resources-bg"),
    ):
        foreground = luminance(colors[foreground_name])
        background = luminance(colors[background_name])
        contrast = (max(foreground, background) + 0.05) / (min(foreground, background) + 0.05)
        assert contrast >= 4.5, f"{foreground_name} on {background_name}: {contrast:.2f}"


def test_social_metadata_uses_absolute_project_urls() -> None:
    parser, _ = parse_page()
    metadata = {
        attrs.get("property"): attrs.get("content")
        for tag, attrs in parser.attributes
        if tag == "meta" and attrs.get("property")
    }

    assert metadata["og:url"] == "https://sjtu-deng-lab.github.io/StreamWAM/"
    assert metadata["og:image"] == "https://sjtu-deng-lab.github.io/StreamWAM/assets/streamwam-social-preview.jpg"


def test_page_exposes_a_complete_research_story_without_draft_placeholders() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)
    theme_colors = [
        attrs.get("content")
        for tag, attrs in parser.attributes
        if tag == "meta" and attrs.get("name") == "theme-color"
    ]

    assert "Why capable world-action models still make robots wait" in visible_text
    assert "Asynchrony removes the wait, but exposes a boundary" in visible_text
    assert "StreamWAM conditions the visual future on the action already underway" in visible_text
    assert "inference-time RTC" in visible_text
    assert "prefix-conditioned" in visible_text
    assert "00 · Abstract" in visible_text
    assert "Task performance." in visible_text
    assert "Inference efficiency." in visible_text
    assert "StreamWAM reaches 98.20% average success" in visible_text
    assert "StreamWAM reports 75.35% average task success" in visible_text
    assert "StreamWAM reaches 87.6 total success" in visible_text
    assert "broader analysis, limitations, and failure cases" in visible_text
    assert "Model lineage." in visible_text
    assert "Code and models are available now" in visible_text
    assert "Successful LIBERO rollout frames." in visible_text
    assert theme_colors == ["#f7f5ef"]
    assert not any(token in html.casefold() for token in ("todo", "tbd", "lorem ipsum"))


def test_hidden_mobile_menu_remains_hidden_without_javascript() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")

    assert ".menu-toggle[hidden]{display:none!important}" in css


def test_page_is_a_linear_text_first_research_article() -> None:
    parser, html = parse_page()
    article_ids = [
        attrs["id"]
        for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id") in ARTICLE_SECTION_IDS
    ]

    assert parser.tags.count("article") >= 1
    assert article_ids == list(ARTICLE_SECTION_IDS)
    assert parser.article_paragraph_count >= 40
    assert "chapter-index" not in html
    assert "future-slots" not in html


def test_article_opens_with_three_detailed_editorial_acts() -> None:
    parser, _ = parse_page()
    act_sections = [
        attrs
        for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id", "").startswith("act-")
    ]
    links = {
        attrs.get("href")
        for tag, attrs in parser.attributes
        if tag == "a" and attrs.get("href")
    }

    assert [section["id"] for section in act_sections] == ["act-wam", "act-async", "act-streamwam"]
    assert all("editorial-act" in section.get("class", "").split() for section in act_sections)
    assert parser.act_body_paragraph_count >= 18
    assert parser.act_heading_tags == []
    assert [section.get("aria-label") for section in act_sections] == [
        "The deployment problem",
        "The asynchronous boundary",
        "Action-conditioned streaming",
    ]
    assert "https://arxiv.org/abs/2608.01880" in links


def test_article_uses_a_continuous_editorial_hierarchy() -> None:
    parser, _ = parse_page()
    visible_text = " ".join(parser.text_parts)
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    removed_display_copy = (
        "Research notes · August 2026",
        "01 · The deployment problem",
        "02 · The asynchronous boundary",
        "03 · Action-conditioned streaming",
        "04 · Evidence",
        "What the Current Results Show",
        "05 · Discussion",
        "Where This Leaves Us",
        "Read, run, and revisit.",
        "Benchmark 01",
        "Benchmark 02",
        "Benchmark 03",
    )
    section_labels = {
        attrs.get("id"): attrs.get("aria-label")
        for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id") in {"experiments", "discussion", "resources"}
    }
    benchmark_heading_ids = {
        attrs["id"]: attrs.get("aria-labelledby")
        for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id", "").startswith("benchmark-")
    }

    assert not any(copy in visible_text for copy in removed_display_copy)
    assert section_labels == {
        "experiments": "Current results",
        "discussion": "Discussion",
        "resources": "Project resources",
    }
    assert len(set(benchmark_heading_ids.values())) == 3
    assert {
        section_id: parser.h3_text_by_id[heading_id]
        for section_id, heading_id in benchmark_heading_ids.items()
        if heading_id is not None
    } == {
        "benchmark-libero": "LIBERO",
        "benchmark-robocasa": "RoboCasa",
        "benchmark-robotwin": "RoboTwin 2.0",
    }
    compact_heading_rule = re.search(r"\.benchmark-intro h3\s*\{([^}]*)\}", css)
    assert compact_heading_rule is not None
    assert re.search(r"font-size:\s*clamp\([^;]*1\.4rem\s*\)", compact_heading_rule.group(1))


def test_all_benchmark_tables_are_visible_without_tabs() -> None:
    parser, html = parse_page()
    benchmark_ids = {
        attrs["id"]
        for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id", "").startswith("benchmark-")
    }

    assert benchmark_ids == {"benchmark-libero", "benchmark-robocasa", "benchmark-robotwin"}
    assert parser.tags.count("table") == 3
    assert parser.captions == [
        "LIBERO success results",
        "RoboCasa target benchmark results",
        "RoboTwin 2.0 clean and randomized results",
    ]
    assert "data-tabs" not in html
    assert "data-panel" not in html
    assert 'role="tab"' not in html


def test_benchmark_tables_and_protocols_match_the_authoritative_results() -> None:
    parsed = parse_benchmarks()

    assert parsed.rows == {
        "benchmark-libero": [
            ["Method", "LIBERO-10", "Spatial", "Goal", "Object", "Average ↑"],
            ["FastWAM", "96.20", "96.20", "94.20", "96.20", "95.70"],
            ["FastWAM-Joint-CD", "97.20", "99.60", "98.60", "100.00", "98.85"],
            ["FastWAM-RTC", "58.40", "76.20", "77.00", "83.40", "73.75"],
            ["StreamWAM", "96.60", "98.80", "97.40", "100.00", "98.20"],
            ["w/o Action Conditioning", "94.40", "96.40", "96.60", "97.60", "96.25"],
            ["w/o Slot Encoder", "95.60", "98.40", "96.80", "99.80", "97.65"],
        ],
        "benchmark-robocasa": [
            ["Method", "Accuracy ↑"],
            ["X-WAM", "75.42%"],
            ["X-WAM-CD", "75.83%"],
            ["StreamWAM", "75.35%"],
        ],
        "benchmark-robotwin": [
            ["Method", "Clean ↑", "Random ↑", "Total ↑"],
            ["StarWAM-Joint", "84.8", "86.0", "85.4"],
            ["StarWAM-CD", "79.0", "79.2", "79.1"],
            ["StreamWAM", "87.2", "88.8", "87.6"],
        ],
    }

    section_text = {name: " ".join(parts) for name, parts in parsed.section_text.items()}
    assert all(fragment in section_text["benchmark-libero"] for fragment in ("four suites", "10 tasks per suite", "50 trials per task", "long and short tasks"))
    assert all(fragment in section_text["benchmark-robocasa"] for fragment in ("50 target tasks", "50 trials per task", "average task success"))
    assert all(fragment in section_text["benchmark-robotwin"] for fragment in ("50 tasks", "100 rollout episodes per task", "Clean", "Random", "domain-randomization"))


def test_latency_charts_match_the_authoritative_efficiency_results() -> None:
    parser, _ = parse_page()
    charts = [
        attrs
        for _, attrs in parser.attributes
        if "latency-chart" in attrs.get("class", "").split()
    ]
    bars = [
        (
            attrs.get("data-benchmark"),
            attrs.get("data-metric"),
            attrs.get("data-method"),
            attrs.get("data-series"),
            attrs.get("data-value"),
            attrs.get("data-unit"),
        )
        for _, attrs in parser.attributes
        if "latency-bar" in attrs.get("class", "").split()
    ]

    assert len(charts) == 6
    assert all(chart.get("role") == "group" for chart in charts)
    assert len({chart.get("aria-labelledby") for chart in charts}) == 6
    assert all(chart.get("aria-labelledby") in parser.ids for chart in charts)
    assert bars == [
        ("libero", "chunk", "FastWAM", "", "493.0", "ms"),
        ("libero", "chunk", "FastWAM-Joint-CD", "", "114.2", "ms"),
        ("libero", "chunk", "FastWAM-RTC", "", "142.3", "ms"),
        ("libero", "chunk", "StreamWAM", "", "41.0", "ms"),
        ("libero", "chunk", "w/o Action Conditioning", "", "35.1", "ms"),
        ("libero", "chunk", "w/o Slot Encoder", "", "36.3", "ms"),
        ("libero", "total", "FastWAM", "long", "16.31", "s"),
        ("libero", "total", "FastWAM", "short", "8.25", "s"),
        ("libero", "total", "FastWAM-Joint-CD", "long", "6.89", "s"),
        ("libero", "total", "FastWAM-Joint-CD", "short", "3.74", "s"),
        ("libero", "total", "FastWAM-RTC", "long", "6.23", "s"),
        ("libero", "total", "FastWAM-RTC", "short", "3.20", "s"),
        ("libero", "total", "StreamWAM", "long", "5.36", "s"),
        ("libero", "total", "StreamWAM", "short", "3.15", "s"),
        ("libero", "total", "w/o Action Conditioning", "long", "5.20", "s"),
        ("libero", "total", "w/o Action Conditioning", "short", "2.92", "s"),
        ("libero", "total", "w/o Slot Encoder", "long", "5.31", "s"),
        ("libero", "total", "w/o Slot Encoder", "short", "3.01", "s"),
        ("robocasa", "chunk", "X-WAM", "", "504.00", "ms"),
        ("robocasa", "chunk", "X-WAM-CD", "", "135.21", "ms"),
        ("robocasa", "chunk", "StreamWAM", "", "136.76", "ms"),
        ("robocasa", "total", "X-WAM", "", "37.31", "s"),
        ("robocasa", "total", "X-WAM-CD", "", "33.60", "s"),
        ("robocasa", "total", "StreamWAM", "", "11.76", "s"),
        ("robotwin", "chunk", "StarWAM-Joint", "", "190.17", "ms"),
        ("robotwin", "chunk", "StarWAM-CD", "", "81.21", "ms"),
        ("robotwin", "chunk", "StreamWAM", "", "47.09", "ms"),
        ("robotwin", "total", "StarWAM-Joint", "", "110.22", "s"),
        ("robotwin", "total", "StarWAM-CD", "", "102.59", "s"),
        ("robotwin", "total", "StreamWAM", "", "77.48", "s"),
    ]
