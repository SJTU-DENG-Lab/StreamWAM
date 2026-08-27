from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse
import xml.etree.ElementTree as ET

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_ROOT = REPO_ROOT / "docs"
INDEX_PATH = PAGE_ROOT / "index.html"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pages.yml"
LATENCY_GENERATOR_PATH = PAGE_ROOT / "generate_latency_figure.py"
METHOD_FIGURE_PATH = PAGE_ROOT / "assets" / "stream-wam-method.svg"
CITATION_COPY_HARNESS_PATH = REPO_ROOT / "tests" / "citation_copy_harness.js"
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
        self.eyebrow_text: list[str] | None = None
        self.eyebrow = ""
        self.header_depth = 0
        self.main_depth = 0
        self.lab_lockup_regions: list[str] = []
        self.caption_text: list[str] | None = None
        self.captions: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = {name: value or "" for name, value in attrs}
        self.tags.append(tag)
        self.attributes.append((tag, normalized))
        if tag == "header":
            self.header_depth += 1
        if tag == "main":
            self.main_depth += 1
        if "lab-lockup" in normalized.get("class", "").split():
            if self.header_depth:
                self.lab_lockup_regions.append("header")
            elif self.main_depth:
                self.lab_lockup_regions.append("main")
            else:
                self.lab_lockup_regions.append("other")
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
        if tag == "p" and "eyebrow" in normalized.get("class", "").split():
            self.eyebrow_text = []
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
            self.hero_title = " ".join(" ".join(self.hero_title_text).split())
            self.hero_title_text = None
        if tag == "p" and self.eyebrow_text is not None:
            self.eyebrow = " ".join(self.eyebrow_text).strip()
            self.eyebrow_text = None
        if tag == "caption" and self.caption_text is not None:
            self.captions.append(" ".join(self.caption_text).strip())
            self.caption_text = None
        if tag == "section":
            self.section_stack.pop()
        if tag == "header":
            self.header_depth -= 1
        if tag == "main":
            self.main_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.act_body_paragraph_text is not None:
            self.act_body_paragraph_text.append(data)
        if self.current_h3_id is not None:
            self.current_h3_text.append(data)
        if self.hero_title_text is not None:
            self.hero_title_text.append(data)
        if self.eyebrow_text is not None:
            self.eyebrow_text.append(data)
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
        self.current_row_styles: list[str] = []
        self.rows: dict[str, list[list[str]]] = {}
        self.cell_styles: dict[str, list[list[str]]] = {}
        self.section_text: dict[str, list[str]] = {}
        self.current_cell_style = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = {name: value or "" for name, value in attrs}
        if tag == "section" and normalized.get("id", "").startswith("benchmark-"):
            self.current_benchmark = normalized["id"]
            self.rows[self.current_benchmark] = []
            self.cell_styles[self.current_benchmark] = []
            self.section_text[self.current_benchmark] = []
        elif self.current_benchmark and tag == "tr":
            self.current_row = []
            self.current_row_styles = []
        elif self.current_row is not None and tag in {"th", "td"}:
            self.current_cell = []
            self.current_cell_style = ""
        elif self.current_cell is not None and tag == "strong":
            self.current_cell_style = "best"
        elif self.current_cell is not None and tag == "u":
            self.current_cell_style = "second"

    def handle_data(self, data: str) -> None:
        if self.current_benchmark and data.strip():
            self.section_text[self.current_benchmark].append(data.strip())
        if self.current_cell is not None and data.strip():
            self.current_cell.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.current_cell is not None and self.current_row is not None:
            self.current_row.append(" ".join(self.current_cell))
            self.current_row_styles.append(self.current_cell_style)
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None and self.current_benchmark:
            self.rows[self.current_benchmark].append(self.current_row)
            self.cell_styles[self.current_benchmark].append(self.current_row_styles)
            self.current_row = None
        elif tag == "section" and self.current_benchmark:
            self.current_benchmark = None


def parse_benchmarks() -> BenchmarkParser:
    parser = BenchmarkParser()
    parser.feed(INDEX_PATH.read_text(encoding="utf-8"))
    return parser


class LatencyDataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_table = False
        self.current_row: list[str] | None = None
        self.current_cell: list[str] | None = None
        self.rows: list[list[str]] = []
        self.sr_only_stack: list[bool] = []
        self.is_contained_by_sr_only = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        normalized = {name: value or "" for name, value in attrs}
        if tag == "div":
            inherited = self.sr_only_stack[-1] if self.sr_only_stack else False
            self.sr_only_stack.append(inherited or "sr-only" in normalized.get("class", "").split())
        if tag == "table" and "latency-data" in normalized.get("class", "").split():
            self.in_table = True
            self.is_contained_by_sr_only = bool(self.sr_only_stack and self.sr_only_stack[-1])
        elif self.in_table and tag == "tr":
            self.current_row = []
        elif self.current_row is not None and tag in {"th", "td"}:
            self.current_cell = []

    def handle_data(self, data: str) -> None:
        if self.current_cell is not None and data.strip():
            self.current_cell.append(data.strip())

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self.current_cell is not None and self.current_row is not None:
            self.current_row.append(" ".join(self.current_cell))
            self.current_cell = None
        elif tag == "tr" and self.current_row is not None:
            self.rows.append(self.current_row)
            self.current_row = None
        elif tag == "table" and self.in_table:
            self.in_table = False
        elif tag == "div" and self.sr_only_stack:
            self.sr_only_stack.pop()


def parse_latency_data() -> LatencyDataParser:
    parser = LatencyDataParser()
    parser.feed(INDEX_PATH.read_text(encoding="utf-8"))
    return parser


def test_page_exposes_the_research_preview_and_available_artifacts() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)
    visible_copy_without_citation = visible_text.split("@misc{", 1)[0]

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
    assert parser.hero_title == "Streaming Your World-Action Model for Real-Time Robot Manipulation."
    assert parser.eyebrow == "Stream-WAM"
    assert parser.lab_lockup_regions == ["main"]
    assert "Think ahead. Act now." not in visible_text
    assert "Streaming Your World-Action Model for Real-Time Robot Manipulation." in visible_text
    assert "committed action prefix" in visible_text
    assert "action-conditioned" in visible_text.lower()
    assert "Coming Soon" in visible_text

    links = {
        attrs.get("href")
        for tag, attrs in parser.attributes
        if tag == "a" and attrs.get("href")
    }
    assert "https://github.com/SJTU-DENG-Lab/StreamWAM" in links
    assert "https://huggingface.co/SJTU-DENG-Lab/StreamWAM" in links
    assert "https://sjtu-deng-lab.github.io/home/" in links
    assert "#" not in links
    assert "MLSys Team" not in visible_text
    assert "RTC-AC" not in html
    assert "AC-StreamWAM" not in html
    assert html.count(">Stream-WAM (Ours)<") == 3
    assert "StreamWAM" not in visible_copy_without_citation

    deng_logos = [
        attrs
        for tag, attrs in parser.attributes
        if tag == "img" and "deng-lab-logo" in attrs.get("class", "").split()
    ]
    assert deng_logos == [
        {
            "class": "deng-lab-logo",
            "src": "assets/deng-lab.webp",
            "alt": "",
            "width": "420",
            "height": "155",
        }
    ]


def test_page_publishes_the_current_benchmark_results() -> None:
    parser, _ = parse_page()
    visible_text = " ".join(parser.text_parts)

    expected_fragments = (
        "98.20%",
        "41.0 ms",
        "9.49 s",
        "96.60",
        "98.80",
        "97.40",
        "100.00",
        "75.35%",
        "87.2",
        "88.8",
        "87.6",
        "24 kitchen manipulation tasks",
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


def test_local_css_and_script_use_matching_release_versions() -> None:
    parser, _ = parse_page()
    local_assets = [
        attrs.get("href") or attrs.get("src")
        for tag, attrs in parser.attributes
        if tag in {"link", "script"}
        and (
            attrs.get("href", "").startswith("styles.css")
            or attrs.get("src", "").startswith("script.js")
        )
    ]

    assert len(local_assets) == 2
    versions = [urlparse(value).query for value in local_assets if value]
    assert len(versions) == 2
    assert versions[0] == versions[1]
    assert re.fullmatch(r"v=\d{8}-\d+", versions[0])


def test_project_page_uses_standard_docs_directory() -> None:
    assert PAGE_ROOT.joinpath("index.html").is_file()
    assert not REPO_ROOT.joinpath("academic_project_page").exists()


def test_pages_workflow_deploys_only_the_project_page() -> None:
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "contents: read" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "actions/configure-pages@v5" in workflow
    assert "actions/upload-pages-artifact@v4" in workflow
    assert "actions/deploy-pages@v4" in workflow
    assert '"docs/**"' in workflow
    assert "path: ./docs" in workflow
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
    assert "Stream-WAM conditions the visual future on the action already underway" in visible_text
    assert "inference-time RTC" in visible_text
    assert "prefix-conditioned" in visible_text
    assert parser.eyebrow == "Stream-WAM"
    assert "Task performance." in visible_text
    assert "Inference efficiency." in visible_text
    assert "Our evaluations use four NVIDIA H100 GPUs" in visible_text
    assert "12.0× on LIBERO" in visible_text
    assert "3.2× on RoboCasa" in visible_text
    assert "1.8× on RoboCasa" in visible_text
    assert "broader analysis, limitations, and failure cases" in visible_text
    assert "Citation" in visible_text
    assert "Action-conditioned attention" in visible_text
    assert theme_colors == ["#f7f5ef"]
    assert not any(token in html.casefold() for token in ("todo", "tbd", "lorem ipsum"))


def test_academic_spacetime_method_figure_replaces_the_illustrated_timeline() -> None:
    parser, html = parse_page()
    figure_images = [
        attrs
        for tag, attrs in parser.attributes
        if tag == "img" and "method-figure-artwork" in attrs.get("class", "").split()
    ]
    caption = (
        "The overview shows Stream-WAM repeatedly predicting the next visual-action "
        "chunk during continuous robot execution. The temporal inset samples O₁ after "
        "A₀[0:8] and reuses A₀[8:16] both as A₁[0:8] and as eight shared action "
        "slots; eight unknown slots complete the 16-slot condition input for the next "
        "visual future."
    )
    description = (
        "A static two-scale method figure. The global t₀-to-t₃ overview shows two "
        "Stream updates overlapped with continuous execution. The lower-left "
        "temporal view marks O₁ after A₀[0:8], then routes A₀[8:16] both into the "
        "hard action prefix A₁[0:8] and into eight shared action slots. Eight unknown "
        "slots are appended to form 16 condition slots for visual-future generation. "
        "The lower-right ten-by-ten mask highlights the two shared-action links into "
        "the next visual-future query."
    )

    assert len(figure_images) == 1
    assert figure_images[0]["src"] == "assets/stream-wam-method.svg?v=20260827-1"
    assert figure_images[0]["width"] == "1600"
    assert figure_images[0]["height"] == "740"
    assert figure_images[0]["alt"] == ""
    assert caption in " ".join(parser.text_parts)
    assert description in " ".join(parser.text_parts)
    assert 'class="method-figure-viewport"' in html
    assert 'aria-describedby="streamwam-method-description streamwam-method-caption"' in html
    assert "Executing action prefix</strong>" not in html

    opening_position = html.index(
        "Stream-WAM conditions the visual future on the action already underway."
    )
    figure_position = html.index('id="streamwam-method-figure"')
    loop_position = html.index("The loop repeats at every boundary.")
    experiments_position = html.index('id="experiments"')
    assert opening_position < figure_position < loop_position < experiments_position


def test_method_figure_extended_description_uses_the_existing_screen_reader_class() -> None:
    parser, _ = parse_page()
    descriptions = [
        attrs
        for tag, attrs in parser.attributes
        if tag == "p" and attrs.get("id") == "streamwam-method-description"
    ]
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")

    assert len(descriptions) == 1
    assert "sr-only" in descriptions[0].get("class", "").split()
    assert re.search(r"\.sr-only\s*\{", css)


def test_streamwam_method_svg_encodes_academic_spacetime_semantics() -> None:
    assert METHOD_FIGURE_PATH.is_file()
    root = ET.parse(METHOD_FIGURE_PATH).getroot()
    ids = {element.attrib["id"] for element in root.iter() if "id" in element.attrib}
    text = " ".join(part.strip() for part in root.itertext() if part.strip())

    assert root.attrib["viewBox"] == "0 0 1600 740"
    assert {
        "time-axis",
        "time-t0",
        "time-t1",
        "time-t2",
        "time-t3",
        "global-overview",
        "overview-cold-start",
        "overview-cycle-1",
        "overview-cycle-2",
        "overview-execution-0",
        "overview-execution-1",
        "overview-update-1",
        "overview-update-2",
        "overview-video-1",
        "overview-action-1",
        "joint-wam-0",
        "video-0",
        "action-0",
        "inset-selection",
        "inset-callout",
        "detail-inset",
        "detail-inset-frame",
        "temporal-overlap-panel",
        "attention-panel",
        "inset-divider",
        "inset-a0-before",
        "inset-a0-overlap",
        "inset-a0-remaining",
        "inset-a1-action-prefix",
        "inset-a1-continuation",
        "inset-observation",
        "inset-shared-action-slots",
        "inset-unknown-slots",
        "shared-to-action-prefix",
        "shared-to-condition-slots",
        "inset-update",
        "inset-video-1",
        "inset-action-1",
        "streamwam-attention-mask",
        "attention-row-labels",
        "attention-column-labels",
    } <= ids
    for label in (
        "Cold start",
        "Streaming overview",
        "Robot execution A₀",
        "Robot execution A₁",
        "Joint WAM",
        "Stream update 1",
        "Stream update 2",
        "Temporal overlap",
        "Action-conditioned attention",
        "A₀[8:16] = A₁[0:8]",
        "executed · 8",
        "shared actions · 8",
        "remaining A₀ · 16",
        "action prefix · 8",
        "predicted actions · 24",
        "shared action slots · 8",
        "unknown slots · 8",
        "16 condition slots",
        "Stream Update",
    ):
        assert label in text
    for obsolete_label in (
        "Aligned action prefix",
        "Next chunk ready",
        "completion time varies",
        "One overlap window",
        "AC-Stream update",
        "Standard Joint WAM",
        "handoff",
        "Observe O₁",
        "predict A₁ while A₀ continues",
        "known action context",
        "unknown future slots",
        "shared prefix",
        "look-ahead",
        "to next visual future",
    ):
        assert obsolete_label not in text
    classes = " ".join(
        element.attrib.get("class", "") for element in root.iter()
    ).split()
    assert "action-cell" not in classes
    assert "condition-slot" not in classes
    assert sum(element.attrib.get("id") == "detail-inset" for element in root.iter()) == 1
    assert not re.search(r"\b(?:VM|IDM|FDM|cache|caches)\b", text, re.IGNORECASE)
    overview = next(element for element in root.iter() if element.attrib.get("id") == "global-overview")
    overview_text = " ".join(part.strip() for part in overview.itertext() if part.strip()).casefold()
    assert "action-conditioned" not in overview_text
    assert "handoff" not in overview_text


def test_streamwam_method_svg_places_prediction_inside_execution_and_inset_in_whitespace() -> None:
    assert METHOD_FIGURE_PATH.is_file()
    root = ET.parse(METHOD_FIGURE_PATH).getroot()
    by_id = {
        element.attrib["id"]: element
        for element in root.iter()
        if "id" in element.attrib
    }
    def bounds(element_id: str) -> tuple[float, float, float, float]:
        element = by_id[element_id]
        left = float(element.attrib["x"])
        top = float(element.attrib["y"])
        return (
            left,
            top,
            left + float(element.attrib["width"]),
            top + float(element.attrib["height"]),
        )

    for execution_id, update_id in (
        ("overview-execution-0", "overview-update-1"),
        ("overview-execution-1", "overview-update-2"),
    ):
        execution = bounds(execution_id)
        update = bounds(update_id)
        assert execution[0] < update[0] < update[2] <= execution[2]

    selection = bounds("inset-selection")
    selected_execution = bounds("overview-execution-0")
    assert max(selection[0], selected_execution[0]) < min(
        selection[2], selected_execution[2]
    )
    assert max(selection[1], selected_execution[1]) < min(
        selection[3], selected_execution[3]
    )
    inset = bounds("detail-inset-frame")
    assert inset[0] == 58
    assert inset[2] == 1520
    assert 0 <= inset[1] < inset[3] <= 740
    overview_boxes = [
        bounds(element_id)
        for element_id in (
            "overview-execution-0",
            "overview-execution-1",
            "overview-update-1",
            "overview-update-2",
            "overview-video-1",
            "overview-action-1",
        )
    ]
    assert max(box[3] for box in overview_boxes) < inset[1]

    temporal_panel = bounds("temporal-overlap-panel")
    attention_panel = bounds("attention-panel")
    divider_x = float(by_id["inset-divider"].attrib["x1"])
    assert temporal_panel[2] < divider_x < attention_panel[0]

    a0_overlap = bounds("inset-a0-overlap")
    action_prefix = bounds("inset-a1-action-prefix")
    assert a0_overlap[0] == action_prefix[0]
    assert a0_overlap[2] == action_prefix[2]
    shared_slots = bounds("inset-shared-action-slots")
    unknown_slots = bounds("inset-unknown-slots")
    assert action_prefix[0] == shared_slots[0]
    assert shared_slots[2] == unknown_slots[0]
    assert unknown_slots[2] == action_prefix[2]
    assert action_prefix[3] < shared_slots[1]
    continuation = bounds("inset-a1-continuation")
    assert continuation[0] == action_prefix[2]
    observation = by_id["inset-observation"]
    assert float(observation.attrib["x1"]) == a0_overlap[0]
    assert float(observation.attrib["x2"]) == a0_overlap[0]
    overview_cycle = by_id["overview-cycle-1"]
    connector_paths = [
        element
        for element in overview_cycle.iter()
        if element.tag.endswith("path")
        and "connector" in element.attrib.get("class", "").split()
    ]
    assert len(connector_paths) == 1
    assert not any(
        "gold-path" in element.attrib.get("class", "").split()
        for cycle_id in ("overview-cycle-1", "overview-cycle-2")
        for element in by_id[cycle_id].iter()
    )


def test_streamwam_method_svg_uses_direct_overlap_mappings_and_routes_outputs() -> None:
    root = ET.parse(METHOD_FIGURE_PATH).getroot()
    by_id = {
        element.attrib["id"]: element
        for element in root.iter()
        if "id" in element.attrib
    }

    def path_endpoint(element_id: str) -> tuple[float, float]:
        tokens = re.findall(r"[MLHV]|-?\d+(?:\.\d+)?", by_id[element_id].attrib["d"])
        x = y = 0.0
        index = 0
        command = ""
        while index < len(tokens):
            token = tokens[index]
            if token in {"M", "L", "H", "V"}:
                command = token
                index += 1
                continue
            if command in {"M", "L"}:
                x, y = float(tokens[index]), float(tokens[index + 1])
                index += 2
            elif command == "H":
                x = float(token)
                index += 1
            elif command == "V":
                y = float(token)
                index += 1
        return x, y

    update = by_id["inset-update"]
    update_x = float(update.attrib["x"])
    update_y = float(update.attrib["y"])
    update_bottom = update_y + float(update.attrib["height"])
    for path_id in ("inset-observation-input", "inset-context-input"):
        endpoint_x, endpoint_y = path_endpoint(path_id)
        assert endpoint_x == update_x - 7
        assert update_y <= endpoint_y <= update_bottom

    observation_path = by_id["inset-observation-input"]
    assert re.findall(r"[A-Za-z]", observation_path.attrib["d"]) == ["M", "H"]

    shared = by_id["inset-a0-overlap"]
    shared_left = float(shared.attrib["x"])
    shared_right = shared_left + float(shared.attrib["width"])
    shared_bottom = float(shared.attrib["y"]) + float(shared.attrib["height"])
    action_prefix_path = by_id["shared-to-action-prefix"]
    action_prefix_coordinates = [
        float(value)
        for value in re.findall(r"-?\d+(?:\.\d+)?", action_prefix_path.attrib["d"])
    ]
    action_source_x, action_source_y = action_prefix_coordinates[:2]
    action_endpoint_x, action_endpoint_y = path_endpoint("shared-to-action-prefix")
    action_prefix = by_id["inset-a1-action-prefix"]
    action_prefix_left = float(action_prefix.attrib["x"])
    action_prefix_right = action_prefix_left + float(action_prefix.attrib["width"])
    action_prefix_top = float(action_prefix.attrib["y"])
    action_prefix_bottom = action_prefix_top + float(action_prefix.attrib["height"])
    assert shared_left <= action_source_x <= shared_right
    assert action_source_y == shared_bottom
    assert action_prefix_left <= action_endpoint_x <= action_prefix_right
    assert action_endpoint_y == action_prefix_top - 7
    assert "gold-path" in action_prefix_path.attrib.get("class", "").split()

    slot_path = by_id["shared-to-condition-slots"]
    slot_commands = re.findall(r"[A-Za-z]", slot_path.attrib["d"])
    slot_coordinates = [
        float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", slot_path.attrib["d"])
    ]
    (
        slot_source_x,
        slot_source_y,
        branch_y,
        bypass_x,
        slot_entry_y,
        _,
    ) = slot_coordinates
    slot_endpoint_x, slot_endpoint_y = path_endpoint("shared-to-condition-slots")
    shared_slots = by_id["inset-shared-action-slots"]
    shared_slots_left = float(shared_slots.attrib["x"])
    shared_slots_top = float(shared_slots.attrib["y"])
    shared_slots_bottom = shared_slots_top + float(shared_slots.attrib["height"])
    assert slot_commands == ["M", "V", "H", "V", "H"]
    assert shared_left <= slot_source_x <= shared_right
    assert slot_source_y == shared_bottom < action_prefix_top
    assert shared_bottom < branch_y < action_prefix_top
    assert bypass_x < action_prefix_left
    assert slot_entry_y == slot_endpoint_y > action_prefix_bottom
    assert slot_endpoint_x == shared_slots_left - 7
    assert shared_slots_top <= slot_endpoint_y <= shared_slots_bottom
    assert "gold-path" in slot_path.attrib.get("class", "").split()

    assert "continuation-to-unknown" not in by_id

    for path_id, target_id in (
        ("inset-condition-to-video", "inset-video-1"),
        ("inset-action-output", "inset-action-1"),
    ):
        endpoint_x, endpoint_y = path_endpoint(path_id)
        target = by_id[target_id]
        target_x = float(target.attrib["x"])
        target_y = float(target.attrib["y"])
        target_bottom = target_y + float(target.attrib["height"])
        assert endpoint_x == target_x - 7
        assert target_y <= endpoint_y <= target_bottom

    visual_path = by_id["inset-condition-to-video"]
    assert "gold-path" in visual_path.attrib.get("class", "").split()
    svg_source = METHOD_FIGURE_PATH.read_text(encoding="utf-8")
    assert re.search(r"\.gold-path\s*\{[^}]*marker-end:url\(#arrow-gold\)", svg_source)


def test_streamwam_method_svg_attention_mask_matches_ac_stream_connectivity() -> None:
    root = ET.parse(METHOD_FIGURE_PATH).getroot()
    namespace = "{http://www.w3.org/2000/svg}"
    by_id = {
        element.attrib["id"]: element
        for element in root.iter()
        if "id" in element.attrib
    }
    token_labels = ["f₀", "f₁", "fₕ", "a₁", "aₕ"] * 2

    mask = by_id["streamwam-attention-mask"]
    cells = mask.findall(f"{namespace}rect")
    assert len(cells) == 100
    assert {
        (int(cell.attrib["data-row"]), int(cell.attrib["data-col"]))
        for cell in cells
    } == {(row, col) for row in range(10) for col in range(10)}

    conditioned = {
        (int(cell.attrib["data-row"]), int(cell.attrib["data-col"]))
        for cell in cells
        if "attention-conditioned" in cell.attrib.get("class", "").split()
    }
    assert conditioned == {(6, 3), (6, 4)}

    visual_cells = {
        (0, 0),
        (1, 0), (1, 1), (1, 2),
        (2, 0), (2, 1), (2, 2),
        (5, 5),
        (6, 5), (6, 6), (6, 7),
        (7, 5), (7, 6), (7, 7),
    }
    action_cells = {
        *((row, col) for row in (3, 4) for col in range(5)),
        *((row, col) for row in (8, 9) for col in range(5, 10)),
    }
    expected_classes = {}
    for row in range(10):
        for col in range(10):
            coordinate = (row, col)
            expected_classes[coordinate] = (
                "attention-conditioned"
                if coordinate in conditioned
                else "attention-visual"
                if coordinate in visual_cells
                else "attention-action"
                if coordinate in action_cells
                else "attention-masked"
            )
    assert {
        (int(cell.attrib["data-row"]), int(cell.attrib["data-col"])): next(
            class_name
            for class_name in cell.attrib.get("class", "").split()
            if class_name.startswith("attention-")
        )
        for cell in cells
    } == expected_classes

    row_labels = [
        "".join(label.itertext())
        for label in by_id["attention-row-labels"].findall(f"{namespace}text")
    ]
    column_labels = [
        "".join(label.itertext())
        for label in by_id["attention-column-labels"].findall(f"{namespace}text")
    ]
    assert row_labels == token_labels
    assert column_labels == token_labels


def test_streamwam_method_svg_reserves_readable_space_for_update_labels() -> None:
    assert METHOD_FIGURE_PATH.is_file()
    root = ET.parse(METHOD_FIGURE_PATH).getroot()
    by_id = {
        element.attrib["id"]: element
        for element in root.iter()
        if "id" in element.attrib
    }

    assert float(by_id["overview-update-1"].attrib["width"]) >= 130
    assert float(by_id["overview-update-2"].attrib["width"]) >= 130
    assert float(by_id["inset-update"].attrib["width"]) >= 115
    assert float(by_id["detail-inset-frame"].attrib["width"]) >= 850

    svg_source = METHOD_FIGURE_PATH.read_text(encoding="utf-8")

    def class_font_size(class_name: str) -> float:
        match = re.search(
            rf"\.{class_name}\s*\{{[^}}]*font:[^;]*?([0-9.]+)px/",
            svg_source,
        )
        assert match is not None
        return float(match.group(1))

    assert class_font_size("label") >= 15
    assert class_font_size("small") >= 14
    assert class_font_size("gold-label") >= 14
    assert class_font_size("legend") >= 13


def test_streamwam_second_update_is_a_lighter_repetition() -> None:
    assert METHOD_FIGURE_PATH.is_file()
    root = ET.parse(METHOD_FIGURE_PATH).getroot()
    second_update = next(
        element for element in root.iter() if element.attrib.get("id") == "overview-cycle-2"
    )

    assert "opacity" not in second_update.attrib
    assert "secondary" in second_update.attrib.get("class", "").split()
    svg_namespace = "{http://www.w3.org/2000/svg}"
    assert all(
        "opacity" not in text.attrib
        for text in second_update.findall(f".//{svg_namespace}text")
    )
    svg_source = METHOD_FIGURE_PATH.read_text(encoding="utf-8")
    assert re.search(
        r"\.secondary\s+rect,\.secondary\s+path\s*\{[^}]*opacity:\.74",
        svg_source,
    )


def test_streamwam_method_svg_uses_flat_academic_style() -> None:
    assert METHOD_FIGURE_PATH.is_file()
    root = ET.parse(METHOD_FIGURE_PATH).getroot()
    svg_namespace = "{http://www.w3.org/2000/svg}"

    for tag in ("filter", "linearGradient", "radialGradient"):
        assert root.findall(f".//{svg_namespace}{tag}") == []
    for rectangle in root.findall(f".//{svg_namespace}rect"):
        assert float(rectangle.attrib.get("rx", "0")) <= 6


def test_streamwam_method_svg_is_fully_static_and_accessible() -> None:
    assert METHOD_FIGURE_PATH.is_file()
    root = ET.parse(METHOD_FIGURE_PATH).getroot()
    svg_source = METHOD_FIGURE_PATH.read_text(encoding="utf-8")
    svg_namespace = "{http://www.w3.org/2000/svg}"

    assert root.attrib["role"] == "img"
    assert root.attrib["aria-labelledby"] == "streamwam-method-title streamwam-method-desc"
    assert root.find(f"{svg_namespace}title") is not None
    assert root.find(f"{svg_namespace}desc") is not None
    assert "@keyframes" not in svg_source
    assert "animation:" not in svg_source
    assert "time-cursor" not in svg_source
    for tag in ("animate", "animateTransform", "set"):
        assert root.findall(f".//{svg_namespace}{tag}") == []


def test_method_figure_scrolls_inside_its_mobile_viewport() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    viewport_rule = re.search(r"\.method-figure-viewport\s*\{([^}]*)\}", css)
    mobile_rule = re.search(r"@media \(max-width: 760px\)\s*\{(.*)\n\}", css, re.DOTALL)

    assert viewport_rule is not None
    assert re.search(r"overflow-x:\s*auto", viewport_rule.group(1))
    assert mobile_rule is not None
    assert re.search(
        r"\.method-figure-artwork\s*\{[^}]*min-width:\s*1100px",
        mobile_rule.group(1),
    )


def test_hidden_mobile_menu_remains_hidden_without_javascript() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")

    assert ".menu-toggle[hidden]{display:none!important}" in css


def test_masthead_and_hero_scale_like_the_visual_reference() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")

    header_rule = re.search(r"\.site-header\s*\{([^}]*)\}", css)
    assert header_rule is not None
    assert re.search(r"position:\s*fixed", header_rule.group(1))
    assert re.search(r"top:\s*18px", header_rule.group(1))
    assert re.search(r"width:\s*min\(1120px,\s*calc\(100%\s*-\s*32px\)\)", header_rule.group(1))
    assert re.search(r"border-radius:\s*18px", header_rule.group(1))

    title_rule = re.search(r"\.hero h1\s*\{([^}]*)\}", css)
    assert title_rule is not None
    assert re.search(r"font-size:\s*clamp\(46px,\s*4\.15vw,\s*72px\)", title_rule.group(1))

    lead_rule = re.search(r"\.hero-lede\s*\{([^}]*)\}", css)
    assert lead_rule is not None
    assert re.search(r"font-size:\s*clamp\(18px,\s*1\.55vw,\s*22px\)", lead_rule.group(1))


def test_wide_hero_accents_only_streaming_and_real_time() -> None:
    _, html = parse_page()
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")

    assert '<span class="title-accent-streaming">Streaming</span>' in html
    assert '<span class="title-accent-realtime">Real-Time</span>' in html
    assert "title-accent-model" not in html
    assert "title-accent-control" not in html
    assert "--shell: min(1220px" in css
    assert "--hero-shell: min(1680px" in css
    realtime_rule = re.search(r"\.title-accent-realtime\s*\{([^}]*)\}", css)
    assert realtime_rule is not None
    assert "white-space: nowrap" in realtime_rule.group(1)


def test_headline_metrics_follow_actions_and_use_libero_speedups() -> None:
    _, html = parse_page()
    actions_start = html.index('class="hero-actions"')
    actions_end = html.index("</div>", actions_start)
    metrics_start = html.index('class="headline-results"')
    figure_start = html.index('class="hero-figure')
    metric_markup = html[metrics_start:figure_start]

    assert actions_end < metrics_start < figure_start
    for text in ("98.20%", "41.0 ms", "12.0×", "4.74 s", "3.4×"):
        assert text in metric_markup
    for label in ("LIBERO average success", "LIBERO chunk time", "LIBERO total time"):
        assert label in metric_markup
    assert metric_markup.count("vs FastWAM") == 2
    assert "RoboCasa total time" not in metric_markup


def test_headline_metrics_use_compact_button_scale() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    container = re.search(r"\.headline-results\s*\{([^}]*)\}", css)
    item = re.search(r"\.headline-results p\s*\{([^}]*)\}", css)

    assert container is not None
    assert item is not None
    assert "box-shadow: none" in container.group(1)
    assert "background: transparent" in container.group(1)
    assert "min-height: 54px" in item.group(1)
    assert "max-height: 68px" in item.group(1)


def test_hero_width_is_independent_from_article_breakouts() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    hero_shell = re.search(r"\.hero\.shell\s*\{([^}]*)\}", css)

    assert hero_shell is not None
    assert "width: var(--hero-shell)" in hero_shell.group(1)


def test_hero_main_and_figure_share_the_second_desktop_grid_row() -> None:
    _, html = parse_page()
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    hero_copy_start = html.index('<div class="hero-copy">')
    lab_start = html.index('class="lab-lockup"', hero_copy_start)
    hero_main_start = html.index('<div class="hero-main">', hero_copy_start)
    figure_start = html.index('class="hero-figure', hero_main_start)
    hero_main_markup = html[hero_main_start:figure_start]

    assert hero_copy_start < lab_start < hero_main_start < figure_start
    assert "lab-lockup" not in hero_main_markup
    for class_name in (
        "eyebrow",
        "hero-lede",
        "hero-actions",
        "headline-results",
    ):
        assert f'class="{class_name}"' in hero_main_markup
    assert 'id="hero-title"' in hero_main_markup

    assert ".hero-copy { display: contents; }" in css
    assert ".hero-main { grid-column: 1; grid-row: 2; }" in css
    assert ".lab-lockup { grid-column: 1; grid-row: 1;" in css
    assert ".hero-figure { grid-column: 2; grid-row: 2;" in css
    assert "align-self: stretch" in css
    stacked_css = css.split("@media (max-width: 1040px)", 1)[1].split(
        "@media (max-width: 760px)", 1
    )[0]
    assert ".hero-copy { display: block; }" in stacked_css


def test_readable_attention_visual_drops_numbers_and_allocates_more_width() -> None:
    _, html = parse_page()
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    pipeline_markup = html[
        html.index('<div class="pipeline-visual"') : html.index(
            '<figcaption id="pipeline-caption"'
        )
    ]

    assert '<span>01</span>' not in pipeline_markup
    assert '<span>02</span>' not in pipeline_markup
    assert pipeline_markup.count('class="visual-section-heading"') == 2

    hero_rule = re.search(r"\.hero\s*\{([^}]*)\}", css)
    panel_heading_rule = re.search(r"\.attention-panel-heading strong\s*\{([^}]*)\}", css)
    path_badge_rule = re.search(r"\.attention-path-badge\s*\{([^}]*)\}", css)
    token_label_rule = re.search(
        r"\.mask-column-labels span, \.mask-row-labels span\s*\{([^}]*)\}", css
    )
    row_label_grid_rule = re.search(r"\.mask-row-labels\s*\{([^}]*)\}", css)
    matrix_grid_rule = re.search(r"\.mask-grid-10\s*\{([^}]*)\}", css)
    matrix_cell_rule = re.search(r"\.mask-grid-10 \.matrix-cell\s*\{([^}]*)\}", css)

    assert hero_rule is not None
    assert "minmax(700px" in hero_rule.group(1)
    assert panel_heading_rule is not None
    assert "font-size: .82rem" in panel_heading_rule.group(1)
    assert path_badge_rule is not None
    assert "font: 800 .74rem/1" in path_badge_rule.group(1)
    assert token_label_rule is not None
    assert "font: 700 .72rem/1" in token_label_rule.group(1)
    assert row_label_grid_rule is not None
    assert "repeat(10,minmax(17px,1fr))" in row_label_grid_rule.group(1)
    assert matrix_grid_rule is not None
    assert "repeat(10,minmax(17px,1fr))" in matrix_grid_rule.group(1)
    assert matrix_cell_rule is not None
    assert "min-height: 17px" in matrix_cell_rule.group(1)
    assert ".mask-chunk-heads" not in css


def test_control_loop_heading_uses_readable_interface_typography() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    heading_rule = re.search(
        r"\.pipeline-heading > div:first-child span\s*\{([^}]*)\}", css
    )

    assert heading_rule is not None
    declarations = heading_rule.group(1)
    for expected in (
        "font-family: inherit",
        "font-size: .875rem",
        "font-weight: 760",
        "letter-spacing: -.01em",
        "color: #dce6eb",
    ):
        assert expected in declarations
    assert "ui-monospace" not in declarations


def test_runtime_visual_has_two_tracks_without_old_flow_copy() -> None:
    _, html = parse_page()
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    pipeline_markup = html[
        html.index('<div class="pipeline-visual"') : html.index(
            '<figcaption id="pipeline-caption"'
        )
    ]

    assert pipeline_markup.count('class="runtime-row') == 2
    assert "Synchronous WAM" in pipeline_markup
    assert "Stream-WAM" in pipeline_markup
    assert "Predict through the second half of each action chunk" in pipeline_markup
    for forbidden in (
        "Naive Async",
        ">Inference<",
        "Video &amp; Action Chunk",
        "Current Observation",
        "Directed feedback",
        "Visual Future + Next Action Chunk",
    ):
        assert forbidden not in pipeline_markup
    assert 'class="generation-window' in pipeline_markup
    assert pipeline_markup.count('class="timeline-curtain"') == 2
    assert 'class="execution-rail execution-continuous"' not in pipeline_markup
    for segment in ("one", "two", "three"):
        assert f'class="execution-rail execution-stream-{segment}"' in pipeline_markup
    assert 'aria-describedby="pipeline-description pipeline-caption"' in pipeline_markup
    assert 'class="sr-only" id="pipeline-description"' in pipeline_markup
    assert not re.search(r"\b\d+\s*(?:actions?|ms|seconds?)\b", pipeline_markup, re.I)
    for label in ("Model Prediction", "Robot Execution"):
        assert label in pipeline_markup
    for old_label in (
        "World-Action Prediction × Robot Execution",
        "World-Action Prediction",
        "Committed Actions",
        "Model update",
        "Robot motion",
        "Action prefix",
    ):
        assert old_label not in pipeline_markup
    assert 'class="committed-actions-segment"' not in pipeline_markup
    legend_rule = re.search(r"\.pipeline-legend span\s*\{([^}]*)\}", css)
    assert legend_rule is not None
    assert "text-transform: uppercase" not in legend_rule.group(1)
    visual_rule = re.search(r"\.pipeline-visual\s*\{([^}]*)\}", css)
    assert visual_rule is not None
    assert "min-height: 680px" not in visual_rule.group(1)
    assert "@keyframes timeline-reveal" in css
    assert "generation-reveal-sync" not in css
    assert "generation-reveal-stream" not in css
    cursor_rule = re.search(r"\.timeline-cursor\s*\{([^}]*)\}", css)
    assert cursor_rule is not None
    assert "animation: timeline-reveal 7s linear infinite" in cursor_rule.group(1)
    curtain_rule = re.search(r"\.timeline-curtain\s*\{([^}]*)\}", css)
    assert curtain_rule is not None
    assert "background: #020911" in curtain_rule.group(1)
    assert "animation: timeline-reveal 7s linear infinite" in curtain_rule.group(1)

    stream_row = pipeline_markup[
        pipeline_markup.index('<div class="runtime-row runtime-stream"') :
    ]
    assert 'class="generation-window generation-four"' in stream_row
    assert stream_row.index('class="generation-window generation-one"') < stream_row.index(
        'class="execution-rail execution-stream-one"'
    )
    for rule, placement in (
        (".runtime-sync .generation-one", ("left: 2%", "width: 25%")),
        (".runtime-sync .execution-one", ("left: 27%", "width: 20%")),
        (".runtime-sync .generation-two", ("left: 47%", "width: 27%")),
        (".runtime-sync .execution-two", ("left: 74%", "width: 24%")),
        (".runtime-stream .generation-two", ("left: 34%", "width: 12%")),
        (".runtime-stream .generation-three", ("left: 59%", "width: 12%")),
        (".runtime-stream .generation-four", ("left: 85%", "width: 13%")),
        (".runtime-stream .execution-stream-one", ("left: 22%", "width: 24%")),
        (".runtime-stream .execution-stream-two", ("left: 47%", "width: 24%")),
        (".runtime-stream .execution-stream-three", ("left: 72%", "width: 26%")),
    ):
        css_rule = re.search(rf"{re.escape(rule)}\s*\{{([^}}]*)\}}", css)
        assert css_rule is not None
        assert all(value in css_rule.group(1) for value in placement)
    assert "@media (prefers-reduced-motion: reduce)" in css
    reduced_motion = css.split("@media (prefers-reduced-motion: reduce)", 1)[1]
    assert ".timeline-curtain" in reduced_motion
    assert ".timeline-cursor" in reduced_motion
    assert "display: none" in reduced_motion


def test_attention_matrix_compares_visual_and_action_attention() -> None:
    _, html = parse_page()
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    pipeline_markup = html[
        html.index('<div class="pipeline-visual"') : html.index(
            '<figcaption id="pipeline-caption"'
        )
    ]

    assert pipeline_markup.count('class="attention-matrix mask-grid-10') == 2
    assert 'class="attention-panel attention-standard"' in pipeline_markup
    assert 'class="attention-panel attention-conditioned"' in pipeline_markup
    assert pipeline_markup.count('class="matrix-cell cross-chunk-condition') == 2
    assert "Action-conditioned attention" in pipeline_markup
    assert "Standard Joint WAM" in pipeline_markup
    for token_label in ("f₀", "f₁", "fₕ", "a₁", "aₕ"):
        assert pipeline_markup.count(token_label) >= 8
    assert "Allowed" in pipeline_markup
    assert "Masked" in pipeline_markup
    assert 'class="attention-path-badge">Action conditioned</span>' in pipeline_markup

    compact_markup = re.sub(r">\s+<", "><", pipeline_markup)
    standard_panel = compact_markup[
        compact_markup.index('class="attention-panel attention-standard"') :
        compact_markup.index('class="attention-panel attention-conditioned"')
    ]
    conditioned_panel = compact_markup[
        compact_markup.index('class="attention-panel attention-conditioned"') :
        compact_markup.index("</section>", compact_markup.index('class="attention-panel attention-conditioned"'))
    ]
    for panel in (standard_panel, conditioned_panel):
        assert "Chunk k" not in panel
        assert "Chunk k+1" not in panel
        assert "ᵏ" not in panel
        assert "⁺¹" not in panel

    def cell_classes(panel: str) -> list[str]:
        return re.findall(r'<i class="([^"]*\bmatrix-cell\b[^"]*)"></i>', panel)

    standard_cells = cell_classes(standard_panel)
    conditioned_cells = cell_classes(conditioned_panel)
    assert len(standard_cells) == 100
    assert len(conditioned_cells) == 100

    visual = "matrix-cell visual-token"
    action = "matrix-cell action-token"
    masked = "matrix-cell masked-cell"
    cross = "matrix-cell cross-chunk-condition"
    expected = [masked] * 100
    for offset in (0, 5):
        expected[(offset + 0) * 10 + offset + 0] = visual
        for row in (offset + 1, offset + 2):
            for column in (offset + 0, offset + 1, offset + 2):
                expected[row * 10 + column] = visual
        for row in (offset + 3, offset + 4):
            for column in range(offset, offset + 5):
                expected[row * 10 + column] = action

    assert standard_cells == expected
    cross_indices = {(6 * 10) + 3, (6 * 10) + 4}
    for index, (standard_cell, conditioned_cell) in enumerate(
        zip(standard_cells, conditioned_cells, strict=True)
    ):
        if index in cross_indices:
            assert standard_cell == masked
            assert conditioned_cell == cross
        else:
            assert conditioned_cell == standard_cell

    off_diagonal = [
        (row * 10) + column
        for row in range(10)
        for column in range(10)
        if (row < 5 <= column) or (column < 5 <= row)
    ]
    assert len(off_diagonal) == 50
    assert all(standard_cells[index] == masked for index in off_diagonal)
    assert all(
        conditioned_cells[index] == masked
        for index in off_diagonal
        if index not in cross_indices
    )

    mobile_css = css.split("@media (max-width: 760px)", 1)[1].split(
        "@media (prefers-reduced-motion: reduce)", 1
    )[0]
    assert ".attention-comparison" in mobile_css
    assert "grid-template-columns: 1fr" in mobile_css


def test_readme_leads_with_project_page_and_has_current_citation() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    normalized_readme = " ".join(readme.split())

    assert readme.index("Project-Page") < readme.index("GitHub-Code")
    runtime = readme.index("## Runtime layout")
    citation = readme.index("## Citation")
    license_heading = readme.index("## License")
    assert runtime < citation < license_heading
    assert readme[runtime:citation].rstrip().endswith("```")
    assert "24 kitchen manipulation tasks" in normalized_readme
    assert "50 trials per task" in normalized_readme
    assert "average success" in normalized_readme
    current_results = readme.split("## Current results", 1)[1].split(
        "## Runtime layout", 1
    )[0]
    expected_tables = (
        """
| Method | Long | Spatial | Goal | Object | Average ↑ |
|---|---:|---:|---:|---:|---:|
| OpenVLA | 53.7 | 84.7 | 79.2 | 88.4 | 76.5 |
| π₀ | 85.2 | 96.8 | 95.8 | 98.8 | 94.1 |
| π₀.₅ | 92.4 | <u>98.8</u> | <u>98.0</u> | 98.2 | 96.9 |
| Motus | **97.6** | 96.8 | 96.6 | <u>99.8</u> | 97.7 |
| Fast-WAM | 95.2 | 98.2 | 97.0 | **100.0** | 97.6 |
| FastWAM-Joint-CD | <u>97.20</u> | **99.60** | **98.60** | **100.00** | **98.85** |
| FastWAM-RTC | 58.40 | 76.20 | 77.00 | 83.40 | 73.75 |
| Stream-WAM (Ours) | 96.60 | <u>98.80</u> | 97.40 | **100.00** | <u>98.20</u> |
| Stream-WAM w/o Action Conditioning | 94.40 | 96.40 | 96.60 | 97.60 | 96.25 |
| Stream-WAM w/o Slot Encoder | 95.60 | 98.40 | 96.80 | <u>99.80</u> | 97.65 |
        """.strip(),
        """
| Method | Clean ↑ | Random ↑ | Total ↑ |
|---|---:|---:|---:|
| π₀ | 65.92 | 58.40 | 62.2 |
| π₀.₅ | 82.74 | 76.76 | 79.8 |
| Motus | <u>88.66</u> | 87.02 | <u>87.8</u> |
| Motus from WAN2.2 | 77.56 | 77.00 | 77.3 |
| Fast-WAM | **91.88** | **91.78** | **91.8** |
| StarWAM-Joint | 84.8 | 86.0 | 85.4 |
| StarWAM-CD | 79.0 | 79.2 | 79.1 |
| Stream-WAM (Ours) | 87.2 | <u>88.8</u> | 87.6 |
        """.strip(),
        """
| Method | Average Success ↑ |
|---|---:|
| π₀.₅ | 41.4% |
| π₀-FAST | 61.2% |
| π₀ | 62.5% |
| Cosmos Policy | 67.1% |
| X-WAM | **75.42%** |
| X-WAM-CD | 75.33% |
| Stream-WAM (Ours) | <u>75.35%</u> |
        """.strip(),
        """
| Benchmark | Method | Chunk Time | Episode Time |
|---|---|---:|---:|
| LIBERO | FastWAM | 493.0 ms | 16.31 s Long / 8.25 s Short |
| LIBERO | FastWAM-Joint-CD | 114.2 ms | 6.89 s Long / 3.74 s Short |
| LIBERO | FastWAM-RTC | 142.3 ms | 6.23 s Long / 3.20 s Short |
| LIBERO | Stream-WAM | 41.0 ms | 5.36 s Long / 3.15 s Short |
| LIBERO | Stream-WAM w/o Action Conditioning | 35.1 ms | 5.20 s Long / 2.92 s Short |
| LIBERO | Stream-WAM w/o Slot Encoder | 36.3 ms | 5.31 s Long / 3.01 s Short |
| RoboTwin 2.0 | StarWAM-Joint | 190.17 ms | 110.22 s |
| RoboTwin 2.0 | StarWAM-CD | 81.21 ms | 102.59 s |
| RoboTwin 2.0 | Stream-WAM | 47.09 ms | 77.48 s |
| RoboCasa | X-WAM | 374.07 ms | 17.36 s |
| RoboCasa | X-WAM-CD | 134.37 ms | 13.04 s |
| RoboCasa | Stream-WAM | 115.98 ms | 9.49 s |
        """.strip(),
    )
    for expected_table in expected_tables:
        assert expected_table in current_results
    for obsolete in (
        "75.83",
        "189.3",
        "81.6",
        "112.2",
        "504.00",
        "37.31",
        "135.21",
        "33.60",
        "136.76",
        "11.76",
        "| StreamWAM |",
    ):
        assert obsolete not in current_results
    for field in (
        "@misc{denglab2026streamwam,",
        "title        = {Stream-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation}",
        "author       = {{DENG Lab}}",
        "year         = {2026}",
        "organization = {Shanghai Jiao Tong University}",
        "url          = {https://sjtu-deng-lab.github.io/StreamWAM/}",
    ):
        assert field in readme


def test_discussion_ends_with_copyable_citation_and_open_source_actions() -> None:
    _, html = parse_page()
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    discussion_start = html.index('<section class="article-section reading-column" id="discussion"')
    conclusion_start = html.index("The third is about the scope of this preview.", discussion_start)
    resources_start = html.find('<section class="project-endmatter" id="resources"', conclusion_start)
    assert resources_start > conclusion_start
    resources_end = html.index("</section>", resources_start)
    discussion_end = html.index("</section>", resources_end + len("</section>"))
    resources_html = html[resources_start:resources_end]

    assert discussion_start < conclusion_start < resources_start < resources_end < discussion_end
    assert '<h2 id="citation-title">Citation</h2>' in resources_html
    assert 'href="https://github.com/SJTU-DENG-Lab/StreamWAM"' in resources_html
    assert 'href="https://huggingface.co/SJTU-DENG-Lab/StreamWAM"' in resources_html
    assert ">Code ↗</a>" in resources_html
    assert ">Models ↗</a>" in resources_html
    assert '<button class="citation-copy" type="button" aria-label="Copy BibTeX citation">Copy</button>' in resources_html
    assert '<pre><code id="citation-bibtex">' in resources_html
    citation = resources_html.split('<pre><code id="citation-bibtex">', 1)[1].split("</code></pre>", 1)[0].strip()
    assert citation == """@misc{denglab2026streamwam,
  title        = {Stream-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation},
  author       = {{DENG Lab}},
  year         = {2026},
  howpublished = {Project page},
  organization = {Shanghai Jiao Tong University},
  url          = {https://sjtu-deng-lab.github.io/StreamWAM/}
}"""

    for obsolete_copy in (
        "article-section resources reading-column",
        "Open source.",
        "Model lineage.",
        "Acknowledgements.",
        "Paper · Coming Soon",
        "Rollout film · Coming Soon",
    ):
        assert obsolete_copy not in resources_html

    assert ".resources" not in css
    citation_rule = re.search(r"\.citation-card pre\s*\{([^}]*)\}", css)
    assert citation_rule is not None
    assert "overflow-x: auto" in citation_rule.group(1)
    assert "border-radius" in citation_rule.group(1)
    assert re.search(r"background:\s*#[0-9a-fA-F]{6}", citation_rule.group(1))


def test_citation_copy_button_uses_clipboard_and_local_preview_fallback() -> None:
    subprocess.run(["node", str(CITATION_COPY_HARNESS_PATH)], check=True, cwd=REPO_ROOT)


def test_page_is_a_linear_text_first_research_article() -> None:
    parser, html = parse_page()
    article_ids = [
        attrs["id"]
        for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id") in ARTICLE_SECTION_IDS
    ]

    assert parser.tags.count("article") >= 1
    assert article_ids == list(ARTICLE_SECTION_IDS)
    assert parser.article_paragraph_count >= 35
    assert "chapter-index" not in html
    assert "future-slots" not in html


def test_article_opens_with_a_compact_abstract_without_a_contents_menu() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")

    assert "Abstract" in visible_text
    assert "World Action Models (WAMs) jointly generate future visual observations and robot actions" in visible_text
    assert "synchronous execution leaves the robot idle during inference" in visible_text
    assert "Actions already committed to execution condition future video generation" in visible_text
    assert "guides a consistent action continuation" in visible_text
    assert "A streaming model should know what the robot is already doing." not in visible_text
    assert "In this article" not in visible_text
    assert "lower is better" not in visible_text.casefold()
    assert 'aria-label="Article contents"' not in html
    assert "article-toc" not in html
    heading_rule = re.search(r"\.article-header h2\s*\{([^}]*)\}", css)
    assert heading_rule is not None
    assert re.search(r"font-size:\s*clamp\([^;]*1\.45rem\s*\)", heading_rule.group(1))


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
        attrs.get("id"): attrs.get("aria-label") or attrs.get("aria-labelledby")
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
        "resources": "citation-title",
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
    benchmark_order = [
        attrs["id"]
        for tag, attrs in parser.attributes
        if tag == "section" and attrs.get("id", "").startswith("benchmark-")
    ]
    assert benchmark_order == ["benchmark-libero", "benchmark-robotwin", "benchmark-robocasa"]
    assert parser.tags.count("table") == 4
    assert parser.captions == [
        "LIBERO success results",
        "RoboTwin 2.0 clean and randomized results",
        "RoboCasa 24-task average success results",
        "Exact latency values shown in the figure",
    ]
    assert "data-tabs" not in html
    assert "data-panel" not in html
    assert 'role="tab"' not in html


def test_results_narrative_reports_protocol_and_speedups() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)

    for fact in (
        "FastWAM-Joint",
        "X-WAM on RoboCasa",
        "StarWAM on RoboTwin 2.0",
        "four NVIDIA H100 GPUs",
        "CD refers to one-step consistency distillation",
        "We also conduct ablation studies on Stream-WAM by removing action conditioning or the slot encoder to evaluate the contribution of each component",
        "12.0× on LIBERO",
        "4.0× on RoboTwin 2.0",
        "3.2× on RoboCasa relative to X-WAM",
        "3.0× and 2.6× on long and short LIBERO tasks",
        "1.4× on RoboTwin 2.0",
        "1.8× on RoboCasa",
    ):
        assert fact in visible_text

    for redundant_summary in (
        "Stream-WAM reaches 98.20% average success",
        "Stream-WAM reaches 87.6 total success",
        "Stream-WAM reports 75.35% average task success",
        "Every bar is labeled with its reported value",
        "Consistency Distillation",
        "Stream-WAM w/o Action Conditioning removes Action Conditioning",
        "Published policy results from",
    ):
        assert redundant_summary not in visible_text

    assert 'class="benchmark-reading"' not in html


def test_result_tables_use_readable_tabular_sans_serif_numbers() -> None:
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")
    numeric_rule = re.search(r"tbody td\s*\{([^}]*)\}", css)

    assert numeric_rule is not None
    assert "font-family: inherit" in numeric_rule.group(1)
    assert "monospace" not in numeric_rule.group(1)
    assert "font-variant-numeric: tabular-nums" in css
    assert ".benchmark-reading" not in css


def test_benchmark_tables_and_protocols_match_the_authoritative_results() -> None:
    parsed = parse_benchmarks()

    assert parsed.rows == {
        "benchmark-libero": [
            ["Method", "Long", "Spatial", "Goal", "Object", "Average ↑"],
            ["OpenVLA", "53.7", "84.7", "79.2", "88.4", "76.5"],
            ["π₀", "85.2", "96.8", "95.8", "98.8", "94.1"],
            ["π₀.₅", "92.4", "98.8", "98.0", "98.2", "96.9"],
            ["Motus", "97.6", "96.8", "96.6", "99.8", "97.7"],
            ["Fast-WAM", "95.2", "98.2", "97.0", "100.0", "97.6"],
            ["FastWAM-Joint-CD", "97.20", "99.60", "98.60", "100.00", "98.85"],
            ["FastWAM-RTC", "58.40", "76.20", "77.00", "83.40", "73.75"],
            ["Stream-WAM (Ours)", "96.60", "98.80", "97.40", "100.00", "98.20"],
            ["Stream-WAM w/o Action Conditioning", "94.40", "96.40", "96.60", "97.60", "96.25"],
            ["Stream-WAM w/o Slot Encoder", "95.60", "98.40", "96.80", "99.80", "97.65"],
        ],
        "benchmark-robocasa": [
            ["Method", "Average Success ↑"],
            ["π₀.₅", "41.4%"],
            ["π₀-FAST", "61.2%"],
            ["π₀", "62.5%"],
            ["Cosmos Policy", "67.1%"],
            ["X-WAM", "75.42%"],
            ["X-WAM-CD", "75.33%"],
            ["Stream-WAM (Ours)", "75.35%"],
        ],
        "benchmark-robotwin": [
            ["Method", "Clean ↑", "Random ↑", "Total ↑"],
            ["π₀", "65.92", "58.40", "62.2"],
            ["π₀.₅", "82.74", "76.76", "79.8"],
            ["Motus", "88.66", "87.02", "87.8"],
            ["Motus from WAN2.2", "77.56", "77.00", "77.3"],
            ["Fast-WAM", "91.88", "91.78", "91.8"],
            ["StarWAM-Joint", "84.8", "86.0", "85.4"],
            ["StarWAM-CD", "79.0", "79.2", "79.1"],
            ["Stream-WAM (Ours)", "87.2", "88.8", "87.6"],
        ],
    }

    assert parsed.cell_styles == {
        "benchmark-libero": [
            ["", "", "", "", "", ""],
            ["", "", "", "", "", ""],
            ["", "", "", "", "", ""],
            ["", "", "second", "second", "", ""],
            ["", "best", "", "", "second", ""],
            ["", "", "", "", "best", ""],
            ["", "second", "best", "best", "best", "best"],
            ["", "", "", "", "", ""],
            ["", "", "second", "", "best", "second"],
            ["", "", "", "", "", ""],
            ["", "", "", "", "second", ""],
        ],
        "benchmark-robocasa": [
            ["", ""],
            ["", ""],
            ["", ""],
            ["", ""],
            ["", ""],
            ["", "best"],
            ["", ""],
            ["", "second"],
        ],
        "benchmark-robotwin": [
            ["", "", "", ""],
            ["", "", "", ""],
            ["", "", "", ""],
            ["", "second", "", "second"],
            ["", "", "", ""],
            ["", "best", "best", "best"],
            ["", "", "", ""],
            ["", "", "", ""],
            ["", "", "second", ""],
        ],
    }

    section_text = {name: " ".join(parts) for name, parts in parsed.section_text.items()}
    assert all(fragment in section_text["benchmark-libero"] for fragment in ("four suites", "10 tasks per suite", "50 trials per task", "long and short tasks"))
    assert all(fragment in section_text["benchmark-robocasa"] for fragment in ("24 kitchen manipulation tasks", "50 trials per task", "average success"))
    assert all(fragment in section_text["benchmark-robotwin"] for fragment in ("50 tasks", "100 rollout episodes per task", "Clean", "Random", "domain-randomization"))


def test_page_embeds_two_wide_latency_figures_for_all_three_benchmarks() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)
    latency_images = [
        attrs
        for tag, attrs in parser.attributes
        if tag == "img" and "latency-plot" in attrs.get("class", "").split()
    ]
    latency_links = [
        attrs["href"]
        for tag, attrs in parser.attributes
        if tag == "a" and "latency-image-link" in attrs.get("class", "").split()
    ]

    assert latency_images == [
        {
            "class": "latency-plot",
            "src": "assets/stream-wam-chunk-time.png?v=robocasa-latency-20260826",
            "alt": "Chunk-time comparison for LIBERO, RoboTwin 2.0, and RoboCasa.",
            "width": "2400",
            "height": "900",
            "aria-describedby": "chunk-time-caption latency-data-caption",
        },
        {
            "class": "latency-plot",
            "src": "assets/stream-wam-episode-time.png?v=robocasa-latency-20260826",
            "alt": "Episode-time comparison for LIBERO, RoboTwin 2.0, and RoboCasa.",
            "width": "2400",
            "height": "900",
            "aria-describedby": "episode-time-caption latency-data-caption",
        },
    ]
    assert latency_links == [
        "assets/stream-wam-chunk-time.png?v=robocasa-latency-20260826",
        "assets/stream-wam-episode-time.png?v=robocasa-latency-20260826",
    ]
    assert "Episode Time" in visible_text
    assert "Total Time" not in visible_text
    assert "latency-bar" not in html
    assert "Chunk Time comparison for LIBERO, RoboTwin 2.0, and RoboCasa" in visible_text
    assert "Episode Time comparison for LIBERO, RoboTwin 2.0, and RoboCasa" in visible_text
    assert html.count('class="latency-viewport"') == 2


def test_static_latency_figure_has_an_accessible_exact_data_table() -> None:
    parsed = parse_latency_data()

    assert parsed.is_contained_by_sr_only
    assert parsed.rows == [
        ["Benchmark", "Method", "Chunk Time", "Episode Time"],
        ["LIBERO", "FastWAM", "493.0 ms", "16.31 s Long / 8.25 s Short"],
        ["LIBERO", "FastWAM-Joint-CD", "114.2 ms", "6.89 s Long / 3.74 s Short"],
        ["LIBERO", "FastWAM-RTC", "142.3 ms", "6.23 s Long / 3.20 s Short"],
        ["LIBERO", "Stream-WAM", "41.0 ms", "5.36 s Long / 3.15 s Short"],
        ["LIBERO", "Stream-WAM w/o Action Conditioning", "35.1 ms", "5.20 s Long / 2.92 s Short"],
        ["LIBERO", "Stream-WAM w/o Slot Encoder", "36.3 ms", "5.31 s Long / 3.01 s Short"],
        ["RoboTwin 2.0", "StarWAM-Joint", "190.17 ms", "110.22 s"],
        ["RoboTwin 2.0", "StarWAM-CD", "81.21 ms", "102.59 s"],
        ["RoboTwin 2.0", "Stream-WAM", "47.09 ms", "77.48 s"],
        ["RoboCasa", "X-WAM", "374.07 ms", "17.36 s"],
        ["RoboCasa", "X-WAM-CD", "134.37 ms", "13.04 s"],
        ["RoboCasa", "Stream-WAM", "115.98 ms", "9.49 s"],
    ]
    robocasa_data = " ".join(
        cell for row in parsed.rows if row[0] == "RoboCasa" for cell in row
    )
    for obsolete in ("504.00", "37.31", "135.21", "33.60", "136.76", "11.76"):
        assert obsolete not in robocasa_data


def test_latency_generator_writes_two_wide_nonempty_pngs(tmp_path: Path) -> None:
    subprocess.run(
        [sys.executable, str(LATENCY_GENERATOR_PATH), "--output-dir", str(tmp_path)],
        check=True,
        cwd=REPO_ROOT,
    )

    outputs = [
        tmp_path / "stream-wam-chunk-time.png",
        tmp_path / "stream-wam-episode-time.png",
    ]
    for output in outputs:
        contents = output.read_bytes()
        assert contents.startswith(b"\x89PNG\r\n\x1a\n")
        assert len(contents) > 50_000
        assert contents == (PAGE_ROOT / "assets" / output.name).read_bytes()
        with Image.open(output) as image:
            assert image.size == (2400, 900)


def test_latency_generator_uses_the_authoritative_values() -> None:
    spec = importlib.util.spec_from_file_location("latency_figure", LATENCY_GENERATOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.LIBERO_CHUNK == (493.0, 114.2, 142.3, 41.0, 35.1, 36.3)
    assert module.LIBERO_LONG == (16.31, 6.89, 6.23, 5.36, 5.20, 5.31)
    assert module.LIBERO_SHORT == (8.25, 3.74, 3.20, 3.15, 2.92, 3.01)
    assert module.ROBOTWIN_CHUNK == (190.17, 81.21, 47.09)
    assert module.ROBOTWIN_EPISODE == (110.22, 102.59, 77.48)
    assert module.ROBOCASA_CHUNK == (374.07, 134.37, 115.98)
    assert module.ROBOCASA_EPISODE == (17.36, 13.04, 9.49)
    assert module.LIBERO_CHUNK_YMAX == 520
    assert module.ROBOCASA_CHUNK_YMAX == 410
    assert module.ROBOCASA_EPISODE_YMAX == 20
    assert tuple(label.replace("\n", " ") for label in module.LIBERO_METHODS) == (
        "FastWAM",
        "Joint-CD",
        "RTC",
        "Stream-WAM",
        "Stream-WAM w/o Action Conditioning",
        "Stream-WAM w/o Slot Encoder",
    )

    figure, axes = module._new_figure("Chunk Time")
    module._style_axis(axes[0], ylabel="Milliseconds", title="LIBERO")
    try:
        assert figure._suptitle.get_fontsize() == 14
        assert [text.get_text() for text in figure.texts] == [
            "Chunk Time",
            "Stream-WAM highlighted in teal  ·  hatched bars are Stream-WAM ablations",
        ]
        assert "LOWER IS BETTER" not in [text.get_text() for text in axes[0].texts]
    finally:
        module.plt.close(figure)


def test_latency_figures_scroll_inside_their_mobile_viewports() -> None:
    css = (PAGE_ROOT / "styles.css").read_text()
    viewport_rule = re.search(r"\.latency-viewport\s*\{([^}]*)\}", css)
    mobile_rule = re.search(r"@media \(max-width: 760px\)\s*\{(.*)\n\}", css, re.DOTALL)

    assert viewport_rule is not None
    assert re.search(r"overflow-x:\s*auto", viewport_rule.group(1))
    assert mobile_rule is not None
    assert re.search(r"\.latency-image-link\s*\{[^}]*min-width:\s*900px", mobile_rule.group(1))
