from __future__ import annotations

from html.parser import HTMLParser
import importlib.util
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import urlparse

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_ROOT = REPO_ROOT / "docs"
INDEX_PATH = PAGE_ROOT / "index.html"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pages.yml"
LATENCY_GENERATOR_PATH = PAGE_ROOT / "generate_latency_figure.py"
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
    assert "Ours" not in html
    assert "StreamWAM" not in visible_text

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
        "11.76 s",
        "96.60",
        "98.80",
        "97.40",
        "100.00",
        "75.35%",
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
    assert "Stream-WAM conditions the visual future on the action already underway" in visible_text
    assert "inference-time RTC" in visible_text
    assert "prefix-conditioned" in visible_text
    assert parser.eyebrow == "Stream-WAM"
    assert "Task performance." in visible_text
    assert "Inference efficiency." in visible_text
    assert "Stream-WAM reaches 98.20% average success" in visible_text
    assert "Stream-WAM reports 75.35% average task success" in visible_text
    assert "Stream-WAM reaches 87.6 total success" in visible_text
    assert "broader analysis, limitations, and failure cases" in visible_text
    assert "Model lineage." in visible_text
    assert "Code and models are available now" in visible_text
    assert "Action-conditioned attention" in visible_text
    assert theme_colors == ["#f7f5ef"]
    assert not any(token in html.casefold() for token in ("todo", "tbd", "lorem ipsum"))


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
    assert 'class="execution-rail execution-continuous"' in pipeline_markup
    assert 'aria-describedby="pipeline-description pipeline-caption"' in pipeline_markup
    assert 'class="sr-only" id="pipeline-description"' in pipeline_markup
    assert not re.search(r"\b\d+\s*(?:actions?|ms|seconds?)\b", pipeline_markup, re.I)
    for label in ("World-Action Prediction", "Robot Execution", "Committed Actions"):
        assert label in pipeline_markup
    for old_label in ("Model update", "Robot motion", "Action prefix"):
        assert old_label not in pipeline_markup
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
    assert stream_row.index('class="generation-window generation-one"') < stream_row.index(
        'class="execution-rail execution-continuous"'
    )
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

    assert 'class="attention-matrix two-chunk-matrix"' in pipeline_markup
    assert pipeline_markup.count('class="chunk-group') >= 2
    assert pipeline_markup.count('class="matrix-cell visual-token') >= 3
    assert pipeline_markup.count('class="matrix-cell action-token') >= 3
    assert pipeline_markup.count('class="matrix-cell cross-chunk-condition') >= 1
    assert "Action-conditioned attention" in pipeline_markup
    assert "Chunk k" in pipeline_markup
    assert "Chunk k+1" in pipeline_markup
    assert "Future Visual" in pipeline_markup
    assert "Committed Actions" in pipeline_markup
    assert "Queries" in pipeline_markup
    assert "Keys" in pipeline_markup
    assert "Allowed" in pipeline_markup
    assert "Masked" in pipeline_markup
    assert "Stream-WAM" in pipeline_markup
    assert "committed actions from Chunk k condition only the future visual tokens in Chunk k+1" in pipeline_markup

    future_visual_row = (
        '<i class="matrix-cell visual-token"></i>'
        '<i class="matrix-cell cross-chunk-condition"></i>'
        '<i class="matrix-cell visual-token"></i>'
        '<i class="matrix-cell masked-cell"></i>'
    )
    next_actions_row = (
        '<i class="matrix-cell visual-token"></i>'
        '<i class="matrix-cell action-token"></i>'
        '<i class="matrix-cell visual-token"></i>'
        '<i class="matrix-cell action-token"></i>'
    )
    compact_markup = re.sub(r">\s+<", "><", pipeline_markup)
    assert future_visual_row in compact_markup
    assert next_actions_row in compact_markup

    mobile_css = css.split("@media (max-width: 760px)", 1)[1].split(
        "@media (prefers-reduced-motion: reduce)", 1
    )[0]
    assert "repeat(4,minmax(0,1fr))" in mobile_css
    assert "repeat(4,minmax(28px,1fr))" not in mobile_css


def test_readme_leads_with_project_page_and_has_current_citation() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert readme.index("Project-Page") < readme.index("GitHub-Code")
    runtime = readme.index("## Runtime layout")
    citation = readme.index("## Citation")
    license_heading = readme.index("## License")
    assert runtime < citation < license_heading
    assert readme[runtime:citation].rstrip().endswith("```")
    for field in (
        "@misc{denglab2026streamwam,",
        "title        = {Stream-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation}",
        "author       = {{DENG Lab}}",
        "year         = {2026}",
        "organization = {Shanghai Jiao Tong University}",
        "url          = {https://sjtu-deng-lab.github.io/StreamWAM/}",
    ):
        assert field in readme


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


def test_article_opens_with_a_compact_abstract_without_a_contents_menu() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)
    css = (PAGE_ROOT / "styles.css").read_text(encoding="utf-8")

    assert "Abstract" in visible_text
    assert "jointly generate visual futures and robot action chunks" in visible_text
    assert "naive asynchronous switching can create disagreement between consecutive chunks" in visible_text
    assert "action currently being executed conditions future-video generation" in visible_text
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
        "RoboCasa target benchmark results",
        "Exact latency values shown in the figure",
    ]
    assert "data-tabs" not in html
    assert "data-panel" not in html
    assert 'role="tab"' not in html


def test_benchmark_tables_and_protocols_match_the_authoritative_results() -> None:
    parsed = parse_benchmarks()

    assert parsed.rows == {
        "benchmark-libero": [
            ["Method", "LIBERO-10", "Spatial", "Goal", "Object", "Average ↑"],
            ["OpenVLA", "53.7", "84.7", "79.2", "88.4", "76.5"],
            ["π₀", "85.2", "96.8", "95.8", "98.8", "94.1"],
            ["π₀.₅", "92.4", "98.8", "98.0", "98.2", "96.9"],
            ["Motus", "97.6", "96.8", "96.6", "99.8", "97.7"],
            ["Fast-WAM", "95.2", "98.2", "97.0", "100.0", "97.6"],
            ["FastWAM-Joint-CD", "97.20", "99.60", "98.60", "100.00", "98.85"],
            ["FastWAM-RTC", "58.40", "76.20", "77.00", "83.40", "73.75"],
            ["Stream-WAM", "96.60", "98.80", "97.40", "100.00", "98.20"],
            ["Stream-WAM w/o Action Conditioning", "94.40", "96.40", "96.60", "97.60", "96.25"],
            ["Stream-WAM w/o Slot Encoder", "95.60", "98.40", "96.80", "99.80", "97.65"],
        ],
        "benchmark-robocasa": [
            ["Method", "Accuracy ↑"],
            ["X-WAM", "75.42%"],
            ["X-WAM-CD", "75.83%"],
            ["Stream-WAM", "75.35%"],
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
            ["Stream-WAM", "87.2", "88.8", "87.6"],
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
            ["", "second"],
            ["", "best"],
            ["", ""],
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
    assert all(fragment in section_text["benchmark-robocasa"] for fragment in ("50 target tasks", "50 trials per task", "average task success"))
    assert all(fragment in section_text["benchmark-robotwin"] for fragment in ("50 tasks", "100 rollout episodes per task", "Clean", "Random", "domain-randomization"))


def test_page_embeds_two_wide_latency_figures_for_all_three_benchmarks() -> None:
    parser, html = parse_page()
    visible_text = " ".join(parser.text_parts)
    latency_images = [
        attrs
        for tag, attrs in parser.attributes
        if tag == "img" and "latency-plot" in attrs.get("class", "").split()
    ]

    assert latency_images == [
        {
            "class": "latency-plot",
            "src": "assets/stream-wam-chunk-time.png",
            "alt": "Chunk-time comparison for LIBERO, RoboTwin 2.0, and RoboCasa.",
            "width": "2400",
            "height": "900",
            "aria-describedby": "chunk-time-caption latency-data-caption",
        },
        {
            "class": "latency-plot",
            "src": "assets/stream-wam-episode-time.png",
            "alt": "Episode-time comparison for LIBERO, RoboTwin 2.0, and RoboCasa.",
            "width": "2400",
            "height": "900",
            "aria-describedby": "episode-time-caption latency-data-caption",
        },
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
        ["RoboCasa", "X-WAM", "504.00 ms", "37.31 s"],
        ["RoboCasa", "X-WAM-CD", "135.21 ms", "33.60 s"],
        ["RoboCasa", "Stream-WAM", "136.76 ms", "11.76 s"],
    ]


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
    assert module.ROBOCASA_CHUNK == (504.00, 135.21, 136.76)
    assert module.ROBOCASA_EPISODE == (37.31, 33.60, 11.76)
    assert module.LIBERO_CHUNK_YMAX == 520
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
