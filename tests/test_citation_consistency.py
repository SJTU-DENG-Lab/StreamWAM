from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CITATION = """@misc{huang2026streamingwam,
  title        = {Streaming-WAM: Streaming Your World-Action Model for Real-Time Robot Manipulation},
  author       = {Xuyao Huang and Yixuan Wang and Zengyao Ye and Haoran Wen and Zhijie Deng},
  year         = {2026},
  howpublished = {Project page},
  organization = {Shanghai Jiao Tong University and Li Auto Inc.},
  url          = {https://sjtu-deng-lab.github.io/Streaming-WAM/}
}"""


def test_public_citation_is_identical_across_all_release_surfaces() -> None:
    public_files = (
        ROOT / "README.md",
        ROOT / "docs" / "index.html",
        ROOT / "docs" / "huggingface" / "README.md",
    )

    for path in public_files:
        contents = path.read_text(encoding="utf-8")
        assert contents.count(CITATION) == 1, f"citation mismatch in {path.relative_to(ROOT)}"
        assert "denglab2026streamingwam" not in contents
        assert "author       = {{DENG Lab}}" not in contents
