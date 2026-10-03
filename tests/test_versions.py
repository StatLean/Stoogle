import json
import subprocess

import pytest

from stoogle.corpus import read_corpus
from stoogle.published import export_published

ORIGIN = "https://github.com/StatLean/Stat-Lean"
MATHLIB = "5e932f97dd25535344f80f9dd8da3aab83df0fe6"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


@pytest.fixture
def tagged_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "StatLean").mkdir(parents=True)
    (repo / "website/src/data").mkdir(parents=True)
    (repo / "StatLean/Example.lean").write_text("def foo : Nat := 1\n")
    (repo / "website/src/data/results.json").write_text("[]")
    (repo / "lean-toolchain").write_text("leanprover/lean4:v4.29.1\n")
    (repo / "lake-manifest.json").write_text(json.dumps({"packages": [
        {"name": "mathlib", "rev": MATHLIB, "inputRev": "v4.29.1"}]}))
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-q", "-m", "snapshot")
    git(repo, "tag", "v0.1.0")
    return repo, git(repo, "rev-parse", "HEAD")


def cached_docs(cache, commit):
    page = cache / commit / "StatLean.Example.html"
    page.parent.mkdir(parents=True)
    page.write_text(f'''<main><div class="decl" id="Example.foo"><div class="def">
    <div class="gh_link"><a href="{ORIGIN}/blob/{commit}/StatLean/Example.lean#L1-L1">source</a></div>
    <div class="decl_header"><span class="decl_kind">def</span> <span class="decl_name">Example.foo</span>
    <span class="decl_type">: Nat</span></div><p>An example.</p></div></div></main>''')


def test_import_by_tag_records_versions(tagged_repo, tmp_path):
    repo, commit = tagged_repo
    cached_docs(tmp_path / "cache", commit)
    manifest = export_published(repo, None, "https://statlean.github.io/docs", tmp_path / "cache",
                                tmp_path / "corpus", tag="v0.1.0")
    assert manifest["commit"] == commit and manifest["statlean_tag"] == "v0.1.0"
    assert manifest["lean_toolchain"] == "leanprover/lean4:v4.29.1"
    assert manifest["commit_date"] == git(repo, "show", "-s", "--format=%cs", commit)
    assert manifest["mathlib_rev"] == MATHLIB and manifest["mathlib_input_rev"] == "v4.29.1"
    records, _ = read_corpus(tmp_path / "corpus")
    assert [r["name"] for r in records] == ["Example.foo"]


def test_tag_must_match_commit(tagged_repo, tmp_path):
    repo, _ = tagged_repo
    with pytest.raises(ValueError, match="points to"):
        export_published(repo, "a" * 40, "https://statlean.github.io/docs", tmp_path / "cache",
                         tmp_path / "corpus", tag="v0.1.0")
    with pytest.raises(ValueError, match="not found"):
        export_published(repo, None, "https://statlean.github.io/docs", tmp_path / "cache",
                         tmp_path / "corpus", tag="v9.9.9")
