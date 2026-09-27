from __future__ import annotations

from dataclasses import dataclass, field

from .config import Settings
from .git.diff import Diff, FileDiff, parse_diff
from .index.files import SECRET_FILE_PATTERNS, matches_any
from .git.repo import GitRepo
from .index.builder import BuildReport, IndexBuilder, settings_for_repo
from .index.store import Store
from .index.view import View
from .llm.client import LLMClient, optional_client
from .retrieval.engine import RetrievalEngine


def withhold_secret_files(diff: Diff) -> Diff:
    files: list[FileDiff] = []
    for file_diff in diff.files:
        paths = [path for path in (file_diff.old_path, file_diff.new_path) if path]
        if any(matches_any(path, SECRET_FILE_PATTERNS) for path in paths):
            file_diff = file_diff.model_copy(update={"hunks": [], "withheld": True})
        files.append(file_diff)
    return diff.model_copy(update={"files": files})


def load_commit_diff(repo: GitRepo, base_sha: str, head_sha: str) -> Diff:
    changes = repo.name_status(base_sha, head_sha)
    expected = [(change.old_path, change.new_path, change.status) for change in changes]
    return withhold_secret_files(parse_diff(repo.diff_text(base_sha, head_sha), base_sha, head_sha, expected))


def load_override_diff(repo: GitRepo, view: View, overrides: dict[str, str | None]) -> Diff:
    sections = []
    expected = []
    for path in sorted(overrides):
        base_file = view.file(path, "base")
        old_blob = base_file.blob_sha if base_file else None
        new_blob = overrides[path]
        if old_blob == new_blob:
            continue
        sections.append(repo.blob_diff_text(old_blob, new_blob, path))
        if old_blob is None:
            expected.append((None, path, "A"))
        elif new_blob is None:
            expected.append((path, None, "D"))
        else:
            expected.append((path, path, "M"))
    return withhold_secret_files(parse_diff("".join(sections), view.base_sha, view.head_sha or "synthetic", expected))


@dataclass
class ReviewSession:
    repo: GitRepo
    settings: Settings
    store: Store
    view: View
    diff: Diff
    base_sha: str
    head_sha: str | None
    report: BuildReport | None = None
    _llm: LLMClient | None = field(default=None, repr=False)
    _llm_checked: bool = False
    _engine: RetrievalEngine | None = field(default=None, repr=False)

    @classmethod
    def open(
        cls,
        repo_path: str,
        base: str | None = None,
        head: str = "HEAD",
        use_llm: bool = True,
        progress=None,
    ) -> ReviewSession:
        repo = GitRepo(repo_path)
        settings = settings_for_repo(repo, head, use_llm)
        head_sha = repo.rev_parse(head)
        target = base or repo.default_target()
        base_sha = repo.merge_base(repo.rev_parse(target), head_sha)
        builder = IndexBuilder(repo, settings, progress=progress)
        view, report = builder.build_overlay(base_sha, head_sha)
        diff = load_commit_diff(repo, base_sha, head_sha)
        return cls(repo=repo, settings=settings, store=builder.store, view=view, diff=diff, base_sha=base_sha, head_sha=head_sha, report=report)

    @classmethod
    def synthetic(
        cls,
        repo: GitRepo,
        settings: Settings,
        store: Store,
        root_sha: str,
        overrides: dict[str, str | None],
        label: str,
        diff_overrides: dict[str, str | None] | None = None,
        progress=None,
    ) -> ReviewSession:
        builder = IndexBuilder(repo, settings, store=store, progress=progress)
        view, report = builder.build_synthetic(root_sha, overrides, label)
        diff = load_override_diff(repo, view, diff_overrides if diff_overrides is not None else overrides)
        return cls(repo=repo, settings=settings, store=store, view=view, diff=diff, base_sha=root_sha, head_sha=None, report=report)

    @property
    def llm(self) -> LLMClient | None:
        if not self._llm_checked:
            self._llm = optional_client(self.settings, self.store)
            self._llm_checked = True
        return self._llm

    @property
    def engine(self) -> RetrievalEngine:
        if self._engine is None:
            self._engine = RetrievalEngine(self.view, self.settings, self.llm)
        return self._engine

    def read_file_text(self, path: str, version: str = "current") -> str | None:
        view_file = self.view.file(path, version)
        if view_file is None:
            return None
        return self.repo.read_text(view_file.blob_sha)

    def grep_revision(self) -> str | None:
        return self.head_sha
