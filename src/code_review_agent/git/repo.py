from __future__ import annotations

import difflib
import subprocess
from dataclasses import dataclass
from pathlib import Path


class GitError(RuntimeError):
    pass


@dataclass(frozen=True)
class TreeEntry:
    path: str
    blob_sha: str
    mode: str
    size: int


@dataclass(frozen=True)
class NameStatus:
    status: str
    old_path: str | None
    new_path: str | None


@dataclass(frozen=True)
class CommitInfo:
    sha: str
    parents: list[str]
    date: str
    subject: str
    body: str


@dataclass(frozen=True)
class GrepMatch:
    path: str
    line: int
    text: str


FIELD_SEP = "\x1f"
RECORD_SEP = "\x1e"


class GitRepo:
    def __init__(self, path: str | Path):
        start = Path(path).expanduser().resolve()
        top = self._run_raw(["git", "-C", str(start), "rev-parse", "--show-toplevel"], timeout=15)
        self.root = Path(top.decode().strip())
        self._blob_cache: dict[str, bytes] = {}
        self._tree_cache: dict[str, dict[str, TreeEntry]] = {}

    def _run_raw(self, args: list[str], timeout: float = 120, input_bytes: bytes | None = None) -> bytes:
        completed = subprocess.run(args, input=input_bytes, capture_output=True, timeout=timeout)
        if completed.returncode != 0:
            message = completed.stderr.decode("utf-8", "replace").strip()
            raise GitError(f"{' '.join(args[:4])}...: {message}")
        return completed.stdout

    def git_bytes(self, *args: str, timeout: float = 120, input_bytes: bytes | None = None) -> bytes:
        command = ["git", "-C", str(self.root), "-c", "core.quotePath=false", *args]
        return self._run_raw(command, timeout=timeout, input_bytes=input_bytes)

    def git(self, *args: str, timeout: float = 120) -> str:
        return self.git_bytes(*args, timeout=timeout).decode("utf-8", "replace")

    def rev_parse(self, ref: str) -> str:
        return self.git("rev-parse", "--verify", f"{ref}^{{commit}}").strip()

    def ref_exists(self, ref: str) -> bool:
        completed = subprocess.run(
            ["git", "-C", str(self.root), "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
            capture_output=True,
            timeout=15,
        )
        return completed.returncode == 0

    def merge_base(self, first: str, second: str) -> str:
        return self.git("merge-base", first, second).strip()

    def default_target(self) -> str:
        completed = subprocess.run(
            ["git", "-C", str(self.root), "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"],
            capture_output=True,
            timeout=15,
        )
        if completed.returncode == 0:
            ref = completed.stdout.decode().strip()
            return ref.removeprefix("refs/remotes/")
        for candidate in ("origin/main", "origin/master", "main", "master"):
            if self.ref_exists(candidate):
                return candidate
        raise GitError("could not find a target branch; pass --base")

    def root_commit_id(self) -> str:
        roots = self.git("rev-list", "--max-parents=0", "HEAD").split()
        return sorted(roots)[0] if roots else "empty"

    def list_tree(self, sha: str) -> dict[str, TreeEntry]:
        output = self.git_bytes("ls-tree", "-r", "-z", "-l", "--full-tree", sha)
        entries: dict[str, TreeEntry] = {}
        for record in output.split(b"\x00"):
            if not record:
                continue
            meta, path_bytes = record.split(b"\t", 1)
            mode, object_type, object_sha, size_text = meta.decode().split()
            if object_type != "blob" or mode == "120000":
                continue
            path = path_bytes.decode("utf-8", "replace")
            size = int(size_text) if size_text.isdigit() else 0
            entries[path] = TreeEntry(path=path, blob_sha=object_sha, mode=mode, size=size)
        return entries

    def read_blobs(self, shas: list[str]) -> dict[str, bytes]:
        result: dict[str, bytes] = {}
        missing = []
        for sha in shas:
            if sha in self._blob_cache:
                result[sha] = self._blob_cache[sha]
            elif sha not in missing:
                missing.append(sha)
        batch_size = 400
        for start in range(0, len(missing), batch_size):
            batch = missing[start : start + batch_size]
            payload = ("\n".join(batch) + "\n").encode()
            output = self.git_bytes("cat-file", "--batch", input_bytes=payload, timeout=300)
            position = 0
            for _ in batch:
                header_end = output.index(b"\n", position)
                header = output[position:header_end].decode().split()
                if len(header) < 3 or header[1] == "missing":
                    position = header_end + 1
                    continue
                object_sha = header[0]
                size = int(header[2])
                body_start = header_end + 1
                body = output[body_start : body_start + size]
                result[object_sha] = body
                position = body_start + size + 1
        return result

    def read_blob(self, sha: str) -> bytes:
        if sha in self._blob_cache:
            return self._blob_cache[sha]
        data = self.git_bytes("cat-file", "blob", sha, timeout=60)
        if len(self._blob_cache) > 256:
            self._blob_cache.clear()
        self._blob_cache[sha] = data
        return data

    def read_text(self, sha: str) -> str:
        return self.read_blob(sha).decode("utf-8", "replace")

    def file_at(self, commit: str, path: str) -> str | None:
        tree = self.list_tree_cached(commit)
        entry = tree.get(path)
        if entry is None:
            return None
        return self.read_text(entry.blob_sha)

    def list_tree_cached(self, sha: str) -> dict[str, TreeEntry]:
        if sha not in self._tree_cache:
            if len(self._tree_cache) >= 8:
                self._tree_cache.clear()
            self._tree_cache[sha] = self.list_tree(sha)
        return self._tree_cache[sha]

    def name_status(self, base: str, head: str) -> list[NameStatus]:
        output = self.git("diff", "--name-status", "-z", "-M", "--no-ext-diff", base, head)
        tokens = output.split("\x00")
        entries: list[NameStatus] = []
        index = 0
        while index < len(tokens):
            status = tokens[index]
            if not status:
                index += 1
                continue
            code = status[0]
            if code in ("R", "C"):
                old_path = tokens[index + 1]
                new_path = tokens[index + 2]
                entries.append(NameStatus(status=code, old_path=old_path, new_path=new_path))
                index += 3
                continue
            path = tokens[index + 1]
            if code == "A":
                entries.append(NameStatus(status="A", old_path=None, new_path=path))
            elif code == "D":
                entries.append(NameStatus(status="D", old_path=path, new_path=None))
            else:
                entries.append(NameStatus(status="M", old_path=path, new_path=path))
            index += 2
        return entries

    def diff_text(self, base: str, head: str, paths: list[str] | None = None, context: int = 3) -> str:
        args = ["diff", "--no-color", "--no-ext-diff", "-M", f"-U{context}", base, head]
        if paths:
            args.append("--")
            args.extend(paths)
        return self.git(*args, timeout=180)

    def blob_diff_text(self, old_blob: str | None, new_blob: str | None, path: str, context: int = 3) -> str:
        if old_blob is None or new_blob is None:
            old_text = self.read_text(old_blob) if old_blob else ""
            new_text = self.read_text(new_blob) if new_blob else ""
            return synthetic_unified_diff(old_text, new_text, path, context, old_blob is None, new_blob is None)
        raw = self.git("diff", "--no-color", "--no-ext-diff", f"-U{context}", old_blob, new_blob, timeout=60)
        return rewrite_blob_diff_paths(raw, path)

    def commits_matching(self, pattern: str, limit: int, rev: str = "HEAD") -> list[CommitInfo]:
        output = self.git(
            "log",
            rev,
            "--no-merges",
            "-i",
            "-E",
            f"--grep={pattern}",
            f"-n{limit}",
            f"--format=%H{FIELD_SEP}%P{FIELD_SEP}%ad{FIELD_SEP}%s{FIELD_SEP}%b{RECORD_SEP}",
            "--date=short",
            timeout=180,
        )
        return parse_commit_records(output)

    def commit_info(self, sha: str) -> CommitInfo:
        output = self.git(
            "log", "-n1", f"--format=%H{FIELD_SEP}%P{FIELD_SEP}%ad{FIELD_SEP}%s{FIELD_SEP}%b{RECORD_SEP}", "--date=short", sha
        )
        records = parse_commit_records(output)
        if not records:
            raise GitError(f"unknown commit {sha}")
        return records[0]

    def show_with_function_context(self, sha: str, context: int = 3) -> str:
        return self.git(
            "-c",
            r"diff.python.xfuncname=^[ \t]*((class|(async[ \t]+)?def)[ \t].*)$",
            "-c",
            r"diff.default.xfuncname=^[ \t]*((class|(async[ \t]+)?def)[ \t].*)$",
            "show",
            "--no-color",
            "--no-ext-diff",
            f"-U{context}",
            "--format=",
            sha,
            timeout=120,
        )

    def grep(self, rev: str, pattern: str, pathspecs: list[str] | None, max_matches: int, timeout: float) -> list[GrepMatch]:
        args = ["grep", "-n", "-I", "-E", "--no-color", "-e", pattern, rev]
        if pathspecs:
            args.append("--")
            args.extend(pathspecs)
        completed = subprocess.run(
            ["git", "-C", str(self.root), "-c", "core.quotePath=false", *args],
            capture_output=True,
            timeout=timeout,
        )
        if completed.returncode not in (0, 1):
            raise GitError(completed.stderr.decode("utf-8", "replace").strip())
        matches: list[GrepMatch] = []
        prefix = f"{rev}:"
        for raw_line in completed.stdout.decode("utf-8", "replace").splitlines():
            line_text = raw_line.removeprefix(prefix)
            parts = line_text.split(":", 2)
            if len(parts) < 3 or not parts[1].isdigit():
                continue
            matches.append(GrepMatch(path=parts[0], line=int(parts[1]), text=parts[2]))
            if len(matches) >= max_matches:
                break
        return matches

    def log_for_path(self, rev: str, path: str, limit: int) -> list[CommitInfo]:
        output = self.git(
            "log",
            rev,
            f"-n{limit}",
            f"--format=%H{FIELD_SEP}%P{FIELD_SEP}%ad{FIELD_SEP}%s{FIELD_SEP}{RECORD_SEP}",
            "--date=short",
            "--",
            path,
            timeout=60,
        )
        return parse_commit_records(output)


def parse_commit_records(output: str) -> list[CommitInfo]:
    commits: list[CommitInfo] = []
    for record in output.split(RECORD_SEP):
        record = record.strip("\n")
        if not record:
            continue
        fields = record.split(FIELD_SEP)
        while len(fields) < 5:
            fields.append("")
        commits.append(
            CommitInfo(
                sha=fields[0].strip(),
                parents=fields[1].split(),
                date=fields[2],
                subject=fields[3],
                body=fields[4].strip(),
            )
        )
    return commits


def rewrite_blob_diff_paths(raw: str, path: str) -> str:
    lines = raw.splitlines()
    output = []
    in_header = True
    for line in lines:
        if line.startswith("@@"):
            in_header = False
        if in_header and line.startswith("diff --git "):
            output.append(f"diff --git a/{path} b/{path}")
        elif in_header and line.startswith("--- "):
            output.append(f"--- a/{path}")
        elif in_header and line.startswith("+++ "):
            output.append(f"+++ b/{path}")
        else:
            output.append(line)
    return "\n".join(output) + ("\n" if output else "")


def synthetic_unified_diff(old_text: str, new_text: str, path: str, context: int, is_new: bool, is_deleted: bool) -> str:
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    header = [f"diff --git a/{path} b/{path}"]
    if is_new:
        header.append("new file mode 100644")
    if is_deleted:
        header.append("deleted file mode 100644")
    old_label = "/dev/null" if is_new else f"a/{path}"
    new_label = "/dev/null" if is_deleted else f"b/{path}"
    body = list(difflib.unified_diff(old_lines, new_lines, old_label, new_label, n=context, lineterm=""))
    return "\n".join(header + body) + "\n"
