from __future__ import annotations

import re

from pydantic import BaseModel, Field

HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@ ?(.*)$")


class Hunk(BaseModel):
    id: str
    path: str
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    header: str
    lines: list[str] = Field(default_factory=list)

    def added_line_numbers(self) -> list[int]:
        numbers = []
        current = self.new_start
        for line in self.lines:
            if line.startswith("+"):
                numbers.append(current)
                current += 1
            elif line.startswith("-"):
                continue
            elif line.startswith("\\"):
                continue
            else:
                current += 1
        return numbers

    def removed_line_numbers(self) -> list[int]:
        numbers = []
        current = self.old_start
        for line in self.lines:
            if line.startswith("-"):
                numbers.append(current)
                current += 1
            elif line.startswith("+"):
                continue
            elif line.startswith("\\"):
                continue
            else:
                current += 1
        return numbers

    def new_range(self) -> tuple[int, int]:
        end = self.new_start + max(self.new_count, 1) - 1
        return self.new_start, end

    def old_range(self) -> tuple[int, int]:
        end = self.old_start + max(self.old_count, 1) - 1
        return self.old_start, end

    def text(self) -> str:
        header = f"@@ -{self.old_start},{self.old_count} +{self.new_start},{self.new_count} @@ {self.header}".rstrip()
        return header + "\n" + "\n".join(self.lines)


class FileDiff(BaseModel):
    old_path: str | None
    new_path: str | None
    status: str
    is_binary: bool = False
    withheld: bool = False
    hunks: list[Hunk] = Field(default_factory=list)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    def added_line_numbers(self) -> set[int]:
        numbers: set[int] = set()
        for hunk in self.hunks:
            numbers.update(hunk.added_line_numbers())
        return numbers

    def removed_line_numbers(self) -> set[int]:
        numbers: set[int] = set()
        for hunk in self.hunks:
            numbers.update(hunk.removed_line_numbers())
        return numbers

    def changed_line_count(self) -> int:
        return len(self.added_line_numbers()) + len(self.removed_line_numbers())

    def text(self) -> str:
        old_label = f"a/{self.old_path}" if self.old_path else "/dev/null"
        new_label = f"b/{self.new_path}" if self.new_path else "/dev/null"
        parts = [f"--- {old_label}", f"+++ {new_label}"]
        if self.withheld:
            parts.append("[secret-like file changed; contents withheld]")
        if self.is_binary:
            parts.append("Binary file changed")
        for hunk in self.hunks:
            parts.append(hunk.text())
        return "\n".join(parts)


class Diff(BaseModel):
    base_sha: str
    head_sha: str
    files: list[FileDiff] = Field(default_factory=list)

    def file(self, path: str) -> FileDiff | None:
        for file_diff in self.files:
            if file_diff.new_path == path or file_diff.old_path == path:
                return file_diff
        return None

    def hunks(self) -> list[Hunk]:
        all_hunks = []
        for file_diff in self.files:
            all_hunks.extend(file_diff.hunks)
        return all_hunks

    def hunk(self, hunk_id: str) -> Hunk | None:
        for hunk in self.hunks():
            if hunk.id == hunk_id:
                return hunk
        return None

    def text(self, path: str | None = None) -> str:
        sections = []
        for file_diff in self.files:
            if path and file_diff.path != path and file_diff.old_path != path:
                continue
            sections.append(file_diff.text())
        return "\n".join(sections)


def split_sections(diff_text: str) -> list[list[str]]:
    sections: list[list[str]] = []
    current: list[str] | None = None
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            current = [line]
            sections.append(current)
        elif current is not None:
            current.append(line)
    return sections


def parse_section(lines: list[str], path_hint: str | None) -> FileDiff:
    old_path = None
    new_path = None
    status = "M"
    is_binary = False
    hunks: list[Hunk] = []
    current_hunk: Hunk | None = None
    for line in lines[1:]:
        if current_hunk is None:
            if line.startswith("new file mode"):
                status = "A"
            elif line.startswith("deleted file mode"):
                status = "D"
            elif line.startswith("rename from "):
                status = "R"
                old_path = line.removeprefix("rename from ")
            elif line.startswith("rename to "):
                new_path = line.removeprefix("rename to ")
            elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
                is_binary = True
            elif line.startswith("--- "):
                label = line[4:].split("\t")[0]
                old_path = None if label == "/dev/null" else label.removeprefix("a/")
            elif line.startswith("+++ "):
                label = line[4:].split("\t")[0]
                new_path = None if label == "/dev/null" else label.removeprefix("b/")
        match = HUNK_HEADER.match(line)
        if match:
            path_for_id = new_path or old_path or path_hint or ""
            current_hunk = Hunk(
                id=f"{path_for_id}#h{len(hunks) + 1}",
                path=path_for_id,
                old_start=int(match.group(1)),
                old_count=int(match.group(2)) if match.group(2) is not None else 1,
                new_start=int(match.group(3)),
                new_count=int(match.group(4)) if match.group(4) is not None else 1,
                header=match.group(5).strip(),
            )
            hunks.append(current_hunk)
            continue
        if current_hunk is not None and line[:1] in ("+", "-", " ", "\\"):
            current_hunk.lines.append(line)
    if status == "A":
        old_path = None
    if status == "D":
        new_path = None
    return FileDiff(old_path=old_path, new_path=new_path, status=status, is_binary=is_binary, hunks=hunks)


def apply_expected_paths(file_diff: FileDiff, old_path: str | None, new_path: str | None, status: str) -> FileDiff:
    path = new_path or old_path or ""
    renamed_hunks = []
    for number, hunk in enumerate(file_diff.hunks):
        renamed_hunks.append(hunk.model_copy(update={"id": f"{path}#h{number + 1}", "path": path}))
    return FileDiff(old_path=old_path, new_path=new_path, status=status, is_binary=file_diff.is_binary, hunks=renamed_hunks)


def parse_diff(
    diff_text: str,
    base_sha: str,
    head_sha: str,
    expected_paths: list[tuple[str | None, str | None, str]] | None = None,
) -> Diff:
    sections = split_sections(diff_text)
    paths_line_up = expected_paths is not None and len(expected_paths) == len(sections)
    files: list[FileDiff] = []
    for index, section in enumerate(sections):
        path_hint = None
        if paths_line_up:
            old_hint, new_hint, _ = expected_paths[index]
            path_hint = new_hint or old_hint
        file_diff = parse_section(section, path_hint)
        if paths_line_up:
            old_hint, new_hint, status_hint = expected_paths[index]
            file_diff = apply_expected_paths(file_diff, old_hint, new_hint, status_hint)
        files.append(file_diff)
    return Diff(base_sha=base_sha, head_sha=head_sha, files=files)
