#!/usr/bin/env python3
"""Rewrites version references after a release, for the Sync workflow.

Reads RELEASE_VERSION and NEXT_SNAPSHOT from the environment and updates pom.xml, README.md,
docs/antora.yml, and any Antora page that carries a "Latest release"/"Latest snapshot" dependency
snippet or this project's own Gradle coordinates.

CHANGELOG.md is intentionally left untouched: its release section and fresh "[Unreleased]" heading
are written before the tag is published, and arrive on develop via the merge that precedes this
script, not by bumping a version string.
"""
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ID = "irurueta-algebra"

# Marker lines (lowercased, prefix-matched) that precede a release-version or snapshot-version
# <version> tag in README.md / Antora pages. Adjust if this repository's wording changes.
RELEASE_MARKERS = ("latest release", "for a released dependency")
SNAPSHOT_MARKERS = ("latest snapshot", "for local development against the current repository snapshot")

GRADLE_COORD_RE = re.compile(rf"(com\.irurueta:{re.escape(ARTIFACT_ID)}:)[^\"')]*")


def replace_dependency_snippets(text, release_version, next_snapshot):
    """Replaces the <version> tag following a release/snapshot marker line, and this project's own
    Gradle coordinate (which always tracks the latest release, regardless of marker proximity)."""
    lines = text.splitlines(keepends=True)
    pending = None
    for i, line in enumerate(lines):
        lower = line.strip().lower()
        if GRADLE_COORD_RE.search(line):
            lines[i] = GRADLE_COORD_RE.sub(rf"\g<1>{release_version}", line)
        elif any(lower.startswith(m) for m in RELEASE_MARKERS):
            pending = release_version
        elif any(lower.startswith(m) for m in SNAPSHOT_MARKERS):
            pending = next_snapshot
        elif pending and "<version>" in line:
            lines[i] = re.sub(r"(<version>)[^<]*(</version>)", rf"\g<1>{pending}\g<2>", line)
            pending = None
    return "".join(lines)


def update_pom(next_snapshot):
    path = REPO_ROOT / "pom.xml"
    text = path.read_text()
    updated, count = re.subn(
        rf"(<artifactId>{re.escape(ARTIFACT_ID)}</artifactId>\s*\n\s*<version>)[^<]*(</version>)",
        rf"\g<1>{next_snapshot}\g<2>",
        text,
        count=1,
    )
    if count != 1:
        sys.exit("pom.xml: could not find the project's own <version> element to bump")
    path.write_text(updated)


def update_readme(release_version, next_snapshot):
    path = REPO_ROOT / "README.md"
    text = path.read_text()
    text = re.sub(
        r"(- Latest release: `)[^`]*(`)",
        rf"\g<1>{release_version}\g<2>",
        text,
    )
    text = re.sub(
        r"(- Current development version: `)[^`]*(`)",
        rf"\g<1>{next_snapshot}\g<2>",
        text,
    )
    text = replace_dependency_snippets(text, release_version, next_snapshot)
    path.write_text(text)


def update_antora_component_version(release_version):
    path = REPO_ROOT / "docs" / "antora.yml"
    if not path.exists():
        return
    text = path.read_text()
    updated, count = re.subn(r"(?m)^version:.*$", f"version: {release_version}", text, count=1)
    if count == 1:
        path.write_text(updated)


def update_antora_pages(release_version, next_snapshot):
    pages_dir = REPO_ROOT / "docs" / "modules" / "ROOT" / "pages"
    if not pages_dir.exists():
        return
    for path in sorted(pages_dir.glob("*.adoc")):
        text = path.read_text()
        if "<version>" not in text and ARTIFACT_ID not in text:
            continue
        updated = replace_dependency_snippets(text, release_version, next_snapshot)
        if updated != text:
            path.write_text(updated)


def main():
    release_version = os.environ["RELEASE_VERSION"]
    next_snapshot = os.environ["NEXT_SNAPSHOT"]

    update_pom(next_snapshot)
    update_readme(release_version, next_snapshot)
    update_antora_component_version(release_version)
    update_antora_pages(release_version, next_snapshot)


if __name__ == "__main__":
    main()
