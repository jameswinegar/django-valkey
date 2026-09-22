#!/usr/bin/env python3
"""Check the declared Python/Django support matrix against upstream schedules.

The classifiers in ``pyproject.toml`` state which Python and Django versions
django-valkey supports. This script compares them with
`endoflife.date <https://endoflife.date>`_ and reports drift:

* a declared Django or Python version that is end-of-life upstream,
* a maintained Django series that is not declared,
* a ``django>=`` floor that disagrees with the oldest declared Django,
* a ``requires-python`` floor that disagrees with the oldest declared Python,
* a declared Python that no declared Django supports, or the newest declared
  Python that the newest declared Django does not support.

Python is checked for end-of-life only: the floor is deliberately higher than
Python's own schedule, so a maintained-but-undeclared Python is not drift.

Usage::

    python util/check_support_matrix.py               # human-readable report
    python util/check_support_matrix.py --format md   # markdown, for an issue

Exit status is 0 when the matrix is current, 1 when it has drifted and 2 when
the check itself could not run (unreachable API, unexpected response shape),
so a scheduled workflow can tell a finding apart from a broken check.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

import tomllib

API = "https://endoflife.date/api/v1/products/{product}"
TIMEOUT = 30
MAX_ATTEMPTS = 4
BACKOFF_BASE = 2.0
BACKOFF_CAP = 30.0
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

EXIT_CURRENT = 0
EXIT_DRIFTED = 1
EXIT_ERROR = 2

PYTHON_PREFIX = "Programming Language :: Python :: "
DJANGO_PREFIX = "Framework :: Django :: "
VERSION_RE = re.compile(r"\d+\.\d+")

REPO_ROOT = Path(__file__).resolve().parent.parent


class Drift(NamedTuple):
    subject: str
    detail: str
    fix: str


class Declared(NamedTuple):
    pythons: list[str]
    djangos: list[str]
    django_floor: str | None
    python_floor: str | None


def die(message: str) -> NoReturn:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(EXIT_ERROR)


def version_key(v: str) -> tuple[int, ...]:
    return tuple(int(part) for part in v.split("."))


def release_key(v: str) -> tuple[int, ...]:
    """Ordering key with trailing zeros dropped, so 5.2.0 and 5.2 compare equal."""
    parts = list(version_key(v))
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def backoff_delay(attempt: int, retry_after: str | None) -> float:
    if retry_after and retry_after.strip().isdigit():
        return min(float(retry_after), BACKOFF_CAP)
    ceiling = min(BACKOFF_BASE * 2 ** (attempt - 1), BACKOFF_CAP)
    return ceiling * (0.5 + random.random() / 2)


def validate_releases(url: str, payload: Any) -> list[dict[str, Any]]:
    """Return the release records, or abort with EXIT_ERROR on an unexpected shape.

    An unhandled exception would exit 1, which is the "drifted" status, so a
    changed API would otherwise be reported as drift with an empty report.
    """
    releases = (
        payload.get("result", {}).get("releases") if isinstance(payload, dict) else None
    )
    if not isinstance(releases, list) or not releases:
        die(
            f"{url}: 'result.releases' is missing or empty; the API shape may have changed"
        )
    for entry in releases:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str) or not re.fullmatch(r"\d+(\.\d+)*", name):
            die(
                f"{url}: release name {name!r} is not a dotted version; the API shape may have changed"
            )
        if not isinstance(entry.get("isMaintained"), bool):
            die(
                f"{url}: release {name} has no boolean 'isMaintained'; the API shape may have changed"
            )
    return releases


def fetch_releases(product: str, max_attempts: int) -> list[dict[str, Any]]:
    url = API.format(product=product)
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    last_error = "unknown error"
    for attempt in range(1, max_attempts + 1):
        retry_after = None
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code} {exc.reason}"
            if exc.code not in RETRYABLE_STATUS:
                die(f"{url} returned {last_error}")
            retry_after = exc.headers.get("Retry-After")
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        except json.JSONDecodeError as exc:
            last_error = f"invalid JSON: {exc}"
        else:
            return validate_releases(url, payload)
        if attempt < max_attempts:
            delay = backoff_delay(attempt, retry_after)
            print(
                f"warning: {url}: {last_error}; retrying in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    die(
        f"could not reach {url} after {max_attempts} attempts; last error: {last_error}"
    )


def maintained(releases: list[dict[str, Any]]) -> list[str]:
    return sorted((r["name"] for r in releases if r["isMaintained"]), key=version_key)


def eol(releases: list[dict[str, Any]]) -> dict[str, str]:
    return {
        r["name"]: r.get("eolFrom") or "an unknown date"
        for r in releases
        if not r["isMaintained"]
    }


def supported_pythons(release: dict[str, Any]) -> tuple[str, str] | None:
    """Parse a Django release's "3.10 - 3.14 (added in 5.2.8)" Python range."""
    raw = (release.get("custom") or {}).get("supportedPythonVersions")
    match = (
        re.match(r"\s*(\d+\.\d+)\s*-\s*(\d+\.\d+)", raw)
        if isinstance(raw, str)
        else None
    )
    return (match.group(1), match.group(2)) if match else None


def read_declared(pyproject: Path) -> Declared:
    with pyproject.open("rb") as f:
        project = tomllib.load(f)["project"]

    def versions(prefix: str) -> list[str]:
        found = (
            c.removeprefix(prefix)
            for c in project["classifiers"]
            if c.startswith(prefix)
        )
        return sorted((v for v in found if VERSION_RE.fullmatch(v)), key=version_key)

    pythons = versions(PYTHON_PREFIX)
    djangos = versions(DJANGO_PREFIX)
    if not pythons:
        die("no 'Programming Language :: Python :: X.Y' classifiers in pyproject.toml")
    if not djangos:
        die("no 'Framework :: Django :: X.Y' classifiers in pyproject.toml")

    django_floor = None
    for dep in project.get("dependencies", []):
        if re.match(r"(?i)django\s*[<>=!~\[]", dep) or dep.strip().lower() == "django":
            if match := re.search(r">=\s*([\d.]+)", dep):
                django_floor = match.group(1)
            break

    python_floor = None
    if match := re.search(r">=\s*([\d.]+)", project.get("requires-python", "")):
        python_floor = match.group(1)

    return Declared(pythons, djangos, django_floor, python_floor)


def check(pyproject: Path, max_attempts: int) -> tuple[list[Drift], list[str]]:
    declared = read_declared(pyproject)
    django_releases = fetch_releases("django", max_attempts)
    python_releases = fetch_releases("python", max_attempts)

    drift: list[Drift] = []
    notes: list[str] = []

    django_eol = eol(django_releases)
    for version in declared.djangos:
        if version in django_eol:
            drift.append(
                Drift(
                    f"Django {version}",
                    f"end-of-life since {django_eol[version]}, but still declared as supported",
                    f'remove "{DJANGO_PREFIX}{version}" from the pyproject classifiers',
                )
            )
    for version in maintained(django_releases):
        if version not in declared.djangos:
            drift.append(
                Drift(
                    f"Django {version}",
                    "still supported upstream, but not declared",
                    f'add "{DJANGO_PREFIX}{version}" to the pyproject classifiers',
                )
            )

    oldest_django = declared.djangos[0]
    if declared.django_floor is None:
        notes.append(
            "no `django>=X.Y` floor found in [project].dependencies; floor not checked"
        )
    elif release_key(declared.django_floor) != release_key(oldest_django):
        drift.append(
            Drift(
                f"django>={declared.django_floor}",
                f"disagrees with the oldest declared series (Django {oldest_django})",
                f'set the dependency floor to "django>={oldest_django}"',
            )
        )

    python_eol = eol(python_releases)
    for version in declared.pythons:
        if version in python_eol:
            drift.append(
                Drift(
                    f"Python {version}",
                    f"end-of-life since {python_eol[version]}, but still declared as supported",
                    f'remove "{PYTHON_PREFIX}{version}" from the pyproject classifiers',
                )
            )

    oldest_python = declared.pythons[0]
    if declared.python_floor is None:
        notes.append('no `requires-python = ">=X.Y"` floor found; floor not checked')
    elif release_key(declared.python_floor) != release_key(oldest_python):
        drift.append(
            Drift(
                f"requires-python >={declared.python_floor}",
                f"disagrees with the oldest declared Python ({oldest_python})",
                f'set requires-python to ">={oldest_python}"',
            )
        )

    # Every declared Python must be supported by at least one declared Django,
    # and the newest Django must support the newest Python. These hold for any
    # shape of test matrix, so the matrix itself can stay hand-written.
    by_name = {r["name"]: r for r in django_releases}
    windows: dict[str, tuple[str, str]] = {}
    for version in declared.djangos:
        window = supported_pythons(by_name[version]) if version in by_name else None
        if window is None:
            notes.append(
                f"Django {version}: upstream Python range not machine-readable; pairing not checked"
            )
        else:
            windows[version] = window
    if windows:

        def supports(django: str, python: str) -> bool:
            low, high = windows[django]
            return version_key(low) <= version_key(python) <= version_key(high)

        for python in declared.pythons:
            if not any(supports(django, python) for django in windows):
                drift.append(
                    Drift(
                        f"Python {python}",
                        "declared, but no declared Django series supports it",
                        f'remove "{PYTHON_PREFIX}{python}" from the pyproject classifiers',
                    )
                )
        newest_django, newest_python = declared.djangos[-1], declared.pythons[-1]
        if newest_django in windows and not supports(newest_django, newest_python):
            low, high = windows[newest_django]
            drift.append(
                Drift(
                    f"Django {newest_django} + Python {newest_python}",
                    f"the newest declared Django supports Python {low} - {high}",
                    "wait for a Django release that supports it, or drop the Python classifier",
                )
            )

    return drift, notes


def render(drift: list[Drift], notes: list[str], fmt: str) -> str:
    lines: list[str] = []
    if fmt == "md":
        if drift:
            lines += [
                "The declared support matrix has drifted from upstream release schedules.",
                "",
                "| Subject | Problem | Suggested fix |",
                "| --- | --- | --- |",
                *(f"| `{d.subject}` | {d.detail} | {d.fix} |" for d in drift),
                "",
                "After updating `pyproject.toml`, also update the test matrix in `.github/workflows/ci.yml` and the version list in `README.rst`.",
            ]
        else:
            lines.append(
                "The declared support matrix matches upstream release schedules."
            )
        if notes:
            lines += ["", "Notes:", *(f"- {n}" for n in notes)]
    else:
        if drift:
            lines.append(f"Support matrix has drifted ({len(drift)} problem(s)):")
            for d in drift:
                lines += [f"  - {d.subject}: {d.detail}", f"    fix: {d.fix}"]
        else:
            lines.append(
                "Support matrix is current: declared versions match upstream release schedules."
            )
        lines += [f"  note: {n}" for n in notes]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pyproject", type=Path, default=REPO_ROOT / "pyproject.toml")
    parser.add_argument("--format", choices=("text", "md"), default="text")
    parser.add_argument("--max-attempts", type=int, default=MAX_ATTEMPTS, metavar="N")
    args = parser.parse_args()
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    drift, notes = check(args.pyproject, args.max_attempts)
    print(render(drift, notes, args.format))
    return EXIT_DRIFTED if drift else EXIT_CURRENT


if __name__ == "__main__":
    sys.exit(main())
