"""Prompt loader for agent markdown prompt files.

Usage
-----
    from src.agents.prompts import load_prompt

    prompts = load_prompt("market_analyst")
    print(prompts["PHASE1_PROMPT"])
    print(prompts["HORIZON_FOCUS"]["long_term"])
"""

import os
from typing import Dict

_PROMPTS_DIR = os.path.dirname(os.path.abspath(__file__))


def _parse_md(path: str) -> dict[str, str | dict[str, str]]:
    """Parse a markdown file and return a dict of section_name -> body.

    Sections that contain ``### key`` subsections are returned as nested dicts.
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    sections: dict[str, str | dict[str, str]] = {}
    current_name: str | None = None
    current_body: list[str] = []

    for line in content.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("## "):
            if current_name is not None:
                sections[current_name] = _process_section(current_body)
            current_name = stripped[3:].strip()
            current_body = []
        elif current_name is not None:
            current_body.append(line)

    if current_name is not None:
        sections[current_name] = _process_section(current_body)

    return sections


def _process_section(lines: list[str]) -> str | dict[str, str]:
    """Process section lines. If subsections (###) exist, return a dict."""
    body = "".join(lines).strip()
    # Check for subsection headers
    subsections: dict[str, str] = {}
    current_sub_name: str | None = None
    current_sub_body: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("### "):
            if current_sub_name is not None:
                subsections[current_sub_name] = "".join(current_sub_body).strip()
            current_sub_name = stripped[4:].strip()
            current_sub_body = []
        elif current_sub_name is not None:
            current_sub_body.append(line)

    if current_sub_name is not None:
        subsections[current_sub_name] = "".join(current_sub_body).strip()

    if subsections:
        return subsections
    return body


def load_prompt(name: str) -> dict[str, str | dict[str, str]]:
    """Load prompt sections from ``<name>.md`` in the prompts directory.

    Parameters
    ----------
    name : str
        Base name of the markdown file (e.g. ``"market_analyst"``).

    Returns
    -------
    dict[str, str | dict[str, str]]
        Mapping of section names (e.g. ``"PHASE1_PROMPT"``) to prompt text.
        Sections with ``###`` subsections (like ``HORIZON_FOCUS``) are returned
        as nested dicts.
    """
    path = os.path.join(_PROMPTS_DIR, f"{name}.md")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return _parse_md(path)
