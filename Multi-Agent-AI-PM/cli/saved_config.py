"""Persist and restore CLI user selections to avoid re-prompting."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

from cli.models import AnalystType, RiskProfile

CONFIG_PATH = Path(__file__).parent / ".saved_config.json"

# Keys that are always re-prompted (never saved).
_EXCLUDED_KEYS = {"analysis_date", "ticker"}


def save_config(selections: Dict[str, Any]) -> None:
    """Serialize user selections to a JSON file on disk.

    Skips ``analysis_date`` and ``ticker`` since those are always prompted.
    """
    data: Dict[str, Any] = {}

    for key, value in selections.items():
        if key in _EXCLUDED_KEYS:
            continue

        if key == "analysts":
            # AnalystType enum -> list of value strings
            data[key] = [a.value if hasattr(a, "value") else str(a) for a in value]
        elif key == "risk_profile" and value is not None:
            data[key] = value.model_dump() if hasattr(value, "model_dump") else value
        else:
            data[key] = value

    CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_config() -> Optional[Dict[str, Any]]:
    """Load saved config from disk.

    Returns ``None`` if the file doesn't exist or is corrupt.
    """
    if not CONFIG_PATH.exists():
        return None

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None

    # Deserialize analyst strings back to AnalystType enums
    if "analysts" in raw:
        try:
            raw["analysts"] = [AnalystType(v) for v in raw["analysts"]]
        except ValueError:
            return None

    # Deserialize risk_profile dict back to RiskProfile model
    if "risk_profile" in raw and isinstance(raw["risk_profile"], dict):
        try:
            raw["risk_profile"] = RiskProfile(**raw["risk_profile"])
        except Exception:
            return None

    return raw


def format_config_summary(config: Dict[str, Any]) -> str:
    """Return a compact, human-readable summary of saved settings."""
    lines = []

    mode = config.get("mode", "?")
    lines.append(f"  [cyan]Mode:[/cyan]           {mode.title()}")

    analysts = config.get("analysts", [])
    analyst_str = ", ".join(
        a.value if hasattr(a, "value") else str(a) for a in analysts
    )
    lines.append(f"  [cyan]Analysts:[/cyan]       {analyst_str}")

    depth_map = {1: "Shallow", 3: "Medium", 5: "Deep"}
    depth = config.get("research_depth", "?")
    lines.append(f"  [cyan]Research Depth:[/cyan] {depth_map.get(depth, depth)}")

    lines.append(f"  [cyan]LLM Provider:[/cyan]  {config.get('llm_provider', '?')}")
    lines.append(f"  [cyan]Quick LLM:[/cyan]     {config.get('shallow_thinker', '?')}")
    lines.append(f"  [cyan]Deep LLM:[/cyan]      {config.get('deep_thinker', '?')}")

    risk = config.get("risk_profile")
    if risk:
        goal = risk.goal if hasattr(risk, "goal") else risk.get("goal", "?")
        risk_val = risk.risk if hasattr(risk, "risk") else risk.get("risk", "?")
        lines.append(f"  [cyan]Risk Profile:[/cyan]  {goal} / {risk_val}")

    alpaca = config.get("alpaca_api_key", "")
    if alpaca:
        masked = alpaca[:4] + "…" + alpaca[-4:] if len(alpaca) > 8 else "****"
        lines.append(f"  [cyan]Alpaca Key:[/cyan]    {masked}")

    return "\n".join(lines)
