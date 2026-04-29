from typing import Optional
import datetime
import json
import typer
from pathlib import Path
from functools import wraps
from rich.console import Console
from dotenv import load_dotenv

# Load environment variables from .env file
from rich.panel import Panel
from rich.spinner import Spinner
from rich.live import Live
from rich.columns import Columns
from rich.markdown import Markdown
from rich.layout import Layout
from rich.text import Text
from rich.table import Table
from collections import deque
import time
from rich.tree import Tree
from rich import box
from rich.align import Align
from rich.rule import Rule

from src.graph.trading_graph import TradingAgentsGraph
from src.default_config import DEFAULT_CONFIG
from cli.utils import *
from cli.announcements import fetch_announcements, display_announcements
from cli.stats_handler import StatsCallbackHandler
from cli.saved_config import save_config, load_config, format_config_summary
from src.graph.stock_screener import StockScreener
from src.dataflows.alpaca import get_open_positions

console = Console()
load_dotenv()


app = typer.Typer(
    name="AI Portfolio Manager",
    help="Multi-Agent AI Portfolio Manager CLI",
    add_completion=True,  # Enable shell completion
)


# Create a deque to store recent messages with a maximum length
class MessageBuffer:
    # Fixed teams that always run (not user-selectable)
    FIXED_AGENTS = {
        "Trading Team": ["Trader"],
    }

    # Analyst name mapping
    ANALYST_MAPPING = {
        "market": "Market Analyst",
        "fundamentals": "Fundamentals Analyst",
    }

    # Report section mapping: section -> (analyst_key for filtering, finalizing_agent)
    # analyst_key: which analyst selection controls this section (None = always included)
    # finalizing_agent: which agent must be "completed" for this report to count as done
    REPORT_SECTIONS = {
        "market_report": ("market", "Market Analyst"),
        "fundamentals_report": ("fundamentals", "Fundamentals Analyst"),
        "composite_signal": (None, "Synthesis Agent"),
    }

    def __init__(self, max_length=100):
        self.messages = deque(maxlen=max_length)
        self.tool_calls = deque(maxlen=max_length)
        self.current_report = None
        self.final_report = None  # Store the complete final report
        self.agent_status = {}
        self.current_agent = None
        self.report_sections = {}
        self.selected_analysts = []
        self._last_message_id = None

    def init_for_analysis(self, selected_analysts):
        """Initialize agent status and report sections based on selected analysts.

        Args:
            selected_analysts: List of analyst type strings (e.g., ["market", "fundamentals"])
        """
        self.selected_analysts = [a.lower() for a in selected_analysts]

        # Build agent_status dynamically
        self.agent_status = {}

        # Add selected analysts
        for analyst_key in self.selected_analysts:
            if analyst_key in self.ANALYST_MAPPING:
                self.agent_status[self.ANALYST_MAPPING[analyst_key]] = "pending"

        # Add fixed teams
        for team_agents in self.FIXED_AGENTS.values():
            for agent in team_agents:
                self.agent_status[agent] = "pending"

        # Build report_sections dynamically
        self.report_sections = {}
        for section, (analyst_key, _) in self.REPORT_SECTIONS.items():
            if analyst_key is None or analyst_key in self.selected_analysts:
                self.report_sections[section] = None

        # Reset other state
        self.current_report = None
        self.final_report = None
        self.current_agent = None
        self.messages.clear()
        self.tool_calls.clear()
        self._last_message_id = None

    def get_completed_reports_count(self):
        """Count reports that are finalized (their finalizing agent is completed).

        A report is considered complete when:
        1. The report section has content (not None), AND
        2. The agent responsible for finalizing that report has status "completed"

        This prevents interim updates from counting as completed.
        """
        count = 0
        for section in self.report_sections:
            if section not in self.REPORT_SECTIONS:
                continue
            _, finalizing_agent = self.REPORT_SECTIONS[section]
            # Report is complete if it has content AND its finalizing agent is done
            has_content = self.report_sections.get(section) is not None
            agent_done = self.agent_status.get(finalizing_agent) == "completed"
            if has_content and agent_done:
                count += 1
        return count

    def add_message(self, message_type, content):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.messages.append((timestamp, message_type, content))

    def add_tool_call(self, tool_name, args):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.tool_calls.append((timestamp, tool_name, args))

    def update_agent_status(self, agent, status):
        if agent in self.agent_status:
            self.agent_status[agent] = status
            self.current_agent = agent

    def update_report_section(self, section_name, content):
        if section_name in self.report_sections:
            self.report_sections[section_name] = content
            self._update_current_report()

    def _update_current_report(self):
        # For the panel display, only show the most recently updated section
        latest_section = None
        latest_content = None

        # Find the most recently updated section
        for section, content in self.report_sections.items():
            if content is not None:
                latest_section = section
                latest_content = content

        if latest_section and latest_content:
            # Format the current section for display
            section_titles = {
                "market_report": "Market Analysis",
                "fundamentals_report": "Fundamentals Analysis",
                "composite_signal": "Synthesis Signal",
            }
            self.current_report = (
                f"### {section_titles[latest_section]}\n"
                f"{_fmt_report_content(latest_content)}"
            )

        # Update the final complete report
        self._update_final_report()

    def _update_final_report(self):
        report_parts = []

        # Analyst Team Reports - use .get() to handle missing sections
        analyst_sections = [
            "market_report",
            "fundamentals_report",
        ]
        if any(self.report_sections.get(section) for section in analyst_sections):
            report_parts.append("## Analyst Team Reports")
            if self.report_sections.get("market_report"):
                report_parts.append(
                    f"### Market Analysis\n"
                    f"{_fmt_report_content(self.report_sections['market_report'])}"
                )
            if self.report_sections.get("fundamentals_report"):
                report_parts.append(
                    f"### Fundamentals Analysis\n"
                    f"{_fmt_report_content(self.report_sections['fundamentals_report'])}"
                )

        # Trading Team Reports
        if self.report_sections.get("composite_signal"):
            report_parts.append("## Trading Team Plan")
            report_parts.append(
                _fmt_report_content(self.report_sections["composite_signal"])
            )

        self.final_report = "\n\n".join(report_parts) if report_parts else None


def _fmt_report_content(content: str) -> str:
    """Try to pretty-print JSON report content; fall back to raw text."""
    if not content:
        return content
    try:
        parsed = json.loads(content)
        pretty = json.dumps(parsed, indent=2)
        return f"```json\n{pretty}\n```"
    except Exception:
        return content


message_buffer = MessageBuffer()


def create_layout():
    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    layout["main"].split_column(
        Layout(name="upper", ratio=3), Layout(name="analysis", ratio=5)
    )
    layout["upper"].split_row(
        Layout(name="progress", ratio=2), Layout(name="messages", ratio=3)
    )
    return layout


def format_tokens(n):
    """Format token count for display."""
    if n >= 1000:
        return f"{n / 1000:.1f}k"
    return str(n)


def _render_status_cell(status):
    """Return a renderable for an agent status value."""
    if status == "in_progress":
        return Spinner("dots", text="[blue]in_progress[/blue]", style="bold cyan")
    status_color = {"pending": "yellow", "completed": "green", "error": "red"}.get(
        status, "white"
    )
    return f"[{status_color}]{status}[/{status_color}]"


# Group agents by team (used by both single and multi-stock views)
ALL_TEAMS = {
    "Analyst Team": [
        "Market Analyst",
        "Fundamentals Analyst",
    ],
    "Trading Team": ["Trader"],
}


def render_agent_progress_panel(buf, title="Progress"):
    """Build the agent progress Panel from a MessageBuffer."""
    progress_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        box=box.SIMPLE_HEAD,
        title=None,
        padding=(0, 2),
        expand=True,
    )
    progress_table.add_column("Team", style="cyan", justify="center", width=20)
    progress_table.add_column("Agent", style="green", justify="center", width=20)
    progress_table.add_column("Status", style="yellow", justify="center", width=20)

    teams = {}
    for team, agents in ALL_TEAMS.items():
        active_agents = [a for a in agents if a in buf.agent_status]
        if active_agents:
            teams[team] = active_agents

    for team, agents in teams.items():
        first_agent = agents[0]
        status = buf.agent_status.get(first_agent, "pending")
        progress_table.add_row(team, first_agent, _render_status_cell(status))

        for agent in agents[1:]:
            status = buf.agent_status.get(agent, "pending")
            progress_table.add_row("", agent, _render_status_cell(status))

        progress_table.add_row("─" * 20, "─" * 20, "─" * 20, style="dim")

    return Panel(progress_table, title=title, border_style="cyan", padding=(1, 2))


def render_messages_panel(buf, title="Messages & Tools"):
    """Build the messages/tools Panel from a MessageBuffer."""
    messages_table = Table(
        show_header=True,
        header_style="bold magenta",
        show_footer=False,
        expand=True,
        box=box.MINIMAL,
        show_lines=True,
        padding=(0, 1),
    )
    messages_table.add_column("Time", style="cyan", width=8, justify="center")
    messages_table.add_column("Type", style="green", width=10, justify="center")
    messages_table.add_column("Content", style="white", no_wrap=False, ratio=1)

    all_messages = []
    for timestamp, tool_name, args in buf.tool_calls:
        formatted_args = format_tool_args(args)
        all_messages.append((timestamp, "Tool", f"{tool_name}: {formatted_args}"))

    for timestamp, msg_type, content in buf.messages:
        content_str = str(content) if content else ""
        if len(content_str) > 200:
            content_str = content_str[:197] + "..."
        all_messages.append((timestamp, msg_type, content_str))

    all_messages.sort(key=lambda x: x[0], reverse=True)
    for timestamp, msg_type, content in all_messages[:12]:
        wrapped_content = Text(content, overflow="fold")
        messages_table.add_row(timestamp, msg_type, wrapped_content)

    return Panel(messages_table, title=title, border_style="blue", padding=(1, 2))


def render_analysis_panel(buf, title="Current Report"):
    """Build the analysis report Panel from a MessageBuffer."""
    if buf.current_report:
        return Panel(
            Markdown(buf.current_report),
            title=title,
            border_style="green",
            padding=(1, 2),
        )
    return Panel(
        "[italic]Waiting for analysis report...[/italic]",
        title=title,
        border_style="green",
        padding=(1, 2),
    )


def render_footer_panel(buf, stats_handler=None, start_time=None):
    """Build the footer stats Panel from a MessageBuffer."""
    agents_completed = sum(
        1 for status in buf.agent_status.values() if status == "completed"
    )
    agents_total = len(buf.agent_status)
    reports_completed = buf.get_completed_reports_count()
    reports_total = len(buf.report_sections)

    stats_parts = [f"Agents: {agents_completed}/{agents_total}"]

    if stats_handler:
        stats = stats_handler.get_stats()
        stats_parts.append(f"LLM: {stats['llm_calls']}")
        stats_parts.append(f"Tools: {stats['tool_calls']}")
        if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
            tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
        else:
            tokens_str = "Tokens: --"
        stats_parts.append(tokens_str)

    stats_parts.append(f"Reports: {reports_completed}/{reports_total}")

    if start_time:
        elapsed = time.time() - start_time
        elapsed_str = f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        stats_parts.append(elapsed_str)

    stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
    stats_table.add_column("Stats", justify="center")
    stats_table.add_row(" | ".join(stats_parts))
    return Panel(stats_table, border_style="grey50")


def update_display(layout, spinner_text=None, stats_handler=None, start_time=None):
    # Header
    layout["header"].update(
        Panel(
            "[bold green]Multi-Agent AI Portfolio Manager[/bold green]",
            title="AI Portfolio Manager",
            border_style="green",
            padding=(1, 2),
            expand=True,
        )
    )
    layout["progress"].update(render_agent_progress_panel(message_buffer))
    layout["messages"].update(render_messages_panel(message_buffer))
    layout["analysis"].update(render_analysis_panel(message_buffer))
    layout["footer"].update(
        render_footer_panel(message_buffer, stats_handler, start_time)
    )


def get_user_selections():
    """Get all user selections before starting the analysis display."""

    # Create a boxed questionnaire for each step
    def create_question_box(title, prompt, default=None):
        box_content = f"[bold]{title}[/bold]\n"
        box_content += f"[dim]{prompt}[/dim]"
        if default:
            box_content += f"\n[dim]Default: {default}[/dim]"
        return Panel(box_content, border_style="blue", padding=(1, 2))

    # ── Check for saved configuration ──────────────────────────────
    saved = load_config()
    if saved:
        console.print(
            Panel(
                f"[bold green]Saved Configuration Found[/bold green]\n\n"
                f"{format_config_summary(saved)}",
                border_style="green",
                padding=(1, 2),
                title="Saved Config",
            )
        )
        import questionary

        use_saved = questionary.select(
            "Use saved configuration?",
            choices=[
                questionary.Choice(
                    "Use saved config — skip to analysis date", value="saved"
                ),
                questionary.Choice(
                    "Reconfigure — set up all options again", value="new"
                ),
            ],
            instruction="\n- Use arrow keys to navigate\n- Press Enter to select",
            style=questionary.Style(
                [
                    ("selected", "fg:cyan noinherit"),
                    ("highlighted", "fg:cyan noinherit"),
                    ("pointer", "fg:cyan noinherit"),
                ]
            ),
        ).ask()

        if use_saved == "saved":
            # Always prompt for analysis date
            default_date = datetime.datetime.now().strftime("%Y-%m-%d")
            console.print(
                create_question_box(
                    "Analysis Date",
                    "Enter the analysis date (YYYY-MM-DD)",
                    default_date,
                )
            )
            analysis_date = get_analysis_date()

            # Determine ticker based on saved mode
            selected_ticker = None
            if saved.get("mode") == "single":
                console.print(
                    create_question_box(
                        "Ticker Symbol",
                        "Enter the ticker symbol to analyze",
                        "SPY",
                    )
                )
                selected_ticker = get_ticker()

            # Backfill any new keys missing from old saved configs
            if "code_agent_model" not in saved:
                console.print(
                    create_question_box(
                        "Code Agent", "Configure the code validation agent model"
                    )
                )
                saved["code_agent_model"] = select_code_agent_model(
                    saved.get("llm_provider", "ollama")
                )
            if "horizons_enabled" not in saved:
                console.print(
                    create_question_box(
                        "Horizons", "Select investment time horizons"
                    )
                )
                saved["horizons_enabled"] = ask_horizons()

            return {
                **saved,
                "analysis_date": analysis_date,
                "ticker": selected_ticker,
            }

    # ── Fresh configuration flow ───────────────────────────────────

    # Step 0: Analysis mode
    console.print(
        create_question_box(
            "Step 0: Analysis Mode",
            "Choose single stock analysis or portfolio discovery",
        )
    )
    analysis_mode = select_analysis_mode()

    if analysis_mode == "single":
        # Step 1: Ticker symbol (existing flow)
        console.print(
            create_question_box(
                "Step 1: Ticker Symbol", "Enter the ticker symbol to analyze", "SPY"
            )
        )
        selected_ticker = get_ticker()
    else:
        selected_ticker = None  # Will be set by screener later

    # Step 2: Analysis date
    default_date = datetime.datetime.now().strftime("%Y-%m-%d")
    console.print(
        create_question_box(
            "Step 2: Analysis Date",
            "Enter the analysis date (YYYY-MM-DD)",
            default_date,
        )
    )
    analysis_date = get_analysis_date()

    # Step 3: Select analysts
    console.print(
        create_question_box(
            "Step 3: Analysts Team", "Select your LLM analyst agents for the analysis"
        )
    )
    selected_analysts = select_analysts()
    console.print(
        f"[green]Selected analysts:[/green] {', '.join(analyst.value for analyst in selected_analysts)}"
    )

    # Step 4: Research depth
    console.print(
        create_question_box(
            "Step 4: Research Depth", "Select your research depth level"
        )
    )
    selected_research_depth = select_research_depth()

    # Step 5: OpenAI backend
    console.print(
        create_question_box("Step 5: OpenAI backend", "Select which service to talk to")
    )
    selected_llm_provider, backend_url = select_llm_provider()

    # Step 6: Thinking agents
    console.print(
        create_question_box(
            "Step 6: Thinking Agents", "Select your thinking agents for analysis"
        )
    )
    selected_shallow_thinker = select_shallow_thinking_agent(selected_llm_provider)
    selected_deep_thinker = select_deep_thinking_agent(selected_llm_provider)

    # Step 7: Provider-specific thinking configuration
    thinking_level = None
    reasoning_effort = None

    provider_lower = selected_llm_provider.lower()
    if provider_lower == "google":
        console.print(
            create_question_box(
                "Step 7: Thinking Mode", "Configure Gemini thinking mode"
            )
        )
        thinking_level = ask_gemini_thinking_config()
    elif provider_lower == "openai":
        console.print(
            create_question_box(
                "Step 7: Reasoning Effort", "Configure OpenAI reasoning effort level"
            )
        )
        reasoning_effort = ask_openai_reasoning_effort()

    # Step 8: Code agent
    console.print(
        create_question_box(
            "Step 8: Code Agent",
            "Configure the LLM that validates analyst code execution",
        )
    )
    code_agent_model = select_code_agent_model(selected_llm_provider.lower())

    # Step 9: Horizons
    console.print(
        create_question_box(
            "Step 9: Investment Horizons", "Select time horizons for analysis"
        )
    )
    horizons_enabled = ask_horizons()

    # Step 10: Alpaca credentials
    console.print(
        create_question_box(
            "Step 10: Alpaca Paper Trading",
            "Enter your Alpaca Paper Trading API credentials for trade execution",
        )
    )
    alpaca_api_key, alpaca_secret_key = ask_alpaca_credentials()

    selections = {
        "mode": analysis_mode,
        "ticker": selected_ticker,
        "analysis_date": analysis_date,
        "analysts": selected_analysts,
        "research_depth": selected_research_depth,
        "llm_provider": selected_llm_provider.lower(),
        "backend_url": backend_url,
        "shallow_thinker": selected_shallow_thinker,
        "deep_thinker": selected_deep_thinker,
        "google_thinking_level": thinking_level,
        "openai_reasoning_effort": reasoning_effort,
        "code_agent_model": code_agent_model,
        "horizons_enabled": horizons_enabled,
        "alpaca_api_key": alpaca_api_key,
        "alpaca_secret_key": alpaca_secret_key,
    }

    # Save for next run
    save_config(selections)

    return selections


def get_ticker():
    """Get ticker symbol from user input."""
    return typer.prompt("", default="SPY")


def get_analysis_date():
    """Get the analysis date from user input."""
    while True:
        date_str = typer.prompt(
            "", default=datetime.datetime.now().strftime("%Y-%m-%d")
        )
        try:
            # Validate date format and ensure it's not in the future
            analysis_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
            if analysis_date.date() > datetime.datetime.now().date():
                console.print("[red]Error: Analysis date cannot be in the future[/red]")
                continue
            return date_str
        except ValueError:
            console.print(
                "[red]Error: Invalid date format. Please use YYYY-MM-DD[/red]"
            )


def save_report_to_disk(final_state, ticker: str, save_path: Path):
    """Save complete analysis report to disk with organized subfolders."""
    save_path.mkdir(parents=True, exist_ok=True)
    sections = []

    # 1. Analysts
    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    if final_state.get("market_report"):
        analysts_dir.mkdir(exist_ok=True)
        market_pretty = _fmt_report_content(final_state["market_report"])
        (analysts_dir / "market.md").write_text(market_pretty)
        analyst_parts.append(("Market Analyst", market_pretty))
    if final_state.get("fundamentals_report"):
        analysts_dir.mkdir(exist_ok=True)
        fund_pretty = _fmt_report_content(final_state["fundamentals_report"])
        (analysts_dir / "fundamentals.md").write_text(fund_pretty)
        analyst_parts.append(
            ("Fundamentals Analyst", fund_pretty)
        )
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    # 3. Trading
    if final_state.get("composite_signal"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        try:
            parsed = json.loads(final_state["composite_signal"])
            composite_pretty = json.dumps(parsed, indent=2)
        except Exception:
            composite_pretty = final_state["composite_signal"]
        (trading_dir / "synthesis.md").write_text(composite_pretty)
        sections.append(
            f"## III. Synthesis Signal\n\n### Synthesis Agent\n```json\n{composite_pretty}\n```"
        )

    # Write consolidated report
    header = f"# Trading Analysis Report: {ticker}\n\nGenerated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    (save_path / "complete_report.md").write_text(header + "\n\n".join(sections))
    return save_path / "complete_report.md"


def display_complete_report(final_state):
    """Display the complete analysis report sequentially (avoids truncation)."""
    console.print()
    console.print(Rule("Complete Analysis Report", style="bold green"))

    # I. Analyst Team Reports
    analysts = []
    if final_state.get("market_report"):
        analysts.append(("Market Analyst", final_state["market_report"]))
    if final_state.get("fundamentals_report"):
        analysts.append(("Fundamentals Analyst", final_state["fundamentals_report"]))
    if analysts:
        console.print(
            Panel("[bold]I. Analyst Team Reports[/bold]", border_style="cyan")
        )
        for title, content in analysts:
            console.print(
                Panel(
                    Markdown(_fmt_report_content(content)),
                    title=title,
                    border_style="blue",
                    padding=(1, 2),
                )
            )

    # III. Synthesis Signal
    if final_state.get("composite_signal"):
        console.print(
            Panel("[bold]III. Synthesis Signal[/bold]", border_style="yellow")
        )
        try:
            parsed = json.loads(final_state["composite_signal"])
            composite_pretty = json.dumps(parsed, indent=2)
        except Exception:
            composite_pretty = final_state["composite_signal"]
        console.print(
            Panel(
                Markdown(f"```json\n{composite_pretty}\n```"),
                title="Synthesis Agent",
                border_style="blue",
                padding=(1, 2),
            )
        )


# Ordered list of analysts for status transitions
ANALYST_ORDER = ["market", "fundamentals", "news"]
ANALYST_AGENT_NAMES = {
    "market": "Market Analyst",
    "fundamentals": "Fundamentals Analyst",
    "news": "News Analyst",
}
ANALYST_REPORT_MAP = {
    "market": "market_report",
    "fundamentals": "fundamentals_report",
    "news": "news_report",
}


def update_analyst_statuses(message_buffer, chunk):
    """Update all analyst statuses based on current report state.

    Logic (parallel execution):
    - Analysts with reports = completed
    - Analysts without reports = in_progress (all run simultaneously)
    - When all analysts done, set Trader to in_progress
    """
    selected = message_buffer.selected_analysts
    all_done = True

    for analyst_key in ANALYST_ORDER:
        if analyst_key not in selected:
            continue

        agent_name = ANALYST_AGENT_NAMES[analyst_key]
        report_key = ANALYST_REPORT_MAP[analyst_key]
        has_report = bool(chunk.get(report_key))

        if has_report:
            message_buffer.update_agent_status(agent_name, "completed")
            message_buffer.update_report_section(report_key, chunk[report_key])
        else:
            message_buffer.update_agent_status(agent_name, "in_progress")
            all_done = False

    # When all analysts complete, transition Trader to in_progress
    if all_done and selected:
        if message_buffer.agent_status.get("Trader") == "pending":
            message_buffer.update_agent_status("Trader", "in_progress")


def extract_content_string(content):
    """Extract string content from various message formats.
    Returns None if no meaningful text content is found.
    """
    import ast

    def is_empty(val):
        """Check if value is empty using Python's truthiness."""
        if val is None or val == "":
            return True
        if isinstance(val, str):
            s = val.strip()
            if not s:
                return True
            try:
                return not bool(ast.literal_eval(s))
            except (ValueError, SyntaxError):
                return False  # Can't parse = real text
        return not bool(val)

    if is_empty(content):
        return None

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, dict):
        text = content.get("text", "")
        return text.strip() if not is_empty(text) else None

    if isinstance(content, list):
        text_parts = [
            item.get("text", "").strip()
            if isinstance(item, dict) and item.get("type") == "text"
            else (item.strip() if isinstance(item, str) else "")
            for item in content
        ]
        result = " ".join(t for t in text_parts if t and not is_empty(t))
        return result if result else None

    return str(content).strip() if not is_empty(content) else None


def classify_message_type(message) -> tuple[str, str | None]:
    """Classify LangChain message into display type and extract content.

    Returns:
        (type, content) - type is one of: User, Agent, Data, Control
                        - content is extracted string or None
    """
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

    content = extract_content_string(getattr(message, "content", None))

    if isinstance(message, HumanMessage):
        return ("User", content)

    if isinstance(message, ToolMessage):
        return ("Data", content)

    if isinstance(message, AIMessage):
        return ("Agent", content)

    # Fallback for unknown types
    return ("System", content)


def format_tool_args(args, max_length=80) -> str:
    """Format tool arguments for terminal display."""
    result = str(args)
    if len(result) > max_length:
        return result[: max_length - 3] + "..."
    return result


def _fmt_mcap_cli(val) -> str:
    """Format market cap for CLI display."""
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.1f}T"
    if val >= 1e9:
        return f"${val / 1e9:.1f}B"
    if val >= 1e6:
        return f"${val / 1e6:.0f}M"
    return f"${val:,.0f}"


def run_analysis():
    # First get all user selections
    selections = get_user_selections()

    # Route to multi-analysis if portfolio discovery mode
    if selections.get("mode") == "portfolio":
        run_multi_analysis(selections)
        return

    # Create config with user selections
    config = DEFAULT_CONFIG.copy()
    depth_map = {1: "shallow", 3: "medium", 5: "deep"}
    config["research_depth"] = depth_map.get(
        selections["research_depth"], "shallow"
    )
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    # Provider-specific thinking configuration
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    # Code agent
    config["code_agent_model"] = selections.get("code_agent_model", "qwen2.5-coder:32b")
    cbu = selections.get("code_agent_base_url", "")
    config["code_agent_base_url"] = cbu if cbu else selections.get("backend_url", "http://localhost:11434")
    # Horizons
    config["horizons_enabled"] = selections.get(
        "horizons_enabled", ["long_term", "medium_term", "short_term"]
    )
    user_risk_profile = selections.get("risk_profile")
    if user_risk_profile is not None:
        user_risk_profile = user_risk_profile.model_dump()

    # Create stats callback handler for tracking LLM/tool calls
    stats_handler = StatsCallbackHandler()

    # Normalize analyst selection to predefined order (selection is a 'set', order is fixed)
    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    # Initialize the graph with callbacks bound to LLMs
    graph = TradingAgentsGraph(
        selected_analyst_keys,
        user_risk_profile,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # Initialize message buffer with selected analysts
    message_buffer.init_for_analysis(selected_analyst_keys)

    # Track start time for elapsed display
    start_time = time.time()

    # Create result directory
    results_dir = (
        Path(config["results_dir"]) / selections["ticker"] / selections["analysis_date"]
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    report_dir = results_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    log_file = results_dir / "message_tool.log"
    log_file.touch(exist_ok=True)

    def save_message_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, message_type, content = obj.messages[-1]
            content = content.replace("\n", " ")  # Replace newlines with spaces
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [{message_type}] {content}\n")

        return wrapper

    def save_tool_call_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            timestamp, tool_name, args = obj.tool_calls[-1]
            args_str = ", ".join(f"{k}={v}" for k, v in args.items())
            with open(log_file, "a") as f:
                f.write(f"{timestamp} [Tool Call] {tool_name}({args_str})\n")

        return wrapper

    def save_report_section_decorator(obj, func_name):
        func = getattr(obj, func_name)

        @wraps(func)
        def wrapper(section_name, content):
            func(section_name, content)
            if (
                section_name in obj.report_sections
                and obj.report_sections[section_name] is not None
            ):
                content = obj.report_sections[section_name]
                if content:
                    file_name = f"{section_name}.md"
                    with open(report_dir / file_name, "w") as f:
                        f.write(content)

        return wrapper

    message_buffer.add_message = save_message_decorator(message_buffer, "add_message")
    message_buffer.add_tool_call = save_tool_call_decorator(
        message_buffer, "add_tool_call"
    )
    message_buffer.update_report_section = save_report_section_decorator(
        message_buffer, "update_report_section"
    )

    # Now start the display layout
    layout = create_layout()

    with Live(layout, refresh_per_second=4) as live:
        # Initial display
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Add initial messages
        message_buffer.add_message("System", f"Selected ticker: {selections['ticker']}")
        message_buffer.add_message(
            "System", f"Analysis date: {selections['analysis_date']}"
        )
        message_buffer.add_message(
            "System",
            f"Selected analysts: {', '.join(analyst.value for analyst in selections['analysts'])}",
        )
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # All analysts start in parallel — set all to in_progress
        for analyst_key in selected_analyst_keys:
            message_buffer.update_agent_status(ANALYST_AGENT_NAMES[analyst_key], "in_progress")
        update_display(layout, stats_handler=stats_handler, start_time=start_time)

        # Create spinner text
        spinner_text = (
            f"Analyzing {selections['ticker']} on {selections['analysis_date']}..."
        )
        update_display(
            layout, spinner_text, stats_handler=stats_handler, start_time=start_time
        )

        # Initialize state and get graph args with callbacks
        init_agent_state = graph.propagator.create_initial_state(
            selections["ticker"], selections["analysis_date"], user_risk_profile
        )
        # Pass callbacks to graph config for tool execution tracking
        # (LLM tracking is handled separately via LLM constructor)
        args = graph.propagator.get_graph_args(callbacks=[stats_handler])

        # Stream the analysis
        trace = []
        for chunk in graph.graph.stream(init_agent_state, **args):
            # Process messages if present (skip duplicates via message ID)
            if len(chunk["messages"]) > 0:
                last_message = chunk["messages"][-1]
                msg_id = getattr(last_message, "id", None)

                if msg_id != message_buffer._last_message_id:
                    message_buffer._last_message_id = msg_id

                    # Add message to buffer
                    msg_type, content = classify_message_type(last_message)
                    if content and content.strip():
                        message_buffer.add_message(msg_type, content)

                    # Handle tool calls
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            if isinstance(tool_call, dict):
                                message_buffer.add_tool_call(
                                    tool_call["name"], tool_call["args"]
                                )
                            else:
                                message_buffer.add_tool_call(
                                    tool_call.name, tool_call.args
                                )

            # Update analyst statuses based on report state (runs on every chunk)
            update_analyst_statuses(message_buffer, chunk)

            # Trading Team
            if chunk.get("composite_signal"):
                message_buffer.update_report_section(
                    "composite_signal", chunk["composite_signal"]
                )
                if message_buffer.agent_status.get("Trader") != "completed":
                    message_buffer.update_agent_status("Trader", "completed")

            # Update the display
            update_display(layout, stats_handler=stats_handler, start_time=start_time)

            trace.append(chunk)

        # Get final state and decision
        final_state = trace[-1]
        decision = graph.process_signal(final_state["composite_signal"])

        # Update all agent statuses to completed
        for agent in message_buffer.agent_status:
            message_buffer.update_agent_status(agent, "completed")

        message_buffer.add_message(
            "System", f"Completed analysis for {selections['analysis_date']}"
        )

        # Update final report sections
        for section in message_buffer.report_sections.keys():
            if section in final_state:
                message_buffer.update_report_section(section, final_state[section])

        update_display(layout, stats_handler=stats_handler, start_time=start_time)

    # Post-analysis prompts (outside Live context for clean interaction)
    console.print("\n[bold cyan]Analysis Complete![/bold cyan]\n")

    # Prompt to save report
    save_choice = typer.prompt("Save report?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        default_path = Path.cwd() / "reports" / f"{selections['ticker']}_{timestamp}"
        save_path_str = typer.prompt(
            "Save path (press Enter for default)", default=str(default_path)
        ).strip()
        save_path = Path(save_path_str)
        try:
            report_file = save_report_to_disk(
                final_state, selections["ticker"], save_path
            )
            console.print(f"\n[green]✓ Report saved to:[/green] {save_path.resolve()}")
            console.print(f"  [dim]Complete report:[/dim] {report_file.name}")
        except Exception as e:
            console.print(f"[red]Error saving report: {e}[/red]")

    # Prompt to display full report
    display_choice = (
        typer.prompt("\nDisplay full report on screen?", default="Y").strip().upper()
    )
    if display_choice in ("Y", "YES", ""):
        display_complete_report(final_state)

    # Prompt to propose trade on Alpaca
    console.print(
        f"\n[bold yellow]Decision: {decision}[/bold yellow] for {selections['ticker']}"
    )
    propose_choice = (
        typer.prompt("\nPropose a trade on Alpaca Paper Trading?", default="N")
        .strip()
        .upper()
    )
    if propose_choice in ("Y", "YES"):
        console.print("\n[cyan]Generating trade proposal...[/cyan]")
        try:
            proposal = graph.propose_trade(
                final_state["composite_signal"],
                selections["ticker"],
                selections["alpaca_api_key"],
                selections["alpaca_secret_key"],
            )

            # Display reasoning
            console.print(
                Panel(
                    Markdown(proposal["reasoning"]),
                    title="Trade Reasoning",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

            if proposal["order_params"] is None:
                console.print(
                    "\n[yellow]No order proposed (HOLD or no actionable trade).[/yellow]"
                )
            else:
                # Display proposed order parameters
                params_display = json.dumps(proposal["order_params"], indent=2)
                console.print(
                    Panel(
                        params_display,
                        title="Proposed Order",
                        border_style="yellow",
                        padding=(1, 2),
                    )
                )

                # Ask for confirmation
                confirm = typer.prompt(
                    "\nExecute this order on Alpaca Paper Trading?",
                    default="N".strip().upper(),
                )
                if confirm in ("Y", "YES"):
                    console.print("\n[cyan]Placing order...[/cyan]")
                    result = graph.execute_order(
                        proposal["order_params"],
                        selections["alpaca_api_key"],
                        selections["alpaca_secret_key"],
                    )
                    console.print(
                        Panel(
                            Markdown(result),
                            title="Trade Execution Result",
                            border_style="green",
                            padding=(1, 2),
                        )
                    )
                else:
                    console.print("\n[yellow]Order not executed.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Trade proposal/execution failed: {e}[/red]")


def run_multi_analysis(selections):
    """Portfolio discovery mode: screen stocks, research all in parallel, portfolio-level trades."""

    # Build config
    config = DEFAULT_CONFIG.copy()
    depth_map = {1: "shallow", 3: "medium", 5: "deep"}
    config["research_depth"] = depth_map.get(
        selections["research_depth"], "shallow"
    )
    config["quick_think_llm"] = selections["shallow_thinker"]
    config["deep_think_llm"] = selections["deep_thinker"]
    config["backend_url"] = selections["backend_url"]
    config["llm_provider"] = selections["llm_provider"].lower()
    config["google_thinking_level"] = selections.get("google_thinking_level")
    config["openai_reasoning_effort"] = selections.get("openai_reasoning_effort")
    config["code_agent_model"] = selections.get("code_agent_model", "qwen2.5-coder:32b")
    cbu = selections.get("code_agent_base_url", "")
    config["code_agent_base_url"] = cbu if cbu else selections.get("backend_url", "http://localhost:11434")
    config["horizons_enabled"] = selections.get(
        "horizons_enabled", ["long_term", "medium_term", "short_term"]
    )
    user_risk_profile = selections.get("risk_profile")
    if user_risk_profile is not None:
        user_risk_profile = user_risk_profile.model_dump()

    stats_handler = StatsCallbackHandler()

    selected_set = {analyst.value for analyst in selections["analysts"]}
    selected_analyst_keys = [a for a in ANALYST_ORDER if a in selected_set]

    graph = TradingAgentsGraph(
        selected_analyst_keys,
        user_risk_profile,
        config=config,
        debug=True,
        callbacks=[stats_handler],
    )

    # --- Step 1: Discover stocks ---
    console.print("\n[bold cyan]Portfolio Discovery Mode[/bold cyan]\n")

    # Fetch portfolio from Alpaca
    portfolio_tickers = []
    try:
        positions = get_open_positions(
            selections["alpaca_api_key"], selections["alpaca_secret_key"]
        )
        portfolio_tickers = [p["symbol"] for p in positions]
        if portfolio_tickers:
            console.print(
                f"[green]Current portfolio:[/green] {', '.join(portfolio_tickers)}"
            )
        else:
            console.print("[yellow]No current Alpaca positions found.[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Could not fetch Alpaca positions: {e}[/yellow]")
        portfolio_tickers = get_portfolio_tickers_manual()

    num_picks = get_num_picks()

    # Run screener with per-stage progress display

    screener = StockScreener(config, llm=graph.quick_thinking_llm)

    def _screener_progress(stage, data):
        """Display screening progress at each pipeline stage."""
        if stage == "quantitative":
            count = len(data) if data else 0
            console.print(
                f"\n[bold cyan]Stage 1 — Quantitative Screen[/bold cyan]: "
                f"[green]{count}[/green] candidates passed filters"
            )
            if data and count <= 60:
                # Show compact ticker list when manageable
                console.print(f"  [dim]{', '.join(sorted(data))}[/dim]")
            elif data:
                console.print(
                    f"  [dim]{', '.join(sorted(data)[:40])} ... and {count - 40} more[/dim]"
                )

        elif stage == "scored":
            if not data:
                return
            score_table = Table(
                title="Stage 2 — Scored & Ranked (Top 100)",
                show_header=True,
                header_style="bold magenta",
                box=box.SIMPLE_HEAD,
                padding=(0, 1),
            )
            score_table.add_column("#", style="dim", width=3)
            score_table.add_column("Ticker", style="cyan bold", width=8)
            score_table.add_column("Name", width=25, no_wrap=True)
            score_table.add_column("Sector", width=18, no_wrap=True)
            score_table.add_column("Mkt Cap", width=10, justify="right")
            score_table.add_column("P/E", width=7, justify="right")
            score_table.add_column("ROE", width=7, justify="right")
            score_table.add_column(
                "Score", width=7, justify="right", style="green bold"
            )

            for i, entry in enumerate(data[:20], 1):
                info = entry.get("info", {})
                mcap = info.get("marketCap")
                mcap_str = _fmt_mcap_cli(mcap)
                pe = info.get("trailingPE")
                pe_str = f"{pe:.1f}" if pe else "N/A"
                roe = info.get("returnOnEquity")
                roe_str = f"{roe * 100:.0f}%" if roe else "N/A"
                name = info.get("name", "")
                if len(name) > 24:
                    name = name[:22] + ".."
                sector = info.get("sector", "")
                if len(sector) > 17:
                    sector = sector[:15] + ".."
                score_table.add_row(
                    str(i),
                    entry["ticker"],
                    name,
                    sector,
                    mcap_str,
                    pe_str,
                    roe_str,
                    f"{entry['score']:.1f}",
                )
            console.print()
            console.print(score_table)
            remaining = len(data) - 20
            if remaining > 0:
                console.print(
                    f"  [dim]... and {remaining} more candidates scored[/dim]"
                )

        elif stage == "llm_picks":
            if data:
                console.print(
                    f"\n[bold cyan]Stage 3 — LLM Selection[/bold cyan]: "
                    f"[green]{', '.join(data)}[/green]"
                )

        elif stage == "final":
            if data:
                # Build a final summary table
                final_table = Table(
                    title="Final Research List",
                    show_header=True,
                    header_style="bold magenta",
                    box=box.ROUNDED,
                )
                final_table.add_column("#", style="dim", width=3)
                final_table.add_column("Ticker", style="cyan bold", width=8)
                final_table.add_column("Source", width=16)

                portfolio_upper = {t.upper() for t in portfolio_tickers}
                for i, t in enumerate(data, 1):
                    source = (
                        "[blue]Portfolio[/blue]"
                        if t.upper() in portfolio_upper
                        else "[green]Discovered[/green]"
                    )
                    final_table.add_row(str(i), t, source)
                console.print()
                console.print(final_table)

    console.print(f"\n[cyan]Screening for {num_picks} new stocks to research...[/cyan]")
    discovered = screener.screen_universe(
        portfolio_tickers,
        selections["analysis_date"],
        num_picks=num_picks,
        risk_profile=user_risk_profile,
        on_progress=_screener_progress,
    )

    if not discovered:
        console.print("[red]No stocks discovered. Try adjusting criteria.[/red]")
        return

    # --- Step 2: Parallel research with multi-stock display ---
    tickers = discovered
    start_time = time.time()

    # Create per-ticker message buffers
    ticker_buffers = {}
    aggregated_buffer = MessageBuffer()
    aggregated_buffer.init_for_analysis(selected_analyst_keys)

    for t in tickers:
        buf = MessageBuffer()
        buf.init_for_analysis(selected_analyst_keys)

        orig_update = buf.update_report_section

        def make_updater(t_val, original, b):
            def patched(sec, content):
                original(sec, content)
                if b.current_report:
                    aggregated_buffer.current_report = (
                        f"**[{t_val}]**\n{b.current_report}"
                    )

            return patched

        buf.update_report_section = make_updater(t, orig_update, buf)
        ticker_buffers[t] = buf

    # Track completion
    completed_tickers = set()
    all_results = {}

    # Focused ticker tracking (mutable container for thread-safe closure access)
    focused_ticker = {"value": tickers[0]}

    def create_multi_detail_layout():
        layout = Layout()
        overview_height = len(tickers) + 5
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="ticker_overview", size=overview_height),
            Layout(name="detail_messages", ratio=3),
            Layout(name="detail_analysis", ratio=5),
            Layout(name="footer", size=3),
        )
        return layout

    def render_ticker_overview():
        """Compact overview table showing all tickers with abbreviated status."""
        progress_table = Table(
            show_header=True,
            header_style="bold magenta",
            box=box.SIMPLE_HEAD,
            expand=True,
            padding=(0, 1),
        )
        progress_table.add_column(
            "Ticker", style="cyan bold", width=8, justify="center"
        )
        progress_table.add_column("Status", style="white", ratio=1)
        progress_table.add_column("Progress", style="green", width=12, justify="center")

        for t in tickers:
            buf = ticker_buffers[t]
            status_parts = []
            for agent, status in buf.agent_status.items():
                if status == "completed":
                    symbol = "[green]\u2713[/green]"
                elif status == "in_progress":
                    symbol = "[blue]\u25cf[/blue]"
                else:
                    symbol = "[dim]\u25cb[/dim]"
                short = agent.split()[0][:4]
                status_parts.append(f"{short}{symbol}")

            status_str = " ".join(status_parts)
            agents_done = sum(1 for s in buf.agent_status.values() if s == "completed")
            agents_total = len(buf.agent_status)

            if t in completed_tickers:
                progress_str = "[green]Done[/green]"
            else:
                progress_str = f"{agents_done}/{agents_total}"

            progress_table.add_row(t, status_str, progress_str)

        return Panel(
            progress_table,
            title="Research Progress",
            border_style="cyan",
            padding=(0, 1),
        )

    def update_multi_detail_display(layout):
        # Header
        completed = len(completed_tickers)
        total = len(tickers)
        layout["header"].update(
            Panel(
                f"[bold green]Overall Progress[/bold green] — "
                f"Researching {total} stocks | {completed}/{total}",
                border_style="green",
                padding=(0, 2),
            )
        )

        # Compact ticker overview
        layout["ticker_overview"].update(render_ticker_overview())

        # Detailed view for the aggregated tickers
        ft = focused_ticker["value"]

        layout["detail_messages"].update(
            render_messages_panel(aggregated_buffer, title="Aggregated Messages")
        )
        layout["detail_analysis"].update(
            render_analysis_panel(aggregated_buffer, title="Latest Report")
        )

        # Footer
        stats_parts = [
            f"Stocks: {len(completed_tickers)}/{len(tickers)}",
        ]
        if stats_handler:
            stats = stats_handler.get_stats()
            stats_parts.append(f"LLM: {stats['llm_calls']}")
            stats_parts.append(f"Tools: {stats['tool_calls']}")
            if stats["tokens_in"] > 0 or stats["tokens_out"] > 0:
                tokens_str = f"Tokens: {format_tokens(stats['tokens_in'])}\u2191 {format_tokens(stats['tokens_out'])}\u2193"
            else:
                tokens_str = "Tokens: --"
            stats_parts.append(tokens_str)
        elapsed = time.time() - start_time
        stats_parts.append(f"\u23f1 {int(elapsed // 60):02d}:{int(elapsed % 60):02d}")

        stats_table = Table(show_header=False, box=None, padding=(0, 2), expand=True)
        stats_table.add_column("Stats", justify="center")
        stats_table.add_row(" | ".join(stats_parts))
        layout["footer"].update(Panel(stats_table, border_style="grey50"))

    def on_chunk(ticker, chunk):
        """Thread-safe callback for streaming chunks."""
        # Auto-focus on the most recently active ticker
        focused_ticker["value"] = ticker
        buf = ticker_buffers[ticker]

        # Process messages (same as single-stock)
        if len(chunk.get("messages", [])) > 0:
            last_msg = chunk["messages"][-1]
            msg_id = getattr(last_msg, "id", None)
            if msg_id != buf._last_message_id:
                buf._last_message_id = msg_id
                msg_type, content = classify_message_type(last_msg)
                if content and content.strip():
                    buf.add_message(msg_type, content)
                    aggregated_buffer.add_message(msg_type, f"[{ticker}] {content}")
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        if isinstance(tool_call, dict):
                            buf.add_tool_call(tool_call["name"], tool_call["args"])
                            aggregated_buffer.add_tool_call(
                                tool_call["name"], f"[{ticker}] {tool_call['args']}"
                            )
                        else:
                            buf.add_tool_call(tool_call.name, tool_call.args)
                            aggregated_buffer.add_tool_call(
                                tool_call.name, f"[{ticker}] {tool_call.args}"
                            )

        # Update analyst statuses and report sections
        update_analyst_statuses(buf, chunk)

        # Trader — status + report section
        if chunk.get("composite_signal"):
            buf.update_report_section(
                "composite_signal", chunk["composite_signal"]
            )
            if buf.agent_status.get("Trader") != "completed":
                buf.update_agent_status("Trader", "completed")

    # Run parallel research with live display
    layout = create_multi_detail_layout()

    with Live(layout, refresh_per_second=4) as live:
        update_multi_detail_display(layout)

        all_results = graph.propagate_multi_streaming(
            tickers,
            selections["analysis_date"],
            on_chunk_callback=lambda t, c: (
                on_chunk(t, c),
                update_multi_detail_display(layout),
            ),
        )

        # Mark all as completed
        for t in tickers:
            completed_tickers.add(t)
            for agent in ticker_buffers[t].agent_status:
                ticker_buffers[t].update_agent_status(agent, "completed")
        update_multi_detail_display(layout)

    # --- Step 3: Post-analysis summary ---
    console.print("\n[bold cyan]All Research Complete![/bold cyan]\n")

    # Summary table
    summary_table = Table(
        title="Research Summary",
        show_header=True,
        header_style="bold magenta",
        box=box.ROUNDED,
    )
    summary_table.add_column("Ticker", style="cyan bold", width=8)
    summary_table.add_column("Decision", width=10, justify="center")
    summary_table.add_column("Status", width=10)

    ticker_signals = {}
    for t in tickers:
        state = all_results.get(t, {})
        if "error" in state:
            summary_table.add_row(t, "[red]ERROR[/red]", str(state["error"])[:50])
        else:
            signal = state.get("composite_signal", "")
            decision = graph.process_signal(signal) if signal else "N/A"
            color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(
                decision, "white"
            )
            summary_table.add_row(
                t, f"[{color}]{decision}[/{color}]", "[green]Complete[/green]"
            )
            if signal:
                ticker_signals[t] = signal

    console.print(summary_table)

    # --- Step 4: Portfolio Manager ---
    if not ticker_signals:
        console.print("[yellow]No valid signals to process.[/yellow]")
        return

    console.print("\n[cyan]Running Portfolio Manager analysis...[/cyan]")
    with console.status("[bold cyan]Portfolio Manager evaluating all signals..."):
        portfolio_result = graph.signal_processor.propose_portfolio_trades(
            ticker_signals,
            selections["alpaca_api_key"],
            selections["alpaca_secret_key"],
        )

    # Display portfolio reasoning
    console.print(
        Panel(
            Markdown(portfolio_result["portfolio_reasoning"]),
            title="Portfolio Strategy",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # Display individual orders
    orders = portfolio_result.get("orders", [])
    if orders:
        orders_table = Table(
            title="Proposed Orders",
            show_header=True,
            header_style="bold magenta",
            box=box.ROUNDED,
        )
        orders_table.add_column("Ticker", style="cyan bold", width=8)
        orders_table.add_column("Decision", width=8, justify="center")
        orders_table.add_column("Reasoning", ratio=1, no_wrap=False)
        orders_table.add_column("Order", width=30)

        for order in orders:
            decision = order.get("decision", "N/A")
            color = {"BUY": "green", "SELL": "red", "HOLD": "yellow"}.get(
                decision, "white"
            )
            params = order.get("order_params")
            if params:
                params_str = f"{params.get('side', '?')} {params.get('qty', params.get('notional', '?'))} @ {params.get('type', 'market')}"
            else:
                params_str = "[dim]No order[/dim]"

            reasoning = order.get("reasoning", "")

            orders_table.add_row(
                order.get("ticker", "?"),
                f"[{color}]{decision}[/{color}]",
                reasoning,
                params_str,
            )

        console.print(orders_table)

    # Ask to execute
    actionable = [o for o in orders if o.get("order_params") is not None]
    if actionable:
        console.print(
            f"\n[bold yellow]{len(actionable)} actionable order(s) ready.[/bold yellow]"
        )
        execute_choice = (
            typer.prompt("Execute all orders on Alpaca Paper Trading?", default="N")
            .strip()
            .upper()
        )
        if execute_choice in ("Y", "YES"):
            console.print("\n[cyan]Executing orders...[/cyan]")
            from src.graph.signal_processing import SignalProcessor

            exec_results = SignalProcessor.execute_portfolio_orders(
                actionable,
                selections["alpaca_api_key"],
                selections["alpaca_secret_key"],
            )
            for er in exec_results:
                status_color = "green" if "Error" not in er["result"] else "red"
                console.print(
                    f"  [{status_color}]{er['ticker']}:[/{status_color}] {er['result']}"
                )
        else:
            console.print("[yellow]Orders not executed.[/yellow]")
    else:
        console.print("\n[yellow]No actionable orders (all HOLD).[/yellow]")

    # Save reports
    save_choice = typer.prompt("\nSave reports?", default="Y").strip().upper()
    if save_choice in ("Y", "YES", ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        for t in tickers:
            state = all_results.get(t, {})
            if "error" not in state:
                save_path = Path.cwd() / "reports" / f"portfolio_{timestamp}" / t
                try:
                    save_report_to_disk(state, t, save_path)
                    console.print(f"  [green]\u2713 {t}:[/green] {save_path.resolve()}")
                except Exception as e:
                    console.print(f"  [red]{t}: Error saving: {e}[/red]")


@app.command()
def analyze():
    # Get selections first to check mode
    # We need to peek at mode, so we call get_user_selections which now includes mode
    # But run_analysis also calls get_user_selections, so we need to restructure
    # Instead, we'll route from run_analysis itself
    run_analysis()


if __name__ == "__main__":
    app()
