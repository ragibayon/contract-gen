from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterator, Mapping, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from rich.theme import Theme
from rich.traceback import Traceback

CONTRACT_AGENT_THEME = Theme(
    {
        "info": "cyan",
        "success": "bold green",
        "warning": "bold yellow",
        "error": "bold red",
        "title": "bold magenta",
        "muted": "dim",
        "path": "bold blue",
        "task": "bold cyan",
        "code": "green",
    }
)


@dataclass
class RichLogger:
    """
    Shared Rich logger for ContractAgent.

    This avoids declaring Console/style logic in every script.
    """

    console: Console

    @contextmanager
    def timed(self, label: str) -> Iterator[None]:
        start = perf_counter()
        self.info(f"{label} started")

        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.success(f"{label} finished in {elapsed:.2f}s")

    def rule(self, title: str) -> None:
        self.console.print(Rule(Text(title, style="title")))

    def info(self, message: str) -> None:
        self.console.print(Text(f"ℹ {message}", style="info"))

    def success(self, message: str) -> None:
        self.console.print(Text(f"✓ {message}", style="success"))

    def warning(self, message: str) -> None:
        self.console.print(Text(f"⚠ {message}", style="warning"))

    def error(self, message: str) -> None:
        self.console.print(Text(f"✗ {message}", style="error"))

    def muted(self, message: str) -> None:
        self.console.print(Text(message, style="muted"))

    @staticmethod
    def _value(value: Any) -> Text:
        return Text(str(value))

    def task_header(
        self,
        task_id: str,
        entry_point: str | None = None,
        task: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        table = Table.grid(padding=(0, 2))
        table.add_column(style="task", justify="right")
        table.add_column()

        table.add_row(self._value("Task ID"), self._value(task_id))
        table.add_row(
            self._value("Entry point"),
            self._value(entry_point or "not specified"),
        )

        for key, value in (metadata or {}).items():
            table.add_row(self._value(key), self._value(value))

        self.console.print(
            Panel(
                table,
                title="[title]Task[/title]",
                border_style="cyan",
            )
        )

        if task:
            self.console.print(
                Panel(
                    Text(task),
                    title="[title]Task prompt[/title]",
                    border_style="cyan",
                )
            )

    def stage_started(
        self,
        name: str,
        mode: str = "generated",
        details: dict[str, Any] | None = None,
    ) -> None:
        context = [f"Stage: {name}", f"mode={mode}"]
        context.extend(f"{key}={value}" for key, value in (details or {}).items())
        self.rule(" | ".join(context))

    def stage_attempt(
        self,
        name: str,
        attempt: int,
        total: int,
        repairing: bool = False,
    ) -> None:
        suffix = " (repair)" if repairing else ""
        self.info(f"{name}: attempt {attempt}/{total}{suffix}")

    def stage_finished(
        self,
        name: str,
        success: bool,
        elapsed_sec: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        status = "completed" if success else "failed"
        message = f"{name}: {status} in {elapsed_sec:.3f}s"
        if details:
            detail_text = " | ".join(f"{key}={value}" for key, value in details.items())
            message = f"{message} | {detail_text}"
        if success:
            self.success(message)
        else:
            self.error(message)

    def result_status(self, passed: bool, label: str = "Execution result") -> None:
        if passed:
            self.success(f"{label}: PASSED")
        else:
            self.error(f"{label}: FAILED")

    def code(
        self,
        code: str,
        title: str = "Code",
        language: str = "python",
        line_numbers: bool = False,
    ) -> None:
        syntax = Syntax(
            code,
            language,
            theme="monokai",
            line_numbers=line_numbers,
            word_wrap=True,
        )

        self.console.print(
            Panel(
                syntax,
                title=f"[title]{title}[/title]",
                border_style="green",
            )
        )

    def text_panel(
        self,
        text: str,
        title: str,
        border_style: str = "blue",
    ) -> None:
        self.console.print(
            Panel(
                text,
                title=f"[title]{title}[/title]",
                border_style=border_style,
            )
        )

    def exception(self, error: BaseException | None = None) -> None:
        if error is not None:
            self.error(f"{type(error).__name__}: {error}")

        self.console.print(Traceback(show_locals=False))

    def kv(self, title: str, values: dict[str, Any]) -> None:
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Key", style="cyan")
        table.add_column("Value")

        for key, value in values.items():
            table.add_row(self._value(key), self._value(value))

        self.console.print(table)

    def records(
        self,
        title: str,
        rows: Sequence[Mapping[str, Any]],
        columns: Mapping[str, str],
    ) -> None:
        table = Table(
            title=title,
            show_header=True,
            header_style="bold magenta",
            expand=True,
        )
        for label in columns.values():
            table.add_column(label)

        for row in rows:
            table.add_row(*(self._value(row.get(key, "")) for key in columns))

        self.console.print(table)


console = Console(theme=CONTRACT_AGENT_THEME)
logger = RichLogger(console=console)
