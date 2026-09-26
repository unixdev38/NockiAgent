"""Nocki Agent — command-line interface (Phase 1).

Run with:  python -m app.main   (or the `nocki` console script)

The CLI keeps *all* logic in the agent/graph layer; it only handles I/O,
slash-commands and rendering. This respects the rule "no business logic in
main.py".
"""

from __future__ import annotations

import sys
import uuid

from langchain_core.messages import HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.agent.graph import AgentRuntime
from app.config import get_settings, reload_settings
from app.llm.base import ModelNotFoundError
from app.llm.ollama import build_llm
from app.logging_config import setup_logging

console = Console()

BANNER = """[bold cyan]
╭──────────────────────────────╮
│        Nocki Agent           │
│        Local AI Agent        │
╰──────────────────────────────╯[/bold cyan]"""

HELP = """\
[bold]Commandes disponibles[/bold]
  /help        Afficher cette aide
  /tools       Lister les outils disponibles
  /model       Afficher / changer le modèle (ex: /model qwen3)
  /workspace   Afficher le workspace
  /clear       Effacer la conversation courante
  /history     Afficher l'historique de la session
  /exit        Quitter
"""

# Phase 1 has no tools yet; kept explicit so /tools reflects reality.
AVAILABLE_TOOLS: list[str] = []


class NockiCLI:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.ensure_directories()
        setup_logging(self.settings.log_level.value)
        self.llm = build_llm(self.settings)
        self.runtime = AgentRuntime(
            self.llm, self.settings, available_tools=AVAILABLE_TOOLS
        )
        self.session_id = uuid.uuid4().hex[:8]

    # -- rendering -----------------------------------------------------------
    def print_banner(self) -> None:
        console.print(BANNER)
        console.print(f"[dim]Model:[/dim]     {self.settings.ollama_model}")
        console.print(f"[dim]Workspace:[/dim] {self.settings.workspace}")
        console.print(f"[dim]Session:[/dim]   {self.session_id}")
        console.print("[dim]Tape /help pour les commandes.[/dim]\n")

    def check_backend(self) -> None:
        if not self.llm.is_available():
            console.print(
                Panel.fit(
                    f"[yellow]⚠ Ollama injoignable sur {self.settings.ollama_base_url}.[/yellow]\n"
                    "Lance le serveur avec [bold]ollama serve[/bold], puis "
                    f"[bold]ollama pull {self.settings.ollama_model}[/bold].\n"
                    "Le CLI reste utilisable ; les réponses échoueront proprement "
                    "tant qu'Ollama est hors ligne.",
                    title="Backend",
                    border_style="yellow",
                )
            )
            return
        try:
            self.llm.ensure_model()
        except ModelNotFoundError as exc:
            console.print(
                Panel.fit(f"[yellow]{exc}[/yellow]", title="Modèle", border_style="yellow")
            )

    # -- slash commands ------------------------------------------------------
    def handle_command(self, line: str) -> bool:
        """Handle a /command. Return False to exit the REPL, True otherwise."""
        parts = line.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in ("/exit", "/quit"):
            console.print("[dim]À bientôt 👋[/dim]")
            return False
        if cmd == "/help":
            console.print(HELP)
        elif cmd == "/tools":
            self._show_tools()
        elif cmd == "/model":
            self._model(arg)
        elif cmd == "/workspace":
            console.print(f"[dim]Workspace:[/dim] {self.settings.workspace}")
        elif cmd == "/clear":
            self.session_id = uuid.uuid4().hex[:8]
            console.print(f"[green]Conversation réinitialisée[/green] (session {self.session_id}).")
        elif cmd == "/history":
            self._show_history()
        else:
            console.print(f"[yellow]Commande inconnue: {cmd}[/yellow] — tape /help.")
        return True

    def _show_tools(self) -> None:
        if not AVAILABLE_TOOLS:
            console.print(
                "[dim]Aucun outil disponible (Phase 1). Les outils arrivent en Phase 2.[/dim]"
            )
            return
        table = Table(title="Outils")
        table.add_column("Nom")
        for tool in AVAILABLE_TOOLS:
            table.add_row(tool)
        console.print(table)

    def _model(self, arg: str) -> None:
        if not arg:
            console.print(f"[dim]Modèle actuel:[/dim] {self.settings.ollama_model}")
            return
        import os

        os.environ["OLLAMA_MODEL"] = arg
        self.settings = reload_settings()
        self.llm = build_llm(self.settings)
        self.runtime = AgentRuntime(self.llm, self.settings, available_tools=AVAILABLE_TOOLS)
        console.print(f"[green]Modèle changé:[/green] {arg}")
        self.check_backend()

    def _show_history(self) -> None:
        messages = self.runtime.history(self.session_id)
        if not messages:
            console.print("[dim]Historique vide.[/dim]")
            return
        for msg in messages:
            role = "You" if isinstance(msg, HumanMessage) else "Nocki"
            style = "cyan" if isinstance(msg, HumanMessage) else "green"
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            console.print(f"[bold {style}]{role}:[/bold {style}] {text}")

    # -- main loop -----------------------------------------------------------
    def chat(self, message: str) -> None:
        console.print("[bold green]Nocki:[/bold green] ", end="")
        collected: list[str] = []
        for chunk in self.runtime.stream(message, session_id=self.session_id):
            collected.append(chunk)
            # markup/highlight off: streamed model text must be printed verbatim
            # (brackets, backticks, etc. are content, not Rich markup).
            console.print(chunk, end="", markup=False, highlight=False)
        console.print()  # newline after stream

    def run(self) -> int:
        self.print_banner()
        self.check_backend()
        while True:
            try:
                line = console.input("[bold cyan]You:[/bold cyan] ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]À bientôt 👋[/dim]")
                return 0
            if not line:
                continue
            if line.startswith("/"):
                if not self.handle_command(line):
                    return 0
                continue
            try:
                self.chat(line)
            except KeyboardInterrupt:
                console.print("\n[yellow]Interrompu.[/yellow]")
            except Exception as exc:  # noqa: BLE001 — never crash the REPL
                console.print(f"\n[red]Erreur inattendue:[/red] {exc}")


def main() -> int:
    try:
        return NockiCLI().run()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Échec du démarrage:[/red] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
