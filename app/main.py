"""Nocki Agent — command-line interface.

Run with:  python -m app.main   (or the `nocki` console script)

The CLI is a *thin* layer on top of `NockiService` (the same core the HTTP
API uses): it only handles I/O, slash-commands and rendering. No business
logic here — that is a hard project rule.
"""

from __future__ import annotations

import sys
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.agent.graph import AgentEvent, EventType
from app.config import get_settings
from app.llm.base import ModelNotFoundError
from app.logging_config import setup_logging
from app.service import NockiService

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
  /sessions    Lister les sessions enregistrées
  /session     Basculer vers une session existante (ex: /session abc123)
  /clear       Nouvelle conversation
  /history     Historique de la session (persistant)
  /exit        Quitter
"""


class NockiCLI:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.settings.ensure_directories()
        setup_logging(self.settings.log_level.value)
        self.service = NockiService(self.settings)
        self.session_id = uuid.uuid4().hex[:8]

    # -- rendering -----------------------------------------------------------
    def print_banner(self) -> None:
        console.print(BANNER)
        console.print(f"[dim]Model:[/dim]     {self.service.llm.model}")
        console.print(f"[dim]Workspace:[/dim] {self.settings.workspace}")
        console.print(f"[dim]Mémoire:[/dim]   {self.settings.database_path}")
        console.print(f"[dim]Tools:[/dim]     {', '.join(self.service.registry.names)}")
        console.print(f"[dim]Session:[/dim]   {self.session_id}")
        console.print("[dim]Tape /help pour les commandes.[/dim]\n")

    def check_backend(self) -> None:
        if not self.service.llm.is_available():
            console.print(
                Panel.fit(
                    f"[yellow]⚠ Ollama injoignable sur {self.settings.ollama_base_url}.[/yellow]\n"
                    "Lance le serveur avec [bold]ollama serve[/bold], puis "
                    f"[bold]ollama pull {self.service.llm.model}[/bold].\n"
                    "Le CLI reste utilisable ; les réponses échoueront proprement "
                    "tant qu'Ollama est hors ligne.",
                    title="Backend",
                    border_style="yellow",
                )
            )
            return
        try:
            self.service.llm.ensure_model()
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
            console.print(f"[green]Nouvelle conversation[/green] (session {self.session_id}).")
        elif cmd == "/history":
            self._show_history()
        elif cmd == "/sessions":
            self._show_sessions()
        elif cmd == "/session":
            self._switch_session(arg)
        else:
            console.print(f"[yellow]Commande inconnue: {cmd}[/yellow] — tape /help.")
        return True

    def _show_tools(self) -> None:
        table = Table(title="Outils")
        table.add_column("Nom", style="cyan")
        table.add_column("Permission")
        table.add_column("Description", overflow="fold")
        for spec in self.service.tool_specs():
            table.add_row(spec["name"], spec["permission"], spec["description"])
        console.print(table)

    def _model(self, arg: str) -> None:
        if not arg:
            console.print(f"[dim]Modèle actuel:[/dim] {self.service.llm.model}")
            return
        self.service.set_model(arg)
        console.print(f"[green]Modèle changé:[/green] {arg}")
        self.check_backend()

    def _show_sessions(self) -> None:
        sessions = self.service.list_sessions()
        if not sessions:
            console.print("[dim]Aucune session enregistrée.[/dim]")
            return
        table = Table(title="Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Mise à jour")
        table.add_column("Messages", justify="right")
        for info in sessions:
            marker = " (courante)" if info.id == self.session_id else ""
            updated = info.updated_at.strftime("%Y-%m-%d %H:%M")
            table.add_row(info.id + marker, updated, str(info.message_count))
        console.print(table)

    def _switch_session(self, arg: str) -> None:
        if not arg:
            console.print("Usage: /session <id>  (voir /sessions)")
            return
        if self.service.get_session(arg) is None:
            console.print(f"[yellow]Session inconnue: {arg}[/yellow] — voir /sessions.")
            return
        self.session_id = arg
        console.print(f"[green]Session active:[/green] {arg} (/history pour le contexte).")

    def _show_history(self) -> None:
        messages = self.service.runtime.history(self.session_id)
        if not messages:
            console.print("[dim]Historique vide.[/dim]")
            return
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role, style = "You", "cyan"
            elif isinstance(msg, AIMessage):
                role, style = "Nocki", "green"
            else:
                role, style = type(msg).__name__, "dim"
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            console.print(f"{role}:", style=f"bold {style}", end="")
            console.print(f" {text}", markup=False, highlight=False)

    # -- chat ------------------------------------------------------------------
    def _render_event(self, event: AgentEvent) -> None:
        """Render one agent event. Model/tool text is printed verbatim."""
        if event.type is EventType.TOOL_CALL:
            console.print(f"⚙ {event.text}", style="dim cyan", markup=False, highlight=False)
        elif event.type is EventType.TOOL_RESULT:
            preview = event.text if len(event.text) <= 400 else event.text[:400] + " …"
            console.print(
                f"← {event.tool}: {preview}",
                style="dim",
                markup=False,
                highlight=False,
            )
        elif event.type is EventType.ERROR:
            console.print(f"⚠ {event.text}", style="red", markup=False, highlight=False)
        elif event.type is EventType.FINAL:
            console.print("Nocki: ", style="bold green", end="")
            console.print(event.text, markup=False, highlight=False)

    def chat(self, message: str) -> None:
        for event in self.service.stream_events(message, session_id=self.session_id):
            self._render_event(event)

    # -- main loop -----------------------------------------------------------
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
