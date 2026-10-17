"""Uploads sent from the chat UI, stored inside the workspace.

Design rules:
- files land ONLY in `<workspace>/<upload_subdir>/` (validated, no traversal),
- the filename is reduced to a safe basename, extensions are allow-listed,
- size is capped by configuration (`UPLOAD_MAX_BYTES`),
- PDF and DOCX are converted to a UTF-8 `.txt` sidecar so the agent can read
  them with the existing `read_file` tool (which refuses binary content).
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from pydantic import BaseModel

#: Plain-text formats stored as-is.
TEXT_EXTENSIONS = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".csv",
        ".tsv",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".cfg",
        ".log",
        ".xml",
        ".html",
        ".htm",
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".java",
        ".c",
        ".h",
        ".cpp",
        ".hpp",
        ".cs",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".sql",
        ".sh",
        ".ps1",
        ".bat",
    }
)
#: Binary documents converted to text on upload.
CONVERTIBLE_EXTENSIONS = frozenset({".pdf", ".docx"})

_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._ -]")
_MAX_NAME_LENGTH = 120


class UploadError(ValueError):
    """Upload refused (type, size, empty file...)."""


class UploadedFile(BaseModel):
    name: str  # stored basename
    path: str  # path relative to the workspace (usable with read_file)
    size: int
    extracted_path: str | None = None  # text sidecar for PDF/DOCX


def safe_filename(raw: str) -> str:
    """Reduce a client-provided name to a safe basename (never empty)."""
    base = Path(raw.replace("\\", "/")).name.strip()
    base = _UNSAFE_CHARS.sub("_", base).strip(" .")
    if not base:
        raise UploadError("nom de fichier invalide")
    if len(base) > _MAX_NAME_LENGTH:
        stem, dot, ext = base.rpartition(".")
        keep = _MAX_NAME_LENGTH - len(ext) - 1
        base = f"{stem[:keep]}.{ext}" if dot else base[:_MAX_NAME_LENGTH]
    return base


def extract_text(data: bytes, extension: str) -> str:
    """Extract plain text from a PDF or DOCX payload (raises UploadError on failure)."""
    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover — depends on the install
            raise UploadError("lecture PDF indisponible (installez 'pypdf')") from exc
        try:
            reader = PdfReader(io.BytesIO(data))
            pages = [(page.extract_text() or "") for page in reader.pages]
        except Exception as exc:  # noqa: BLE001 — corrupted PDF
            raise UploadError(f"PDF illisible: {exc}") from exc
        return "\n\n".join(p.strip() for p in pages if p.strip())
    if extension == ".docx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                xml = archive.read("word/document.xml")
        except (zipfile.BadZipFile, KeyError) as exc:
            raise UploadError("DOCX illisible") from exc
        root = ElementTree.fromstring(xml)  # noqa: S314 — local, trusted-size payload
        ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
        paragraphs = []
        for para in root.iter(f"{ns}p"):
            paragraphs.append("".join(node.text or "" for node in para.iter(f"{ns}t")))
        return "\n".join(p for p in paragraphs if p.strip())
    raise UploadError(f"format non convertible: {extension}")


class UploadStore:
    """Save, list and convert uploaded documents inside the workspace."""

    def __init__(self, uploads_dir: Path, workspace: Path, max_bytes: int) -> None:
        self.uploads_dir = uploads_dir.resolve()
        self.workspace = workspace.resolve()
        self.max_bytes = max_bytes

    def _rel(self, path: Path) -> str:
        return path.resolve().relative_to(self.workspace).as_posix()

    def save(self, raw_name: str, data: bytes) -> UploadedFile:
        name = safe_filename(raw_name)
        extension = Path(name).suffix.lower()
        if not data:
            raise UploadError("fichier vide")
        if len(data) > self.max_bytes:
            raise UploadError(f"fichier trop volumineux (max {self.max_bytes} octets)")
        if extension not in TEXT_EXTENSIONS | CONVERTIBLE_EXTENSIONS:
            raise UploadError(f"type de fichier non autorisé: {extension or '(sans extension)'}")

        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        target = self._unique_path(self.uploads_dir / name)
        extracted: str | None = None

        if extension in CONVERTIBLE_EXTENSIONS:
            text = extract_text(data, extension)
            if not text.strip():
                raise UploadError("aucun texte extractible (document scanné ?)")
            target.write_bytes(data)
            sidecar = target.with_name(target.name + ".txt")
            sidecar.write_text(text, encoding="utf-8")
            extracted = self._rel(sidecar)
        else:
            try:
                data.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise UploadError("fichier texte non UTF-8 (binaire ?)") from exc
            target.write_bytes(data)

        return UploadedFile(
            name=target.name,
            path=self._rel(target),
            size=len(data),
            extracted_path=extracted,
        )

    def list(self) -> list[UploadedFile]:
        if not self.uploads_dir.is_dir():
            return []
        files: list[UploadedFile] = []
        for path in sorted(self.uploads_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or self._is_sidecar(path):
                continue  # text sidecars are listed through their original document
            sidecar = path.with_name(path.name + ".txt")
            files.append(
                UploadedFile(
                    name=path.name,
                    path=self._rel(path),
                    size=path.stat().st_size,
                    extracted_path=self._rel(sidecar) if sidecar.is_file() else None,
                )
            )
        return files

    def _is_sidecar(self, path: Path) -> bool:
        """`<doc>.pdf.txt` is the extracted text of `<doc>.pdf` (hidden from the list)."""
        if path.suffix != ".txt":
            return False
        original = path.with_name(path.name[: -len(".txt")])
        return Path(original).suffix.lower() in CONVERTIBLE_EXTENSIONS and original.is_file()

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        stem, suffix = path.stem, path.suffix
        for index in range(2, 1000):
            candidate = path.with_name(f"{stem}-{index}{suffix}")
            if not candidate.exists():
                return candidate
        raise UploadError("trop de fichiers portant le même nom")
