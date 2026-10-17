"""Uploads: filename safety, type/size limits, text + DOCX/PDF extraction."""

from __future__ import annotations

import io
import zipfile

import pytest

from app.documents.store import UploadError, UploadStore, safe_filename


def _docx(paragraphs: list[str]) -> bytes:
    body = "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paragraphs)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", xml)
    return buffer.getvalue()


@pytest.fixture
def store(tmp_path):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    return UploadStore(workspace / "uploads", workspace, max_bytes=1_000)


def test_safe_filename_strips_paths_and_unsafe_chars():
    assert safe_filename("../../etc/passwd.txt") == "passwd.txt"
    assert safe_filename("C:\\Users\\me\\notes?.md") == "notes_.md"
    with pytest.raises(UploadError):
        safe_filename("...")


def test_save_text_file_inside_uploads(store):
    saved = store.save("notes.md", b"# Titre\nbonjour")
    assert saved.path == "uploads/notes.md"
    assert saved.extracted_path is None
    assert (store.workspace / saved.path).read_text(encoding="utf-8").startswith("# Titre")


def test_duplicate_names_are_suffixed(store):
    first = store.save("a.txt", b"one")
    second = store.save("a.txt", b"two")
    assert first.name == "a.txt" and second.name == "a-2.txt"


def test_refuses_unknown_extension_and_binary_text(store):
    with pytest.raises(UploadError, match="non autorisé"):
        store.save("tool.exe", b"MZ")
    with pytest.raises(UploadError, match="UTF-8"):
        store.save("data.txt", b"\xff\xfe\x00bin")


def test_refuses_empty_and_oversized(store):
    with pytest.raises(UploadError, match="vide"):
        store.save("a.txt", b"")
    with pytest.raises(UploadError, match="volumineux"):
        store.save("a.txt", b"x" * 1_001)


def test_docx_is_converted_to_text_sidecar(store):
    saved = store.save("rapport.docx", _docx(["Premier paragraphe", "Second"]))
    assert saved.path == "uploads/rapport.docx"
    assert saved.extracted_path == "uploads/rapport.docx.txt"
    text = (store.workspace / saved.extracted_path).read_text(encoding="utf-8")
    assert text == "Premier paragraphe\nSecond"


def test_corrupted_docx_is_refused(store):
    with pytest.raises(UploadError, match="DOCX illisible"):
        store.save("bad.docx", b"not a zip")


def test_list_hides_sidecars_and_reports_extracted_path(store):
    store.save("a.txt", b"hello")
    store.save("rapport.docx", _docx(["texte"]))
    listed = {f.name: f for f in store.list()}
    assert set(listed) == {"a.txt", "rapport.docx"}
    assert listed["rapport.docx"].extracted_path == "uploads/rapport.docx.txt"
    assert listed["a.txt"].extracted_path is None


def test_list_empty_when_no_uploads(store):
    assert store.list() == []
