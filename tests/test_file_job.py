"""Tests de file_job.py — cycle de vie, chemin de sortie, annulation, erreurs."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import audio_convert
import file_job
from file_job import (
    FileJob,
    FileJobCallbacks,
    JobState,
    resolve_output_path,
)
from file_transcriber import FileTranscript
from transcriber import Segment


@pytest.fixture
def cb():
    return FileJobCallbacks(
        on_progress=MagicMock(name="on_progress"),
        on_done=MagicMock(name="on_done"),
    )


@pytest.fixture
def job(cb):
    """Job dont la « transcription » est exécutée en direct (pas de worker)."""
    return FileJob(callbacks=cb, run_transcription=lambda work: work())


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Neutralise conversion + transcription : on teste le job, pas le modèle."""
    state = {
        "duration": 120.0,
        "transcript": FileTranscript(
            segments=[Segment(start=0.0, end=5.0, text="Bonjour à tous.")],
            language="fr",
            duration_s=120.0,
        ),
    }

    monkeypatch.setattr(
        "file_job.audio_convert.probe_duration", lambda _p: state["duration"]
    )
    monkeypatch.setattr(
        "file_job.audio_convert.ensure_16k_mono", lambda p: (p, False)
    )
    monkeypatch.setattr(
        "file_job.file_transcriber.transcribe_file",
        lambda **kwargs: state["transcript"],
    )
    return state


def _source(tmp_path: Path, name: str = "reunion.m4a") -> Path:
    path = tmp_path / name
    path.write_bytes(b"faux audio")
    return path


def _submit_and_wait(job: FileJob, cb, source: Path, tmp_path: Path, **kw) -> None:
    kw.setdefault("output_dir", tmp_path / "out")
    job.submit(
        source=source,
        transcriber=MagicMock(name="transcriber"),
        model_name="mlx-community/whisper-large-v3-mlx",
        **kw,
    )
    deadline = time.time() + 5.0
    while not cb.on_done.called and time.time() < deadline:
        time.sleep(0.01)
    assert cb.on_done.called, "le job ne s'est jamais terminé"


# ---- Cycle nominal ----


def test_successful_job_writes_txt(job, cb, stub_pipeline, tmp_path: Path):
    source = _source(tmp_path)
    _submit_and_wait(job, cb, source, tmp_path)

    result = cb.on_done.call_args[0][0]
    assert result.state is JobState.DONE
    assert result.output_path.name == "reunion.txt"
    assert result.output_path.exists()
    assert "Bonjour à tous." in result.output_path.read_text(encoding="utf-8")


def test_txt_includes_header_and_timestamps(job, cb, stub_pipeline, tmp_path: Path):
    _submit_and_wait(job, cb, _source(tmp_path), tmp_path)
    content = cb.on_done.call_args[0][0].output_path.read_text(encoding="utf-8")
    assert "Transcription — reunion.m4a" in content
    assert "[00:00:00]" in content


def test_timestamps_can_be_disabled(job, cb, stub_pipeline, tmp_path: Path):
    _submit_and_wait(
        job, cb, _source(tmp_path), tmp_path, include_timestamps=False
    )
    content = cb.on_done.call_args[0][0].output_path.read_text(encoding="utf-8")
    assert "Bonjour à tous." in content
    assert "[00:00:00]" not in content


def test_state_returns_to_done_and_name_cleared(
    job, cb, stub_pipeline, tmp_path: Path
):
    _submit_and_wait(job, cb, _source(tmp_path), tmp_path)
    assert job.state is JobState.DONE
    assert job.is_running is False
    assert job.current_name is None


def test_no_part_file_left_behind(job, cb, stub_pipeline, tmp_path: Path):
    """L'écriture passe par un .part renommé : aucun résidu à la fin."""
    _submit_and_wait(job, cb, _source(tmp_path), tmp_path)
    assert list((tmp_path / "out").glob("*.part")) == []


# ---- Un seul job à la fois ----


def test_second_submit_is_refused_while_running(cb, stub_pipeline, tmp_path: Path):
    gate = threading.Event()
    job = FileJob(
        callbacks=cb,
        run_transcription=lambda work: (gate.wait(timeout=5), work())[1],
    )
    source = _source(tmp_path)

    first = job.submit(
        source=source, transcriber=MagicMock(), model_name="m",
        output_dir=tmp_path / "out",
    )
    # Laisse le thread démarrer.
    deadline = time.time() + 2.0
    while not job.is_running and time.time() < deadline:
        time.sleep(0.01)

    second = job.submit(
        source=source, transcriber=MagicMock(), model_name="m",
        output_dir=tmp_path / "out",
    )

    assert first is True
    assert second is False
    gate.set()


# ---- Annulation ----


def test_cancelled_job_keeps_the_partial(job, cb, stub_pipeline, tmp_path: Path):
    """Jeter 90 % d'un job de 55 minutes serait inacceptable : le partiel est
    écrit, avec un en-tête qui lève toute ambiguïté."""
    stub_pipeline["transcript"] = FileTranscript(
        segments=[Segment(start=0.0, end=5.0, text="Début seulement.")],
        language="fr",
        duration_s=3600.0,
        cancelled_at_s=2952.0,
    )

    _submit_and_wait(job, cb, _source(tmp_path), tmp_path)

    result = cb.on_done.call_args[0][0]
    assert result.state is JobState.CANCELLED
    content = result.output_path.read_text(encoding="utf-8")
    assert "Début seulement." in content
    assert "[Transcription interrompue à 00:49:12]" in content


def test_cancel_sets_the_event(job):
    job.cancel()
    assert job._cancel.is_set() is True


# ---- Refus et erreurs ----


def test_file_too_long_is_refused_before_conversion(
    job, cb, stub_pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Inutile de payer la conversion d'un fichier de 6 h choisi par erreur."""
    stub_pipeline["duration"] = 6 * 3600
    convert = MagicMock(return_value=(tmp_path / "x", False))
    monkeypatch.setattr("file_job.audio_convert.ensure_16k_mono", convert)

    _submit_and_wait(
        job, cb, _source(tmp_path), tmp_path, max_duration_s=14_400
    )

    result = cb.on_done.call_args[0][0]
    assert result.state is JobState.FAILED
    assert "limite" in result.error
    convert.assert_not_called()


def test_unreadable_duration_is_reported(
    job, cb, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr("file_job.audio_convert.probe_duration", lambda _p: None)

    _submit_and_wait(job, cb, _source(tmp_path), tmp_path)

    result = cb.on_done.call_args[0][0]
    assert result.state is JobState.FAILED
    assert "durée" in result.error


def test_conversion_failure_is_reported_once(
    job, cb, stub_pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "file_job.audio_convert.ensure_16k_mono",
        MagicMock(side_effect=audio_convert.AudioConversionError("format inconnu")),
    )

    _submit_and_wait(job, cb, _source(tmp_path), tmp_path)

    assert cb.on_done.call_count == 1
    result = cb.on_done.call_args[0][0]
    assert result.state is JobState.FAILED
    assert "format inconnu" in result.error


def test_transcription_failure_is_reported(
    job, cb, stub_pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        "file_job.file_transcriber.transcribe_file",
        MagicMock(side_effect=RuntimeError("modèle indisponible")),
    )

    _submit_and_wait(job, cb, _source(tmp_path), tmp_path)

    result = cb.on_done.call_args[0][0]
    assert result.state is JobState.FAILED
    assert "modèle indisponible" in result.error


def test_temp_audio_is_deleted_even_on_failure(
    job, cb, stub_pipeline, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    temp = tmp_path / "converti.wav"
    temp.write_bytes(b"x")
    monkeypatch.setattr(
        "file_job.audio_convert.ensure_16k_mono", lambda _p: (temp, True)
    )
    monkeypatch.setattr(
        "file_job.file_transcriber.transcribe_file",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    _submit_and_wait(job, cb, _source(tmp_path), tmp_path)

    assert not temp.exists()


# ---- resolve_output_path ----


def test_output_path_uses_source_stem(tmp_path: Path):
    out = resolve_output_path(Path("/a/reunion.m4a"), tmp_path)
    assert out.name == "reunion.txt"
    assert out.parent == tmp_path


def test_output_path_avoids_collision(tmp_path: Path):
    """On n'écrase jamais un transcript existant."""
    (tmp_path / "reunion.txt").write_text("déjà là", encoding="utf-8")
    out = resolve_output_path(Path("/a/reunion.m4a"), tmp_path)
    assert out.name == "reunion-2.txt"


def test_output_path_increments_until_free(tmp_path: Path):
    for name in ("reunion.txt", "reunion-2.txt", "reunion-3.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    out = resolve_output_path(Path("/a/reunion.m4a"), tmp_path)
    assert out.name == "reunion-4.txt"


def test_output_dir_is_created(tmp_path: Path):
    target = tmp_path / "profond" / "Voxtral"
    resolve_output_path(Path("/a/x.m4a"), target)
    assert target.is_dir()


def test_unwritable_dir_falls_back(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Mieux vaut un fichier ailleurs qu'un job de 40 minutes perdu à la
    dernière seconde."""
    fallback = tmp_path / "repli"
    fallback.mkdir()
    monkeypatch.setattr(file_job, "_FALLBACK_DIR", fallback)

    real_mkdir = Path.mkdir

    def refusing_mkdir(self: Path, **kwargs):  # noqa: ANN202
        if "interdit" in str(self):
            raise OSError("read-only file system")
        return real_mkdir(self, **kwargs)

    monkeypatch.setattr(Path, "mkdir", refusing_mkdir)

    out = resolve_output_path(Path("/a/x.m4a"), Path("/interdit/nope"))

    assert out.parent == fallback
    assert out.name == "x.txt"
