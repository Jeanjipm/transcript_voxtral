"""
Cycle de vie d'un job de transcription de fichier.

Assemble les pièces : conversion audio → boucle de transcription → écriture
du `.txt`. Gère l'annulation, la progression, et la résolution du chemin de
sortie.

Un seul job à la fois, volontairement : deux jobs concurrents se
disputeraient l'inference-worker sans rien accélérer, et la progression
affichée dans la menu bar deviendrait ambiguë.

## Contrat de threads

`submit()` est appelé depuis le main thread (clic de menu) et rend la main
immédiatement. Le travail se fait sur un thread `file-job` : conversion
`afconvert`, lecture du fichier, écriture du `.txt`. La transcription
elle-même est déléguée à l'inference-worker via `transcribe_block`, en
priorité basse, pour qu'une dictée puisse toujours passer devant.

Les callbacks de progression et de fin sont invoqués depuis le thread
`file-job` ; les implémentations qui touchent Cocoa doivent repasser sur le
main thread.
"""

from __future__ import annotations

import enum
import os
import threading
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import audio_convert
import file_transcriber
from file_transcriber import FileTranscript
from transcriber import Transcriber


# Suffixe d'écriture atomique : on écrit là, puis on renomme. Un crash en
# cours d'écriture ne laisse donc jamais un .txt tronqué qui aurait l'air
# complet.
_PART_SUFFIX = ".part"

# Repli quand le dossier de sortie configuré n'est pas inscriptible.
_FALLBACK_DIR = Path.home() / "Documents"


class JobState(enum.Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class JobResult:
    state: JobState
    output_path: Path | None = None
    transcript: FileTranscript | None = None
    error: str | None = None


@dataclass
class FileJobCallbacks:
    """Points de sortie vers l'UI. Invoqués depuis le thread `file-job`."""

    # (secondes traitées, secondes totales)
    on_progress: Callable[[float, float], None]
    on_done: Callable[[JobResult], None]


class FileJob:
    """Un job de transcription de fichier à la fois."""

    def __init__(
        self,
        callbacks: FileJobCallbacks,
        # Exécute la transcription là où il faut (typiquement : mise en file
        # sur l'inference-worker en priorité basse, puis attente du résultat).
        run_transcription: Callable[[Callable[[], FileTranscript]], FileTranscript],
    ) -> None:
        self._cb = callbacks
        self._run_transcription = run_transcription
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._state = JobState.IDLE
        self._current_name: str | None = None

    # ---- État ----

    @property
    def state(self) -> JobState:
        with self._lock:
            return self._state

    @property
    def is_running(self) -> bool:
        return self.state is JobState.RUNNING

    @property
    def current_name(self) -> str | None:
        with self._lock:
            return self._current_name

    # ---- Contrôle ----

    def submit(
        self,
        source: Path,
        transcriber: Transcriber,
        model_name: str,
        output_dir: Path,
        block_duration_s: int = 300,
        max_duration_s: int = 14_400,
        language: str = "auto",
        task: str = "transcribe",
        include_timestamps: bool = True,
    ) -> bool:
        """Démarre un job. Retourne False si un autre est déjà en cours."""
        with self._lock:
            if self._state is JobState.RUNNING:
                return False
            self._state = JobState.RUNNING
            self._current_name = source.name
            self._cancel = threading.Event()

        self._thread = threading.Thread(
            target=self._run,
            args=(
                source, transcriber, model_name, output_dir, block_duration_s,
                max_duration_s, language, task, include_timestamps,
            ),
            daemon=True,
            name="file-job",
        )
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Demande l'annulation. Le partiel déjà transcrit sera conservé."""
        self._cancel.set()

    # ---- Exécution ----

    def _run(
        self,
        source: Path,
        transcriber: Transcriber,
        model_name: str,
        output_dir: Path,
        block_duration_s: int,
        max_duration_s: int,
        language: str,
        task: str,
        include_timestamps: bool,
    ) -> None:
        temp_audio: Path | None = None
        try:
            duration = audio_convert.probe_duration(source)
            if duration is None:
                raise audio_convert.AudioConversionError(
                    f"Impossible de lire la durée de {source.name}. Le fichier "
                    f"est peut-être corrompu ou ne contient pas d'audio."
                )
            # Refuser AVANT de convertir : inutile de payer la conversion d'un
            # fichier de 6 h choisi par erreur.
            if duration > max_duration_s:
                raise audio_convert.AudioConversionError(
                    f"{source.name} dure "
                    f"{file_transcriber.format_duration(duration)}, au-delà de "
                    f"la limite de "
                    f"{file_transcriber.format_duration(max_duration_s)}. "
                    f"Tu peux relever cette limite dans les Préférences."
                )

            audio_path, is_temp = audio_convert.ensure_16k_mono(source)
            if is_temp:
                temp_audio = audio_path

            transcript = self._run_transcription(
                lambda: file_transcriber.transcribe_file(
                    transcriber=transcriber,
                    audio_path=audio_path,
                    total_duration_s=duration,
                    block_duration_s=block_duration_s,
                    language=language,
                    task=task,
                    cancel=self._cancel,
                    on_progress=self._cb.on_progress,
                )
            )

            output_path = self._write_transcript(
                transcript=transcript,
                source=source,
                model_name=model_name,
                output_dir=output_dir,
                include_timestamps=include_timestamps,
            )

            state = (
                JobState.CANCELLED if transcript.cancelled else JobState.DONE
            )
            self._finish(
                JobResult(
                    state=state, output_path=output_path, transcript=transcript
                )
            )

        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._finish(JobResult(state=JobState.FAILED, error=str(exc)))
        finally:
            if temp_audio is not None:
                try:
                    temp_audio.unlink(missing_ok=True)
                except OSError:
                    pass

    def _finish(self, result: JobResult) -> None:
        with self._lock:
            self._state = result.state
            self._current_name = None
        try:
            self._cb.on_done(result)
        except Exception:  # noqa: BLE001
            traceback.print_exc()

    # ---- Écriture du .txt ----

    def _write_transcript(
        self,
        transcript: FileTranscript,
        source: Path,
        model_name: str,
        output_dir: Path,
        include_timestamps: bool,
    ) -> Path:
        content = file_transcriber.format_transcript(
            transcript,
            source_name=source.name,
            model_name=model_name,
            include_timestamps=include_timestamps,
        )
        target = resolve_output_path(source, output_dir)

        # Écriture puis renommage : os.replace est atomique sur le même volume,
        # donc on ne peut jamais observer un .txt à moitié écrit.
        part = target.with_suffix(target.suffix + _PART_SUFFIX)
        part.write_text(content, encoding="utf-8")
        os.replace(part, target)
        return target


def resolve_output_path(source: Path, output_dir: Path) -> Path:
    """Choisit le chemin du .txt : `<dossier>/<nom source>.txt`.

    En cas de collision, suffixe `-2`, `-3`… plutôt que d'écraser un
    transcript existant. Si le dossier n'est pas créable ou inscriptible, on
    se replie sur ~/Documents — mieux vaut un fichier ailleurs qu'un job de
    40 minutes perdu à la dernière seconde.
    """
    directory = _writable_dir(output_dir)
    stem = source.stem
    candidate = directory / f"{stem}.txt"
    index = 2
    while candidate.exists():
        candidate = directory / f"{stem}-{index}.txt"
        index += 1
    return candidate


def _writable_dir(output_dir: Path) -> Path:
    for candidate in (output_dir.expanduser(), _FALLBACK_DIR):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            if os.access(candidate, os.W_OK):
                return candidate
        except OSError:
            continue
    # Dernier recours : le dossier personnel existe toujours.
    return Path.home()
