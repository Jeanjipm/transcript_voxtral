"""Tests de housekeeping.py — journal borné et purge des vieilles dictées.

Tout se passe sur `tmp_path` : les deux fonctions prennent des chemins, elles
ne connaissent ~/.voxtral que par les valeurs par défaut de
`run_startup_housekeeping`.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

import housekeeping
from config import MaintenanceConfig
from housekeeping import purge_recordings, run_startup_housekeeping, trim_log


DAY = housekeeping.SECONDS_PER_DAY


# ---- trim_log ----


def _lines(n: int, width: int = 40) -> bytes:
    return b"".join(f"ligne {i:06d} ".ljust(width - 1).encode() + b"\n" for i in range(n))


def test_trim_log_leaves_small_file_alone(tmp_path: Path):
    log = tmp_path / "voxtral.log"
    log.write_bytes(_lines(10))
    before = log.read_bytes()

    assert trim_log(log, max_bytes=10_000, keep_bytes=100) == 0
    assert log.read_bytes() == before


def test_trim_log_keeps_the_tail_on_a_line_boundary(tmp_path: Path):
    """Au-dessus du seuil, on garde la fin du fichier, coupée sur un début de
    ligne, précédée d'une ligne qui dit ce qui s'est passé."""
    log = tmp_path / "voxtral.log"
    log.write_bytes(_lines(100))  # 4 000 octets
    size = log.stat().st_size

    removed = trim_log(log, max_bytes=1_000, keep_bytes=250)

    text = log.read_text(encoding="utf-8")
    body = text.split("\n", 1)[1]
    assert text.startswith("[housekeeping] journal tronqué le ")
    # 250 octets = 6 lignes + un bout : le bout est jeté, les 6 restent.
    assert body == _lines(100)[-240:].decode()
    assert body.startswith("ligne 000094")
    assert body.endswith("ligne 000099 ".ljust(39) + "\n")
    assert removed == size - log.stat().st_size
    assert removed > 0


def test_trim_log_result_stays_under_the_threshold(tmp_path: Path):
    """Seuil égal à la queue conservée (log_max_mb: 1) : le fichier tronqué
    doit passer SOUS le seuil, sinon il serait retronqué à chaque lancement."""
    log = tmp_path / "voxtral.log"
    log.write_bytes(_lines(30_000))  # 1,2 Mo

    removed = trim_log(log, max_bytes=1_000_000, keep_bytes=1_000_000)

    assert removed > 0
    assert log.stat().st_size <= 1_000_000
    assert trim_log(log, max_bytes=1_000_000, keep_bytes=1_000_000) == 0


def test_trim_log_disabled_with_zero(tmp_path: Path):
    log = tmp_path / "voxtral.log"
    log.write_bytes(_lines(100))

    assert trim_log(log, max_bytes=0, keep_bytes=10) == 0
    assert log.stat().st_size == 4_000


def test_trim_log_missing_file_is_not_an_error(tmp_path: Path):
    assert trim_log(tmp_path / "absent.log", max_bytes=10) == 0


def test_trim_log_repositions_an_open_stderr(tmp_path: Path):
    """Le journal EST le stderr du process. Après la troncature, une écriture
    par le descripteur déjà ouvert doit atterrir à la suite de ce qu'on a
    gardé — pas au-delà, derrière un trou de zéros."""
    log = tmp_path / "voxtral.log"
    log.write_bytes(_lines(100))
    fd = os.open(log, os.O_WRONLY)  # volontairement SANS O_APPEND
    os.lseek(fd, 0, os.SEEK_END)
    saved = os.dup(2)
    os.dup2(fd, 2)
    try:
        trim_log(log, max_bytes=1_000, keep_bytes=250)
        os.write(2, b"apres\n")
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(fd)

    data = log.read_bytes()
    assert b"\x00" not in data
    assert data.endswith(b"ligne 000099 ".ljust(39) + b"\napres\n")


# ---- purge_recordings ----


def _wav(directory: Path, name: str, age_days: float, now: float) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"RIFF")
    stamp = now - age_days * DAY
    os.utime(path, (stamp, stamp))
    return path


def test_purge_removes_only_old_dictation_files(tmp_path: Path):
    now = time.time()
    rec = tmp_path / "recordings"
    old = _wav(rec, "dictee-20260801-101010.wav", 45, now)
    recent = _wav(rec, "dictee-20261101-101010.wav", 3, now)
    foreign_old = _wav(rec, "memo-perso.wav", 400, now)
    note = _wav(rec, "dictee-notes.txt", 400, now)

    removed = purge_recordings(rec, max_age_days=30, now=now)

    assert removed == [old]
    assert not old.exists()
    assert recent.exists()
    assert foreign_old.exists(), "un fichier qui n'est pas à nous reste"
    assert note.exists()


def test_purge_disabled_with_zero(tmp_path: Path):
    now = time.time()
    rec = tmp_path / "recordings"
    old = _wav(rec, "dictee-20260101-000000.wav", 400, now)

    assert purge_recordings(rec, max_age_days=0, now=now) == []
    assert old.exists()


def test_purge_missing_directory_is_empty(tmp_path: Path):
    assert purge_recordings(tmp_path / "nope", max_age_days=30) == []


def test_purge_boundary_is_strictly_older(tmp_path: Path):
    now = time.time()
    rec = tmp_path / "recordings"
    exactly = _wav(rec, "dictee-exact.wav", 30, now)
    just_over = _wav(rec, "dictee-over.wav", 30.01, now)

    removed = purge_recordings(rec, max_age_days=30, now=now)

    assert removed == [just_over]
    assert exactly.exists()


# ---- run_startup_housekeeping ----


def test_startup_housekeeping_applies_both(tmp_path: Path, capsys):
    now = time.time()
    log = tmp_path / "voxtral.log"
    log.write_bytes(_lines(100))
    rec = tmp_path / "recordings"
    _wav(rec, "dictee-vieux.wav", 90, now)

    run_startup_housekeeping(
        MaintenanceConfig(log_max_mb=1, recordings_max_age_days=30),
        log_path=log,
        recordings_dir=rec,
    )

    # 1 Mo > 4 000 octets : rien à tronquer, le journal est intact.
    assert log.stat().st_size == 4_000
    assert list(rec.iterdir()) == []
    err = capsys.readouterr().err
    assert "1 enregistrement(s) de plus de 30 jours" in err
    assert "tronqué" not in err


def test_startup_housekeeping_never_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    """Un entretien qui plante ne doit pas empêcher la dictée de démarrer."""
    def boom(*_a, **_k):
        raise OSError("disque en lecture seule")

    monkeypatch.setattr(housekeeping, "trim_log", boom)
    monkeypatch.setattr(housekeeping, "purge_recordings", boom)

    run_startup_housekeeping(
        MaintenanceConfig(), log_path=tmp_path / "l", recordings_dir=tmp_path
    )

    assert "disque en lecture seule" in capsys.readouterr().err
