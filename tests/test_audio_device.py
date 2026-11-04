"""Tests d'audio_device.py — détection d'un changement de micro par défaut.

La lecture CoreAudio elle-même (ctypes) n'est pas testable hors Mac : on
injecte une fonction de lecture et on vérifie la logique « a-t-on changé ? ».
"""

from __future__ import annotations

from unittest.mock import MagicMock

from audio_device import DefaultInputWatcher, DeviceInfo


BUILTIN = DeviceInfo(device_id=51, name="MacBook Pro Microphone")
HEADSET = DeviceInfo(device_id=87, name="AirPods de JJ")


def _watcher(readings):
    """Watcher dont la lecture rend successivement les valeurs de `readings`."""
    read = MagicMock(side_effect=list(readings))
    on_change = MagicMock(name="on_change")
    return DefaultInputWatcher(on_change=on_change, read=read), on_change


def test_first_poll_memorises_without_firing():
    """Au lancement, le flux micro n'est pas encore ouvert : rien à
    reconstruire, on retient juste le périphérique courant."""
    watcher, on_change = _watcher([BUILTIN])
    assert watcher.poll() is False
    on_change.assert_not_called()
    assert watcher.current == BUILTIN


def test_change_fires_with_old_and_new():
    watcher, on_change = _watcher([BUILTIN, HEADSET])
    watcher.poll()
    assert watcher.poll() is True
    on_change.assert_called_once_with(BUILTIN, HEADSET)
    assert watcher.current == HEADSET


def test_same_device_does_not_fire():
    watcher, on_change = _watcher([BUILTIN, BUILTIN, BUILTIN])
    for _ in range(3):
        assert watcher.poll() is False
    on_change.assert_not_called()


def test_unplug_then_replug_fires_twice():
    """Casque branché puis débranché : deux reconstructions, une par
    changement réel."""
    watcher, on_change = _watcher([BUILTIN, HEADSET, BUILTIN])
    watcher.poll()
    watcher.poll()
    watcher.poll()
    assert on_change.call_count == 2
    assert on_change.call_args_list[1].args == (HEADSET, BUILTIN)


def test_failed_reading_is_ignored():
    """`None` (lecture échouée, aucun micro) n'est pas un changement : on ne
    déclenche rien et on ne perd pas le périphérique mémorisé."""
    watcher, on_change = _watcher([BUILTIN, None, BUILTIN])
    watcher.poll()
    assert watcher.poll() is False
    assert watcher.current == BUILTIN
    assert watcher.poll() is False
    on_change.assert_not_called()


def test_failed_reading_before_first_success_is_ignored():
    watcher, on_change = _watcher([None, BUILTIN, HEADSET])
    assert watcher.poll() is False
    assert watcher.current is None
    assert watcher.poll() is False  # premier succès : mémorisation
    assert watcher.poll() is True
    on_change.assert_called_once_with(BUILTIN, HEADSET)


def test_reading_exception_is_swallowed(capsys):
    """Le watcher tourne dans un timer de la menu bar : une exception de
    lecture ne doit jamais s'échapper, sinon la surveillance meurt."""
    read = MagicMock(side_effect=OSError("CoreAudio indisponible"))
    watcher = DefaultInputWatcher(on_change=MagicMock(), read=read)
    assert watcher.poll() is False
    assert "CoreAudio indisponible" in capsys.readouterr().err


def test_on_change_exception_is_swallowed(capsys):
    on_change = MagicMock(side_effect=RuntimeError("boum"))
    read = MagicMock(side_effect=[BUILTIN, HEADSET, HEADSET])
    watcher = DefaultInputWatcher(on_change=on_change, read=read)
    watcher.poll()
    assert watcher.poll() is True  # signalé malgré l'erreur du callback
    assert "boum" in capsys.readouterr().err
    assert watcher.current == HEADSET  # et l'état a bien avancé
    assert watcher.poll() is False


def test_renamed_device_with_same_id_is_not_a_change():
    """Le nom n'est que cosmétique (journal) : seul l'identifiant compte."""
    watcher, on_change = _watcher(
        [BUILTIN, DeviceInfo(device_id=51, name="Micro renommé")]
    )
    watcher.poll()
    assert watcher.poll() is False
    on_change.assert_not_called()
