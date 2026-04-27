"""Tests d'updater.py — check_for_update, apply_update, comportement offline."""

from __future__ import annotations

import json
import subprocess
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import URLError

import pytest

import updater
from updater import (
    ApplyResult,
    UpdateInfo,
    apply_update,
    check_for_update,
    get_local_sha,
)


# ---- Fixtures ----


@pytest.fixture
def fake_compare_response_with_update():
    """Réponse GitHub /compare quand 2 commits sont disponibles."""
    return {
        "ahead_by": 2,
        "behind_by": 0,
        "files": [
            {"filename": "app.py", "status": "modified"},
            {"filename": "README.md", "status": "modified"},
        ],
        "commits": [
            {
                "sha": "aaa111",
                "commit": {"message": "feat: première feature\nDétails…"},
            },
            {
                "sha": "bbb222",
                "commit": {"message": "fix: corrige le bug X"},
            },
        ],
    }


@pytest.fixture
def fake_compare_response_uptodate():
    """Réponse GitHub /compare quand on est à jour."""
    return {
        "ahead_by": 0,
        "behind_by": 0,
        "files": [],
        "commits": [],
    }


def _http_response_mock(payload: dict | None):
    """Crée un context manager qui imite urlopen → resp.read() retourne du JSON."""
    if payload is None:
        return None
    resp = MagicMock()
    resp.__enter__ = MagicMock(return_value=BytesIO(json.dumps(payload).encode()))
    resp.__exit__ = MagicMock(return_value=False)
    return resp


# ---- get_local_sha ----


def test_get_local_sha_returns_sha(monkeypatch: pytest.MonkeyPatch):
    """Cas normal : git rev-parse retourne un SHA."""
    fake_run = MagicMock(return_value=MagicMock(stdout="abc123def\n"))
    monkeypatch.setattr(subprocess, "run", fake_run)

    assert get_local_sha() == "abc123def"


def test_get_local_sha_returns_none_on_git_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """Pas dans un repo git → CalledProcessError → None."""
    def raise_err(*args, **kwargs):
        raise subprocess.CalledProcessError(128, "git", "fatal: not a git repo")

    monkeypatch.setattr(subprocess, "run", raise_err)
    assert get_local_sha() is None


def test_get_local_sha_returns_none_when_git_not_found(
    monkeypatch: pytest.MonkeyPatch,
):
    """git pas dans le PATH → FileNotFoundError → None."""
    def raise_err(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", raise_err)
    assert get_local_sha() is None


# ---- check_for_update ----


def test_check_for_update_returns_info_when_behind(
    monkeypatch: pytest.MonkeyPatch, fake_compare_response_with_update,
):
    monkeypatch.setattr(updater, "get_local_sha", lambda: "local123")
    monkeypatch.setattr(
        updater, "_http_get_json", lambda url: fake_compare_response_with_update
    )

    info = check_for_update()
    assert info is not None
    assert isinstance(info, UpdateInfo)
    assert info.commits_behind == 2
    assert info.local_sha == "local123"
    assert info.remote_sha == "bbb222"  # dernier commit
    # Première ligne du dernier commit, sans le détail.
    assert info.head_message == "fix: corrige le bug X"


def test_check_for_update_returns_none_when_uptodate(
    monkeypatch: pytest.MonkeyPatch, fake_compare_response_uptodate,
):
    monkeypatch.setattr(updater, "get_local_sha", lambda: "local123")
    monkeypatch.setattr(
        updater, "_http_get_json", lambda url: fake_compare_response_uptodate
    )

    assert check_for_update() is None


def test_check_for_update_returns_none_when_offline(
    monkeypatch: pytest.MonkeyPatch,
):
    """Pas de réseau → _http_get_json renvoie None → on retourne None
    silencieusement, l'app continue."""
    monkeypatch.setattr(updater, "get_local_sha", lambda: "local123")
    monkeypatch.setattr(updater, "_http_get_json", lambda url: None)

    assert check_for_update() is None


def test_check_for_update_returns_none_when_no_local_sha(
    monkeypatch: pytest.MonkeyPatch,
):
    """Pas dans un repo git → on n'a pas de SHA à comparer → None."""
    monkeypatch.setattr(updater, "get_local_sha", lambda: None)
    # _http_get_json ne devrait même pas être appelé.
    fake = MagicMock()
    monkeypatch.setattr(updater, "_http_get_json", fake)

    assert check_for_update() is None
    fake.assert_not_called()


def test_check_for_update_detects_risky_files(
    monkeypatch: pytest.MonkeyPatch,
):
    """Si install.sh ou requirements.txt modifiés → flag requires_manual_action."""
    payload = {
        "ahead_by": 1,
        "files": [
            {"filename": "app.py"},
            {"filename": "install.sh"},
            {"filename": "requirements.txt"},
        ],
        "commits": [{"sha": "x", "commit": {"message": "big update"}}],
    }
    monkeypatch.setattr(updater, "get_local_sha", lambda: "local")
    monkeypatch.setattr(updater, "_http_get_json", lambda url: payload)

    info = check_for_update()
    assert info is not None
    assert info.requires_manual_action is True
    assert "install.sh" in info.risky_files
    assert "requirements.txt" in info.risky_files
    # app.py n'est PAS risqué (pas dans RISKY_FILES).
    assert "app.py" not in info.risky_files


def test_check_for_update_no_risky_files(
    monkeypatch: pytest.MonkeyPatch, fake_compare_response_with_update,
):
    monkeypatch.setattr(updater, "get_local_sha", lambda: "local")
    monkeypatch.setattr(
        updater, "_http_get_json", lambda url: fake_compare_response_with_update
    )

    info = check_for_update()
    assert info is not None
    assert info.requires_manual_action is False
    assert info.risky_files == []


def test_check_for_update_truncates_long_message(
    monkeypatch: pytest.MonkeyPatch,
):
    """Messages très longs → tronqués à 120 chars pour l'UI."""
    long_msg = "x" * 200
    payload = {
        "ahead_by": 1,
        "files": [],
        "commits": [{"sha": "x", "commit": {"message": long_msg}}],
    }
    monkeypatch.setattr(updater, "get_local_sha", lambda: "local")
    monkeypatch.setattr(updater, "_http_get_json", lambda url: payload)

    info = check_for_update()
    assert info is not None
    assert len(info.head_message) <= 120


def test_check_for_update_handles_empty_commits_list(
    monkeypatch: pytest.MonkeyPatch,
):
    """Edge case : ahead_by > 0 mais commits=[] (ne devrait pas arriver
    en vrai, mais on ne crash pas)."""
    payload = {"ahead_by": 1, "files": [], "commits": []}
    monkeypatch.setattr(updater, "get_local_sha", lambda: "local")
    monkeypatch.setattr(updater, "_http_get_json", lambda url: payload)

    info = check_for_update()
    assert info is not None
    assert info.commits_behind == 1
    assert info.head_message == ""
    # Si pas de commits, remote_sha tombe sur local_sha (pas idéal mais pas crash).
    assert info.remote_sha == "local"


# ---- _http_get_json (test direct) ----


def test_http_get_json_returns_dict_on_success(
    monkeypatch: pytest.MonkeyPatch,
):
    payload = {"key": "value"}
    monkeypatch.setattr(
        "updater.urlopen", lambda *a, **kw: _http_response_mock(payload)
    )
    result = updater._http_get_json("https://example.com/api")
    assert result == payload


def test_http_get_json_returns_none_on_url_error(
    monkeypatch: pytest.MonkeyPatch,
):
    """URLError (pas de réseau) → silent failure → None."""
    def raise_url(*args, **kwargs):
        raise URLError("DNS failure")

    monkeypatch.setattr("updater.urlopen", raise_url)
    assert updater._http_get_json("https://example.com") is None


def test_http_get_json_returns_none_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
):
    def raise_timeout(*args, **kwargs):
        raise TimeoutError("slow")

    monkeypatch.setattr("updater.urlopen", raise_timeout)
    assert updater._http_get_json("https://example.com") is None


def test_http_get_json_returns_none_on_invalid_json(
    monkeypatch: pytest.MonkeyPatch,
):
    """Réponse non-JSON (ex. HTML d'une page d'erreur GitHub) → None."""
    bad_resp = MagicMock()
    bad_resp.__enter__ = MagicMock(return_value=BytesIO(b"not json"))
    bad_resp.__exit__ = MagicMock(return_value=False)
    monkeypatch.setattr("updater.urlopen", lambda *a, **kw: bad_resp)
    assert updater._http_get_json("https://example.com") is None


# ---- apply_update ----


def test_apply_update_success(monkeypatch: pytest.MonkeyPatch):
    """git fetch + pull --ff-only OK → success + requires_restart."""
    fake_run = MagicMock(return_value=MagicMock(stdout=""))
    monkeypatch.setattr(subprocess, "run", fake_run)

    result = apply_update()
    assert isinstance(result, ApplyResult)
    assert result.success is True
    assert result.requires_restart is True
    # 2 appels : fetch + pull
    assert fake_run.call_count == 2


def test_apply_update_failure_on_pull_error(monkeypatch: pytest.MonkeyPatch):
    """git pull échoue (ex. divergent branches, conflict) → success=False."""
    def fake_run(args, **kwargs):
        if "pull" in args:
            err = subprocess.CalledProcessError(
                1, args, output="", stderr="Not possible to fast-forward\n"
            )
            raise err
        return MagicMock(stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    result = apply_update()
    assert result.success is False
    assert "fast-forward" in result.message


def test_apply_update_failure_when_git_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    """git pas installé → message explicite à l'user."""
    def raise_fnf(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", raise_fnf)

    result = apply_update()
    assert result.success is False
    assert "git introuvable" in result.message.lower()


def test_apply_update_uses_ff_only(monkeypatch: pytest.MonkeyPatch):
    """Régression : on doit toujours passer --ff-only au git pull pour
    éviter un merge accidentel."""
    captured_calls = []

    def fake_run(args, **kwargs):
        captured_calls.append(args)
        return MagicMock(stdout="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    apply_update()

    pull_calls = [c for c in captured_calls if "pull" in c]
    assert len(pull_calls) == 1
    assert "--ff-only" in pull_calls[0]


# ---- UpdateInfo dataclass ----


def test_update_info_requires_manual_action_true_with_risky():
    info = UpdateInfo(
        local_sha="a", remote_sha="b", commits_behind=1,
        head_message="msg", risky_files=["install.sh"],
    )
    assert info.requires_manual_action is True


def test_update_info_requires_manual_action_false_when_empty():
    info = UpdateInfo(
        local_sha="a", remote_sha="b", commits_behind=1,
        head_message="msg",
    )
    assert info.requires_manual_action is False
    assert info.risky_files == []
