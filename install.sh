#!/bin/bash
# Voxtral Dictée — installation one-liner pour macOS Apple Silicon.
#
# Usage :
#   curl -fsSL https://raw.githubusercontent.com/Jeanjipm/transcript_voxtral/main/install.sh | bash
#
# Étapes :
#   1. Vérifie Apple Silicon
#   2. Installe Homebrew (si absent)
#   3. Installe Python 3.11+ (si absent)
#   4. Clone le repo dans ~/.voxtral/app
#   5. Crée un venv et installe les dépendances
#   6. Télécharge le modèle par défaut (~3.2 Go)
#   7. Crée la commande `voxtral`
#   8. Propose le démarrage automatique
#   9. Lance l'app

# Self-reexec quand on est piped (curl | bash). Sans ça, des sous-processus
# comme `brew install` consomment des octets du script depuis le tube, ce
# qui corrompt l'exécution (lignes qui disparaissent, script qui meurt en
# silence).
#
# On bascule sur un fichier temporaire avec stdin = /dev/tty si dispo
# (permet aussi aux prompts 'read -r -p' de fonctionner), sinon /dev/null
# (les prompts prendront leur valeur par défaut — c'est OK, les défauts
# sont pensés pour un install non-interactive).
if [[ ! -t 0 ]]; then
  TMPSCRIPT=$(mktemp "${TMPDIR:-/tmp}/voxtral-install.XXXXXX")
  cat > "$TMPSCRIPT"
  if bash -c ": </dev/tty" >/dev/null 2>&1; then
    exec bash "$TMPSCRIPT" </dev/tty
  else
    exec bash "$TMPSCRIPT" </dev/null
  fi
fi

set -euo pipefail

# ---------- Paramètres ----------
REPO_URL="https://github.com/Jeanjipm/transcript_voxtral.git"
INSTALL_DIR="$HOME/.voxtral/app"
VENV_DIR="$HOME/.voxtral/venv"
LOG_FILE="$HOME/.voxtral/voxtral.log"
LAUNCH_AGENT_PLIST="$HOME/Library/LaunchAgents/com.voxtral.dictee.plist"

# Sélection du dossier pour la commande `voxtral`. Sur Apple Silicon, le
# chemin historique /usr/local/bin n'existe pas par défaut et Homebrew
# utilise /opt/homebrew/bin. On privilégie un dossier déjà présent dans
# le PATH utilisateur pour éviter de demander une élévation de droits.
pick_bin_dir() {
  # Déjà dans le PATH ? On regarde dans l'ordre de préférence.
  for candidate in "$HOME/.local/bin" "/opt/homebrew/bin" "/usr/local/bin"; do
    if [[ -d "$candidate" && -w "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  # Aucun candidat existant : on crée ~/.local/bin (standard XDG).
  mkdir -p "$HOME/.local/bin"
  echo "$HOME/.local/bin"
}
BIN_DIR="$(pick_bin_dir)"

# ---------- Couleurs (TTY) ----------
if [[ -t 1 ]]; then
  C_INFO=$'\033[1;34m'
  C_OK=$'\033[1;32m'
  C_WARN=$'\033[1;33m'
  C_ERR=$'\033[1;31m'
  C_END=$'\033[0m'
else
  C_INFO=""; C_OK=""; C_WARN=""; C_ERR=""; C_END=""
fi

info()  { echo "${C_INFO}[i]${C_END} $*"; }
ok()    { echo "${C_OK}[✓]${C_END} $*"; }
warn()  { echo "${C_WARN}[!]${C_END} $*"; }
fail()  { echo "${C_ERR}[✗]${C_END} $*" >&2; exit 1; }

# ---------- 1. Vérification Apple Silicon ----------
info "Vérification de l'architecture..."
if [[ "$(uname -m)" != "arm64" ]]; then
  fail "Voxtral Dictée nécessite un Mac Apple Silicon (M1/M2/M3/M4). Mac Intel non supporté."
fi
if [[ "$(uname -s)" != "Darwin" ]]; then
  fail "Voxtral Dictée fonctionne uniquement sur macOS."
fi
ok "Apple Silicon détecté."

# ---------- 2. Homebrew ----------
if ! command -v brew >/dev/null 2>&1; then
  info "Installation de Homebrew (gestionnaire de paquets macOS)..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  # Ajoute brew au PATH pour la session courante (Apple Silicon)
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
ok "Homebrew présent."

# ---------- 3. Python 3.11+ ----------
PYTHON_BIN=""
# Préférence : 3.13 d'abord (écosystème ML + pyobjc bien rodés), puis
# 3.12 et 3.11 (anciens stables), enfin 3.14 en dernier recours. 3.14 est
# trop récent début 2026 : pyobjc-framework-AppKit n'a pas encore de wheel
# compatible, ce qui casse l'install. Si seul 3.14 est dispo on l'utilise
# quand même — l'utilisateur verra l'erreur pip et pourra installer 3.13.
for v in 3.13 3.12 3.11 3.14; do
  if command -v "python${v}" >/dev/null 2>&1; then
    PYTHON_BIN="python${v}"
    break
  fi
done
if [[ -z "$PYTHON_BIN" ]]; then
  info "Aucun Python 3.11+ trouvé — installation de Python 3.13 via Homebrew..."
  brew install python@3.13
  PYTHON_BIN="python3.13"
fi
ok "Python disponible : $($PYTHON_BIN --version)"

# tkinter n'est pas livré par défaut avec les python@X.Y de Homebrew —
# nécessaire pour la fenêtre Préférences (settings_ui.py). On installe
# le paquet python-tk correspondant à la version Python choisie.
PY_VERSION="${PYTHON_BIN#python}"
TK_FORMULA="python-tk@${PY_VERSION}"
if brew list "$TK_FORMULA" >/dev/null 2>&1; then
  ok "$TK_FORMULA déjà installé."
else
  info "Installation de $TK_FORMULA (tkinter pour la fenêtre Préférences)..."
  brew install "$TK_FORMULA" || warn "Échec install $TK_FORMULA — Préférences indisponible tant que ce n'est pas corrigé."
fi

# ---------- 4. Clone / pull du repo ----------
mkdir -p "$(dirname "$INSTALL_DIR")"
if [[ -d "$INSTALL_DIR/.git" ]]; then
  info "Mise à jour du dépôt existant..."
  git -C "$INSTALL_DIR" pull --ff-only
else
  info "Clonage du dépôt dans $INSTALL_DIR..."
  git clone "$REPO_URL" "$INSTALL_DIR"
fi
ok "Code source à jour."

# ---------- 5. venv + dépendances ----------
if [[ ! -d "$VENV_DIR" ]]; then
  info "Création de l'environnement virtuel Python..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

info "Installation des dépendances (peut prendre quelques minutes)..."
pip install --upgrade pip setuptools wheel
pip install -r "$INSTALL_DIR/requirements.txt"
ok "Dépendances installées."

# ---------- 6. Téléchargement du modèle par défaut ----------
info "Téléchargement du modèle par défaut (~3.2 Go, dépend de la connexion)..."
"$VENV_DIR/bin/python" "$INSTALL_DIR/download_model.py"
ok "Modèle prêt."

# ---------- 7. Commande `voxtral` ----------
LAUNCHER="$INSTALL_DIR/voxtral-launcher.sh"
# Idem que le launcher du bundle Voxtral.app : on doit charger brew shellenv
# (PATH + HOMEBREW_PREFIX) sinon Python ne trouve pas les dylibs Tcl/Tk
# de python-tk@3.13 quand l'app est lancée hors shell interactif (ex.
# depuis un LaunchAgent, qui a un environnement minimal).
cat > "$LAUNCHER" <<EOF
#!/bin/bash
eval "\$(/opt/homebrew/bin/brew shellenv)"
exec "$VENV_DIR/bin/python" "$INSTALL_DIR/app.py" "\$@"
EOF
chmod +x "$LAUNCHER"

ln -sf "$LAUNCHER" "$BIN_DIR/voxtral"
ok "Commande 'voxtral' installée dans $BIN_DIR."
# Avertir si ce dossier n'est pas dans le PATH (~/.local/bin par défaut
# hors zsh récent, par exemple).
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) warn "Ajoute cette ligne à ton ~/.zshrc pour taper 'voxtral' partout :"
     echo "    export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac

# ---------- 7b. Voxtral.app (launcher clic depuis Finder / Spotlight / Dock) ----------
# Bundle minimal : Info.plist + executable qui exec voxtral-launcher.sh.
# LSUIElement=true → pas d'icône Dock ni de menu dans la barre (l'app est
# déjà représentée par son icône 🎤 dans la menu bar via rumps).
# Placement dans ~/Applications : indexé par Spotlight, pas de sudo requis.
APP_BUNDLE="$HOME/Applications/Voxtral.app"
info "Création de $APP_BUNDLE..."
mkdir -p "$APP_BUNDLE/Contents/MacOS"

cat > "$APP_BUNDLE/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>voxtral</string>
    <key>CFBundleIdentifier</key>
    <string>com.voxtral.dictee</string>
    <key>CFBundleName</key>
    <string>Voxtral</string>
    <key>CFBundleDisplayName</key>
    <string>Voxtral Dictée</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleVersion</key>
    <string>0.2.0</string>
    <key>CFBundleShortVersionString</key>
    <string>0.2.0</string>
    <key>LSUIElement</key>
    <true/>
    <key>LSMinimumSystemVersion</key>
    <string>13.0</string>
    <key>NSMicrophoneUsageDescription</key>
    <string>Voxtral utilise le micro pour la dictée vocale locale (aucune donnée ne quitte votre Mac).</string>
    <key>ISGraphicIconConfiguration</key>
    <dict>
        <key>ISEnclosureColor</key>
        <string>blue</string>
        <key>ISSymbolColor</key>
        <string>white</string>
        <key>ISSymbolName</key>
        <string>mic.fill</string>
    </dict>
</dict>
</plist>
EOF

mkdir -p "$(dirname "$LOG_FILE")"
# CFBundleExecutable = script Python direct (shebang sur le python3 du venv)
# plutôt qu'un trampoline bash qui fait `exec python`. Le trampoline avec
# exec() casse l'enregistrement NSStatusItem quand l'app est lancée depuis
# Spotlight/Finder (bug macOS confirmé par Apple DTS, FB21015611 —
# reproduit aussi sur Plover avec trampoline C). Le shebang garde un seul
# exec en chaîne, LaunchServices → python3 direct, bundle identity préservée.
# Remplit l'équivalent de `brew shellenv` (PATH + HOMEBREW_PREFIX) via
# os.environ pour que Python trouve les dylibs Tcl/Tk (_tkinter).
cat > "$APP_BUNDLE/Contents/MacOS/voxtral" <<EOF
#!$VENV_DIR/bin/python3
# -*- coding: utf-8 -*-
"""CFBundleExecutable de Voxtral.app — Python direct, sans trampoline bash."""
import os
import sys

os.environ.setdefault("HOMEBREW_PREFIX", "/opt/homebrew")
os.environ.setdefault("HOMEBREW_CELLAR", "/opt/homebrew/Cellar")
os.environ["PATH"] = "/opt/homebrew/bin:/opt/homebrew/sbin:" + os.environ.get("PATH", "")

_log = open("$LOG_FILE", "a", buffering=1)
sys.stdout = _log
sys.stderr = _log

sys.path.insert(0, "$INSTALL_DIR")
import app
app.main()
EOF
chmod +x "$APP_BUNDLE/Contents/MacOS/voxtral"

# Rafraîchit le cache LaunchServices pour que macOS relise Info.plist —
# indispensable pour que ISGraphicIconConfiguration génère la nouvelle
# icône micro. Sans ça, Finder/Spotlight continuent d'afficher l'ancienne.
touch "$APP_BUNDLE"
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[[ -x "$LSREGISTER" ]] && "$LSREGISTER" -f "$APP_BUNDLE" >/dev/null 2>&1 || true
killall Dock 2>/dev/null || true

ok "Voxtral.app disponible dans ~/Applications/. Ouvre-le depuis Spotlight (Cmd+Espace → 'Voxtral') ou glisse-le dans le Dock."
info "Logs runtime : $LOG_FILE (tail -f pour les voir en direct)."

# ---------- 8. Démarrage automatique (optionnel) ----------
# Toujours unload + rm l'ancien plist s'il existe → état propre à chaque
# install, et on profite des éventuels correctifs appris depuis.
if [[ -f "$LAUNCH_AGENT_PLIST" ]]; then
  launchctl unload "$LAUNCH_AGENT_PLIST" 2>/dev/null || true
  rm -f "$LAUNCH_AGENT_PLIST"
  info "LaunchAgent précédent retiré (sera re-créé si tu réactives l'autostart)."
fi

read -r -p "Lancer Voxtral automatiquement au démarrage du Mac ? [y/N] " AUTOSTART
if [[ "$AUTOSTART" =~ ^[Yy]$ ]]; then
  mkdir -p "$(dirname "$LAUNCH_AGENT_PLIST")"
  mkdir -p "$(dirname "$LOG_FILE")"
  # On lance le bundle Voxtral.app via /usr/bin/open plutôt que le script
  # directement : ça donne au process une identité de bundle macOS correcte,
  # nécessaire pour que l'icône menu bar (NSStatusItem) s'affiche quand
  # lancée par launchd (sinon rumps crée l'icône mais macOS la cache car
  # le process est traité comme un daemon sans UI).
  cat > "$LAUNCH_AGENT_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.voxtral.dictee</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/open</string>
        <string>$APP_BUNDLE</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <false/>
    <key>StandardOutPath</key>
    <string>$LOG_FILE</string>
    <key>StandardErrorPath</key>
    <string>$LOG_FILE</string>
</dict>
</plist>
EOF
  launchctl load "$LAUNCH_AGENT_PLIST"
  ok "Démarrage automatique activé. Logs : $LOG_FILE"
fi

# ---------- 9. Lancement ----------
ok "Installation terminée."
echo
echo "${C_OK}Important :${C_END} au premier lancement, macOS demandera deux"
echo "permissions (Microphone + Accessibilité). Accepte-les depuis :"
echo "  Réglages Système → Confidentialité et sécurité"
echo
echo "Raccourci par défaut : maintenir ⌥ Option DROITE pour parler."
echo

# Évite le double-lancement : si Voxtral tourne déjà (instance Terminal
# précédente, LaunchAgent du step 8, ou lancé via Voxtral.app), on ne
# relance pas, sinon on aurait deux icônes 🎤 dans la menu bar.
if pgrep -f "voxtral/app/app.py" >/dev/null 2>&1; then
  ok "Voxtral est déjà en cours d'exécution (icône 🎤 visible dans la menu bar)."
else
  read -r -p "Lancer Voxtral maintenant ? [Y/n] " LAUNCH
  if [[ ! "$LAUNCH" =~ ^[Nn]$ ]]; then
    nohup "$LAUNCHER" >/dev/null 2>&1 &
    ok "Voxtral lancé (icône 🎤 dans la barre de menu)."
  fi
fi
