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
# Préférence : version installée la plus récente >= 3.11 (minimum projet).
# Ordre du plus récent au plus ancien, avec 3.11 comme borne basse.
for v in 3.14 3.13 3.12 3.11; do
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
cat > "$LAUNCHER" <<EOF
#!/bin/bash
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

# ---------- 8. Démarrage automatique (optionnel) ----------
read -r -p "Lancer Voxtral automatiquement au démarrage du Mac ? [y/N] " AUTOSTART
if [[ "$AUTOSTART" =~ ^[Yy]$ ]]; then
  mkdir -p "$(dirname "$LAUNCH_AGENT_PLIST")"
  mkdir -p "$(dirname "$LOG_FILE")"
  cat > "$LAUNCH_AGENT_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.voxtral.dictee</string>
    <key>ProgramArguments</key>
    <array>
        <string>$LAUNCHER</string>
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
  launchctl unload "$LAUNCH_AGENT_PLIST" 2>/dev/null || true
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
read -r -p "Lancer Voxtral maintenant ? [Y/n] " LAUNCH
if [[ ! "$LAUNCH" =~ ^[Nn]$ ]]; then
  nohup "$LAUNCHER" >/dev/null 2>&1 &
  ok "Voxtral lancé (icône 🎤 dans la barre de menu)."
fi
