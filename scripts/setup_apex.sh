#!/usr/bin/env bash
# Install the official Apex CLI. Run inside WSL 2 (Ubuntu) or Linux/macOS —
# the apex CLI does not support native Windows.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d apex ]; then
  git clone https://github.com/macrocosm-os/apex.git
else
  git -C apex pull --ff-only
fi

(cd apex && ./install_cli.sh)

export PATH="$HOME/.local/bin:$PATH"
apex --help >/dev/null && echo "apex CLI installed. Add this to ~/.bashrc if 'apex' isn't found later:"
echo '  export PATH="$HOME/.local/bin:$PATH"'
