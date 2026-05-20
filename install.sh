#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCRIPTS_DIR="$HOME/.claude/scripts"
echo "[cc-vision] Installing..."
pip install -r "$SCRIPT_DIR/requirements.txt"
mkdir -p "$SCRIPTS_DIR"
cp "$SCRIPT_DIR/cc-vision.py" "$SCRIPTS_DIR/cc-vision.py"
chmod +x "$SCRIPTS_DIR/cc-vision.py"
cat > "$SCRIPTS_DIR/cc-vision" << 'WRAPPER_EOF'
#!/bin/bash
exec python3 "$HOME/.claude/scripts/cc-vision.py" "$@"
WRAPPER_EOF
chmod +x "$SCRIPTS_DIR/cc-vision"
echo "[cc-vision] Done! Run: cc-vision <file> [--pages RANGE]"
