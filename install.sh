#!/bin/sh
# Livescribe installer — downloads the latest release binary from GitHub.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/nonatofabio/livescribe/main/install.sh | sh
#
# Or with a specific version:
#   curl -fsSL https://raw.githubusercontent.com/nonatofabio/livescribe/main/install.sh | sh -s v1.2.0

set -e

REPO="nonatofabio/livescribe"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

# Detect OS and architecture
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Darwin)
            case "$ARCH" in
                arm64|aarch64) PLATFORM="macos-arm64" ;;
                *) echo "Error: macOS $ARCH is not supported. Pre-built binaries are available for ARM64." >&2; exit 1 ;;
            esac
            ;;
        Linux)
            case "$ARCH" in
                x86_64|amd64) PLATFORM="linux-x64" ;;
                *) echo "Error: Linux $ARCH is not supported. Pre-built binaries are available for x86_64." >&2; exit 1 ;;
            esac
            ;;
        *)
            echo "Error: $OS is not supported." >&2
            exit 1
            ;;
    esac
}

# Get latest release tag or use provided version
get_version() {
    if [ -n "$1" ]; then
        VERSION="$1"
    else
        VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep '"tag_name"' | cut -d'"' -f4)"
        if [ -z "$VERSION" ]; then
            echo "Error: Could not determine latest version." >&2
            exit 1
        fi
    fi
}

main() {
    detect_platform
    get_version "$1"

    BINARY_NAME="livescribe-${PLATFORM}"
    URL="https://github.com/$REPO/releases/download/$VERSION/$BINARY_NAME.tar.gz"

    echo "Installing livescribe $VERSION for $PLATFORM..."
    echo "  From: $URL"
    echo "  To:   $INSTALL_DIR/livescribe"

    # Download and extract
    TMPDIR="$(mktemp -d)"
    trap 'rm -rf "$TMPDIR"' EXIT

    curl -fsSL "$URL" -o "$TMPDIR/$BINARY_NAME.tar.gz"
    tar xzf "$TMPDIR/$BINARY_NAME.tar.gz" -C "$TMPDIR"

    # Install
    if [ -w "$INSTALL_DIR" ]; then
        mv "$TMPDIR/livescribe" "$INSTALL_DIR/livescribe"
    else
        echo "  (requires sudo for $INSTALL_DIR)"
        sudo mv "$TMPDIR/livescribe" "$INSTALL_DIR/livescribe"
    fi

    chmod +x "$INSTALL_DIR/livescribe"

    echo ""
    echo "Done! livescribe $VERSION installed to $INSTALL_DIR/livescribe"
    echo ""
    echo "Prerequisites:"
    echo "  brew install espeak-ng cmake    # macOS"
    echo "  apt install espeak-ng cmake     # Linux"
    echo ""
    echo "Get started:"
    echo "  livescribe listen               # transcribe microphone"
    echo "  livescribe speak doc.pdf        # read document aloud"
    echo "  livescribe speak doc.pdf --rewrite  # AI-narrated reading"
}

main "$@"
