#!/usr/bin/env bash
set -euo pipefail
trap "echo 'error: Script failed: see failed command above'" ERR

declare -a pkgs=(
    curl

    # APT.
    apt-transport-https
    ca-certificates
    gnupg
    software-properties-common
)

apt-get update
apt-get install -y "${pkgs[@]}"
