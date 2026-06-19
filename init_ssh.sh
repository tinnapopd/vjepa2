#!/bin/bash
#
# init_ssh.sh
#
# Description:
#   Sets up and starts an OpenSSH server on port 2222 inside a container or
#   remote environment (e.g. a Jupyter pod). This allows you to SSH in from
#   your local machine (e.g. Mac) without needing root terminal access.
#
# Usage:
#   chmod +x init_ssh.sh
#   bash init_ssh.sh
#
#   When prompted, paste the contents of your local public key:
#     cat ~/.ssh/id_ed25519.pub   (or id_rsa.pub)
#
# Connect from Mac after running:
#   ssh -p 2222 root@<server-ip-or-hostname>
#
# Notes:
#   - Requires root privileges (apt-get, sshd).
#   - sshd runs in the foreground on port 2222 after setup.
#   - Safe to re-run; authorized_keys are appended, not overwritten.

set -e

apt-get update && apt-get install -y openssh-server
ssh-keygen -A && mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "Paste your Mac public key (e.g. contents of ~/.ssh/id_ed25519.pub), then press Enter:"
read -r MAC_PUBLIC_KEY
echo "$MAC_PUBLIC_KEY" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys && mkdir -p /run/sshd && /usr/sbin/sshd -p 2222
