#!/usr/bin/env bash
# Recover the NVIDIA module after an unattended kernel update boots before the
# matching restricted module package has been installed.
set -euo pipefail

if modinfo nvidia >/dev/null 2>&1; then
  modprobe nvidia
  modprobe nvidia_uvm
  exit 0
fi

kernel="$(uname -r)"
driver_pkg="$({ dpkg-query -W -f='${binary:Package}\n' 'nvidia-driver-*-open' 2>/dev/null || true; } \
  | sed -nE 's/^nvidia-driver-([0-9]+)-open(:[^ ]+)?$/\1/p' \
  | sort -rn | head -n1)"

if [[ -z "$driver_pkg" ]]; then
  echo "No installed nvidia-driver-<branch>-open package identifies the required module branch" >&2
  exit 1
fi

module_pkg="linux-modules-nvidia-${driver_pkg}-open-${kernel}"
echo "NVIDIA module is absent for ${kernel}; installing ${module_pkg}" >&2
export DEBIAN_FRONTEND=noninteractive
apt-get install -y --no-install-recommends "$module_pkg"
depmod -a "$kernel"
modprobe nvidia
modprobe nvidia_uvm
nvidia-smi -L >/dev/null
