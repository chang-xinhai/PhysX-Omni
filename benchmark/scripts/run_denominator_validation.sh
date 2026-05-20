#!/usr/bin/env bash
set -euo pipefail
RUN_VALIDATE="${RUN_VALIDATE:-1}"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

validate_denominators
