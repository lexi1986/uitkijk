#!/usr/bin/env bash
# Adds Google's Coral package source (libedgetpu, edgetpu-compiler). Sourced by install.sh and
# export-model.sh; run as root.
coral_repo() {
  [ -f /etc/apt/sources.list.d/coral-edgetpu.list ] && return 0
  apt-get install -y -qq --no-install-recommends curl gnupg ca-certificates >/dev/null
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor --yes -o /usr/share/keyrings/coral-edgetpu.gpg
  echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    > /etc/apt/sources.list.d/coral-edgetpu.list
  if ! apt-get update -qq; then
    # Google's source is offline (403 since Google stopped supporting the Coral): remove it again,
    # otherwise every following apt-get update fails.
    rm -f /etc/apt/sources.list.d/coral-edgetpu.list /usr/share/keyrings/coral-edgetpu.gpg
    apt-get update -qq || true
    echo "Note: Google's Coral package source is not reachable." >&2
    return 1
  fi
}
