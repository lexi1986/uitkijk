#!/usr/bin/env bash
# Voegt Google's Coral-pakketbron toe (libedgetpu, edgetpu-compiler). Wordt gesourced door
# install.sh en export-model.sh; draai als root.
coral_repo() {
  [ -f /etc/apt/sources.list.d/coral-edgetpu.list ] && return 0
  apt-get install -y -qq --no-install-recommends curl gnupg ca-certificates >/dev/null
  curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg \
    | gpg --dearmor --yes -o /usr/share/keyrings/coral-edgetpu.gpg
  echo "deb [signed-by=/usr/share/keyrings/coral-edgetpu.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    > /etc/apt/sources.list.d/coral-edgetpu.list
  if ! apt-get update -qq; then
    # Google's bron is offline (403 sinds Google de Coral niet meer ondersteunt): weer weghalen,
    # anders faalt elke volgende apt-get update.
    rm -f /etc/apt/sources.list.d/coral-edgetpu.list /usr/share/keyrings/coral-edgetpu.gpg
    apt-get update -qq || true
    echo "Let op: Google's Coral-pakketbron is niet bereikbaar." >&2
    return 1
  fi
}
