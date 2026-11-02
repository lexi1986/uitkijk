#!/bin/sh
# Mappen die Docker als root aanmaakte overdragen aan de service, en dan als gewone gebruiker starten.
set -e
mkdir -p /data/state /data/models /data/go2rtc /clips
chown camsvc:camsvc /data /data/state /data/models /data/go2rtc /clips
[ -f /data/config.yaml ] && chown camsvc:camsvc /data/config.yaml
# toegang tot de Intel-iGPU (/dev/dri) als die is doorgegeven
if [ -e /dev/dri/renderD128 ]; then
  gid=$(stat -c %g /dev/dri/renderD128)
  getent group "$gid" >/dev/null || groupadd -g "$gid" render-host
  usermod -aG "$(getent group "$gid" | cut -d: -f1)" camsvc
fi
exec setpriv --reuid=camsvc --regid=camsvc --init-groups "$@"
