#!/bin/sh
# Hand folders that Docker created as root over to the service, then start as a normal user.
set -e
mkdir -p /data/state /data/models /data/go2rtc /clips
chown camsvc:camsvc /data /data/state /data/models /data/go2rtc /clips
[ -f /data/config.yaml ] && chown camsvc:camsvc /data/config.yaml
# access to the Intel iGPU (/dev/dri) when it is passed through
if [ -e /dev/dri/renderD128 ]; then
  gid=$(stat -c %g /dev/dri/renderD128)
  getent group "$gid" >/dev/null || groupadd -g "$gid" render-host
  usermod -aG "$(getent group "$gid" | cut -d: -f1)" camsvc
fi
exec setpriv --reuid=camsvc --regid=camsvc --init-groups "$@"
