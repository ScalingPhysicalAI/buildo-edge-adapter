"""Live-stream the workstation server log (Ctrl+C to stop)."""
import os
import subprocess

HOME = os.path.expanduser("~/edgevla")
with open(f"{HOME}/pod_address.txt") as f:
    IP, SSH_PORT, APP_PORT = f.read().split()

# sed masks any IP addresses (viewer's and pod's) so the stream is safe to show
# on camera; -u keeps it unbuffered so lines appear live.
subprocess.run(["ssh", "-p", SSH_PORT, "-i", os.path.expanduser("~/.ssh/id_ed25519"),
                f"root@{IP}",
                "tail -n 0 -f /workspace/asyncvla/server.log"
                " | grep --line-buffered -v 'client dropped'"
                " | sed -u -E 's/[0-9]{1,3}(\\.[0-9]{1,3}){3}/[ip-hidden]/g'"])
