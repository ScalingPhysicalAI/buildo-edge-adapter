"""Start (or verify) the EdgeVLA guidance server on the GPU machine.

Usage: python start_server.py
Reads pod address from ~/edgevla/pod_address.txt (ip, ssh_port, app_port).
"""
import os
import subprocess
import time

HOME = os.path.expanduser("~/edgevla")
KEY = os.path.expanduser("~/.ssh/id_ed25519")

with open(f"{HOME}/pod_address.txt") as f:
    IP, SSH_PORT, APP_PORT = f.read().split()
DEST = f"root@{IP}"

def ssh(cmd, **kw):
    return subprocess.run(["ssh", "-o", "ConnectTimeout=15", "-p", SSH_PORT,
                           "-i", KEY, DEST, cmd],
                          capture_output=True, text=True, **kw)

# Already running?
r = ssh("pgrep -f '[s]erver\\.py' >/dev/null && grep -q LISTENING /workspace/asyncvla/server.log && echo RUNNING_OK")
if "RUNNING_OK" in r.stdout:
    print("Server is already running and LISTENING. Nothing to do.")
    raise SystemExit(0)

print("Uploading latest server.py and launching...")
subprocess.run(["scp", "-P", SSH_PORT, "-i", KEY, f"{HOME}/server.py",
                f"{DEST}:/workspace/asyncvla/AsyncVLA/server.py"], check=True)
ssh("pkill -f '[s]erver\\.py'")
subprocess.Popen(
    ["ssh", "-p", SSH_PORT, "-i", KEY, DEST,
     "cd /workspace/asyncvla/AsyncVLA && nohup /workspace/asyncvla/env/bin/python "
     "server.py > /workspace/asyncvla/server.log 2>&1 & echo LAUNCHED"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)
print("Launched. Loading the 8.3B model on the pod (takes ~2-6 min)...")

for i in range(40):
    time.sleep(15)
    r = ssh("tail -1 /workspace/asyncvla/server.log 2>/dev/null")
    line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "(no log yet)"
    print(f"  [{(i+1)*15:4d}s] {line}")
    if "LISTENING" in line:
        print("\nServer is up and LISTENING. You can now run the edge client.")
        break
else:
    print("Timed out waiting for LISTENING; check with watch_server.py")
