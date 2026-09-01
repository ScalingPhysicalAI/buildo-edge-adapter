"""Recover after a pod stop/restart.

A restarted pod gets a fresh container: sshd is down, host keys are gone, and
the public IP / TCP ports may have changed. This script goes in through
RunPod's always-available SSH proxy, restarts sshd, re-adds the laptop's key,
reads the new IP/port mapping, and rewrites pod_address.txt.
"""
import os
import re
import subprocess

HOME = os.path.expanduser("~/edgevla")
# Your pod SSH-proxy identity (RunPod console -> Connect -> SSH command).
# Changes with every new pod; update before running.
PROXY = "<pod-id>@ssh.runpod.io"
PUBKEY = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()

script = f"""
ssh-keygen -A
mkdir -p /root/.ssh
grep -qF "{PUBKEY}" /root/.ssh/authorized_keys 2>/dev/null || echo "{PUBKEY}" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
service ssh start
echo IP=$RUNPOD_PUBLIC_IP SSH=$RUNPOD_TCP_PORT_22 APP=$RUNPOD_TCP_PORT_15001
exit
"""
result = subprocess.run(["ssh", "-tt", "-o", "StrictHostKeyChecking=accept-new",
                         PROXY, "bash -s"],
                        input=script.encode(), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT)
text = result.stdout.decode(errors="replace")
matches = re.findall(r"IP=(\S+) SSH=(\d+) APP=(\d+)", text)
if not matches:
    print(text[-2000:])
    raise SystemExit("Could not read pod address - is the pod running?")
ip, ssh_port, app_port = matches[-1]
with open(f"{HOME}/pod_address.txt", "w") as f:
    f.write(f"{ip} {ssh_port} {app_port}\n")
print(f"Pod recovered. New address: ip={ip} ssh_port={ssh_port} app_port={app_port}")
print("pod_address.txt updated. Now run start_server.py.")
