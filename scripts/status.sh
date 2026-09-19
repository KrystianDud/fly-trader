#!/usr/bin/env bash
# One place to see what the GPU box is doing, what it has cost, and when it dies.
#   ./scripts/status.sh
set -u
cd "$(dirname "$0")/.."

KEY=${FLY_SSH_KEY:-~/.ssh/fly-gpu.pem}
IP=$(cat /tmp/fly_ip.txt 2>/dev/null)
ID=$(cat /tmp/fly_instance.txt 2>/dev/null)
RATE=1.277   # g6.xlarge on-demand, eu-west-2, USD/hour

if [ -z "${IP:-}" ]; then echo "no instance recorded"; exit 1; fi

echo "════ GPU BOX ─ $IP ─ $(date -u +%H:%MZ) ($(TZ=Europe/London date +%H:%M) BST)"

if [ -n "${ID:-}" ]; then
  LAUNCH=$(AWS_PROFILE=${AWS_PROFILE:-default} AWS_REGION=eu-west-2 aws ec2 describe-instances \
    --instance-ids "$ID" --query 'Reservations[0].Instances[0].LaunchTime' \
    --output text 2>/dev/null)
  if [ -n "$LAUNCH" ]; then
    HOURS=$(python3 -c "
from datetime import datetime, timezone
t = datetime.fromisoformat('$LAUNCH'.replace('Z','+00:00'))
print(f'{(datetime.now(timezone.utc)-t).total_seconds()/3600:.2f}')")
    echo "   up ${HOURS}h · spent \$$(python3 -c "print(f'{$HOURS*$RATE:.2f}')") ≈ £$(python3 -c "print(f'{$HOURS*$RATE*0.79:.2f}')")"
  fi
fi

ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -i "$KEY" ubuntu@"$IP" '
  echo "   gpu: $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"
  echo "   dies: $(cat /run/systemd/shutdown/scheduled 2>/dev/null | grep USEC | cut -d= -f2 | \
        xargs -I{} python3 -c "import datetime,sys; print(datetime.datetime.fromtimestamp(int({})/1e6, datetime.timezone.utc).strftime(\"%a %H:%MZ\"))" 2>/dev/null || echo "no timer set")"
  echo
  running=$(pgrep -fl "extract.p[y]|train_gra[p]h" | head -2)
  if [ -n "$running" ]; then
    echo "   RUNNING: $(echo "$running" | head -1 | cut -c1-110)"
  else
    echo "   RUNNING: nothing"
  fi
  echo
  echo "── last progress ──"
  tail -n 5 ~/fruit_fly/logs/*.log ~/fruit_fly/data/activations/logs/*.log 2>/dev/null | tail -16
  echo
  echo "── results so far ──"
  ls -t ~/fruit_fly/reports/*.json 2>/dev/null | head -3 | while read f; do
    python3 -c "
import json, sys, os
d = json.load(open(sys.argv[1]))
name = os.path.basename(sys.argv[1])
arms = d.get(\"arms\") or []
if arms and isinstance(arms, list) and \"ic\" in arms[0]:
    done = d.get(\"complete\")
    tag = \"\" if done is None else (\" [complete]\" if done else \" [partial]\")
    print(\"   \" + name + tag)
    for a in arms:
        print(\"      %-12s IC %+0.4f  Sharpe %+0.2f\" % (a[\"arm\"], a[\"ic\"], a.get(\"sharpe\", 0)))
" "$f"
  done
'
echo
echo "   terminate now:  AWS_PROFILE=${AWS_PROFILE:-default} AWS_REGION=eu-west-2 aws ec2 terminate-instances --instance-ids $ID"
echo "   extend/shorten: ssh -i $KEY ubuntu@$IP 'sudo shutdown -c; sudo shutdown -h HH:MM'"
