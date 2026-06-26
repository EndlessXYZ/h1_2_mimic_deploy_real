#!/bin/bash
# Safe cleanup script for WalkVelocity test processes
set -euo pipefail

echo "=== Cleaning up WalkVelocity test processes ==="

for proc in "h1_2_ctrl" "unitree_mujoco"; do
    pids=$(pgrep -f "$proc" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Found $proc processes: $pids"
        kill $pids 2>/dev/null || true
        sleep 0.5
        # Force kill if still alive
        pids=$(pgrep -f "$proc" 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "Force killing $proc: $pids"
            kill -9 $pids 2>/dev/null || true
        fi
        echo "Cleaned $proc"
    else
        echo "No $proc processes running"
    fi
done

echo "=== Cleanup complete ==="
