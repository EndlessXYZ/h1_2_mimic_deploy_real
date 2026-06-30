#!/bin/bash
# Safe cleanup script for simulation processes
# Usage: ./kill_sim.sh

pids=$(pgrep -f 'unitree_mujoco' 2>/dev/null)
if [ -n "$pids" ]; then
    echo "Killing unitree_mujoco (PIDs: $pids)"
    kill $pids 2>/dev/null
    sleep 1
    # Force kill if still alive
    pids=$(pgrep -f 'unitree_mujoco' 2>/dev/null)
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null
    fi
fi

pids=$(pgrep -f 'h1_2_ctrl' 2>/dev/null)
if [ -n "$pids" ]; then
    echo "Killing h1_2_ctrl (PIDs: $pids)"
    kill $pids 2>/dev/null
    sleep 1
    pids=$(pgrep -f 'h1_2_ctrl' 2>/dev/null)
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null
    fi
fi

echo "Done."
