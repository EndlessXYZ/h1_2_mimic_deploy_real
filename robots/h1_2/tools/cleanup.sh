#!/bin/bash
pkill -9 -f 'h1_2_ctrl' 2>/dev/null || true
pkill -9 -f 'unitree_mujoco' 2>/dev/null || true
sleep 1
if pgrep -f 'h1_2_ctrl|unitree_mujoco' > /dev/null 2>&1; then
    echo "Still running"
else
    echo "All cleaned"
fi
