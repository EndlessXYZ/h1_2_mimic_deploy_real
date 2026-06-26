#!/usr/bin/env python3
"""
Simulate keyboard presses for MuJoCo WalkVelocity simulation using Xlib XTEST.
Injects key events at the X server level to drive the MuJoCo KeyboardJoystick.
"""

import time
import sys
from Xlib.display import Display
from Xlib import X
from Xlib.ext import xtest
from Xlib.XK import string_to_keysym

def press_key(display, keysym, duration=0.15):
    """Press and release a key using XTEST."""
    keycode = display.keysym_to_keycode(keysym)
    if keycode == 0:
        print(f"  WARNING: no keycode for keysym {keysym}")
        return
    xtest.fake_input(display, X.KeyPress, keycode)
    display.sync()
    time.sleep(duration)
    xtest.fake_input(display, X.KeyRelease, keycode)
    display.sync()

def hold_key(display, keysym):
    """Press a key and hold it (no release)."""
    keycode = display.keysym_to_keycode(keysym)
    if keycode == 0:
        print(f"  WARNING: no keycode for keysym {keysym}")
        return
    xtest.fake_input(display, X.KeyPress, keycode)
    display.sync()

def release_key(display, keysym):
    """Release a held key."""
    keycode = display.keysym_to_keycode(keysym)
    if keycode == 0:
        return
    xtest.fake_input(display, X.KeyRelease, keycode)
    display.sync()

def find_and_focus_window(display, title_part):
    """Find a window by title substring and focus it."""
    root = display.screen().root
    def search(window):
        try:
            name = window.get_wm_name()
        except:
            name = None
        if name and title_part.lower() in name.lower():
            return window
        try:
            children = window.query_tree().children
        except:
            children = []
        for child in children:
            result = search(child)
            if result:
                return result
        return None
    win = search(root)
    if win:
        try:
            win.set_input_focus(X.RevertToParent, X.CurrentTime)
            display.sync()
            print(f"Focused window: {win.get_wm_name()}")
            return True
        except:
            return False
    return False

def main():
    display = Display()

    # Key syms
    KEY_SPACE = string_to_keysym('space')
    KEY_9 = string_to_keysym('9')
    KEY_Q = string_to_keysym('q')
    KEY_UP = string_to_keysym('Up')
    KEY_V = string_to_keysym('v')
    KEY_G = string_to_keysym('g')
    KEY_W = string_to_keysym('w')
    KEY_LSHIFT = string_to_keysym('Shift_L')

    # Focus MuJoCo window (title is "MuJoCo : Unitree H1-1")
    focused = False
    for title in ['MuJoCo :', 'MuJoCo']:
        if find_and_focus_window(display, title):
            focused = True
            break
    if not focused:
        print("WARNING: MuJoCo window not found by title, injecting keys to focused window")

    time.sleep(0.5)

    # Step 1: Press Space to start simulation loop
    print("[1/6] Pressing Space to start simulation loop...")
    press_key(display, KEY_SPACE, 0.15)
    time.sleep(1.5)

    # Step 2: Press 9 to enable elastic band
    print("[2/6] Pressing 9 to enable elastic band...")
    press_key(display, KEY_9, 0.15)
    time.sleep(1.0)

    # Step 3: Press Q + Up to trigger Passive -> FixStand
    print("[3/6] Pressing Q + Up to trigger Passive -> FixStand...")
    hold_key(display, KEY_Q)
    time.sleep(0.2)
    press_key(display, KEY_UP, 0.25)
    time.sleep(0.2)
    release_key(display, KEY_Q)
    print("  Waiting 5s for FixStand transition...")
    time.sleep(5)

    # Step 4: Press V + G (RB + Y) to trigger FixStand -> WalkVelocity
    print("[4/6] Pressing V + G (RB + Y) to trigger FixStand -> WalkVelocity...")
    hold_key(display, KEY_V)
    time.sleep(0.2)
    press_key(display, KEY_G, 0.25)
    time.sleep(0.2)
    release_key(display, KEY_V)
    print("  Waiting 3s for WalkVelocity transition...")
    time.sleep(3)

    # Step 5: Press W to drive forward walking
    print("[5/6] Pressing W to drive forward walking (3s)...")
    hold_key(display, KEY_W)
    time.sleep(3)
    release_key(display, KEY_W)
    time.sleep(1)

    # Step 6: Press Q + Left Shift (LT + B) to trigger WalkVelocity -> Passive
    print("[6/6] Pressing Q + Left Shift (LT + B) to trigger WalkVelocity -> Passive...")
    hold_key(display, KEY_Q)
    time.sleep(0.2)
    press_key(display, KEY_LSHIFT, 0.25)
    time.sleep(0.2)
    release_key(display, KEY_Q)
    time.sleep(1)

    print("Keyboard simulation complete!")
    display.close()

if __name__ == "__main__":
    main()
