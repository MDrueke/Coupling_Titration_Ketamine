"""
Test script for the Recording class.
"""

import tomllib
from pathlib import Path
from recording import Recording, resolve_session_paths

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

with open("sessions.toml", "rb") as f:
    sessions_cfg = tomllib.load(f)

# use first session by default
paths = resolve_session_paths(sessions_cfg["sessions"]["dirs"][0])
recording_dir = paths["recording_dir"]

print("Loading recording...")
rec = Recording(recording_dir, config)

print("\n" + "=" * 50)
print(rec)
print("=" * 50)
