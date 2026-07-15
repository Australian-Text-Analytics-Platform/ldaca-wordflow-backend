# quick script to find wide characters in COOEE text files for the Windoes
# encoding bug

from pathlib import Path

d = Path("./cooee/data")
print(d)
for fn in sorted(d.glob("*.txt")):
    try:
        with open(fn, "r", encoding="cp1252") as fh:
            dummy = fh.readlines()
    except Exception as e:
        print(f"read failed {fn} {e}")
