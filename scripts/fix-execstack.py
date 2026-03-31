#!/usr/bin/env python3
"""Clear executable stack flags on ELF shared libraries in .venv.

NixOS blocks libraries with RWE GNU_STACK. ctranslate2 (used by whisperx)
ships with this flag set. This script clears the execute bit.
"""

import glob
import struct
import sys


def fix_execstack(path: str) -> bool:
    with open(path, "r+b") as f:
        if f.read(4) != b"\x7fELF":
            return False
        if f.read(1)[0] != 2:  # 64-bit only
            return False
        f.seek(32)
        phoff = struct.unpack("<Q", f.read(8))[0]
        f.seek(54)
        phsz = struct.unpack("<H", f.read(2))[0]
        phnum = struct.unpack("<H", f.read(2))[0]
        for i in range(phnum):
            off = phoff + i * phsz
            f.seek(off)
            if struct.unpack("<I", f.read(4))[0] == 0x6474E551:  # PT_GNU_STACK
                flags = struct.unpack("<I", f.read(4))[0]
                if flags & 1:  # PF_X set
                    f.seek(off + 4)
                    f.write(struct.pack("<I", flags & ~1))
                    return True
                break
    return False


if __name__ == "__main__":
    fixed = 0
    for so in glob.glob(".venv/**/ctranslate2.libs/*.so*", recursive=True):
        if fix_execstack(so):
            print(f"  Fixed execstack: {so}")
            fixed += 1
    sys.exit(0)
