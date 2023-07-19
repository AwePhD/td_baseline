"""Various useful functions"""

from pathlib import Path

def extract_int_from_str(s: str) -> int:
    return int("".join(c for c in s if c.isdigit()))


def prompt_rm_to_user(h5_file: Path) -> bool:
    print(f"{h5_file.name} already exists.")
    user_input = input("Delete the file (y/N): ")
    return user_input.lower() in ['y', 'yes']

