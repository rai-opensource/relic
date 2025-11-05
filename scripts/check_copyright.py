# Copyright (c) 2024 Robotics and AI Institute LLC dba RAI Institute. All rights reserved.

"""Check that files have a copyright. Does not accept mild variations of the copyright format"""
import re
import sys
from pathlib import Path

COPYRIGHT_PATTERN = re.compile(
    r"^#? ?Copyright \(c\) (?P<start_year>[0-9]{4})(?:-(?P<end_year>[0-9]{4}))?[, ]+(?P<holder>.+?)\.(?: +All rights reserved\.)?"  # noqa: E501
)


def file_has_copyright(path: Path) -> bool:
    with open(path, "r") as f:
        return text_has_copyright(f.read())


def text_has_copyright(text: str) -> bool:
    match = re.search(COPYRIGHT_PATTERN, text)
    return match is not None


def main() -> int:
    files_without_copyright = [
        file for file in sys.argv[1:] if not file_has_copyright(Path(file))
    ]

    if files_without_copyright:
        for file in files_without_copyright:
            print(file)
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
