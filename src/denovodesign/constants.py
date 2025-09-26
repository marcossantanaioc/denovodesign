"""Tokenizer constants."""

import re

from typing import Final


# Matches any atom.
REGEX_ATOMS: Final[re.Pattern] = re.compile(
  r"""
    [a-z]       |   # Aromatic atoms (e.g., c, n, o)
    [A-Z][a-z]  |   # 2-letter atoms (e.g., Cl, Br)
    [A-Z]       |   # 1-letter atoms (e.g., C, N, O)
    \[.*?\]     |   # Special environments (e.g., [NH+])
    =           |   # Double bonds
    \#          |   # Triple bonds
    ~           |   # Tilde bond (needs escaping)
    \d          |   # Ring closure/opening digits
    [\(\)]      |   # Branches (parentheses)
    \.              # Disconnected structures/salts
""",
  re.VERBOSE,
)

BOS: Final[str] = "<BOS>"
EOS: Final[str] = "<EOS>"
