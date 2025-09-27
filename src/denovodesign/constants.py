"""Tokenizer constants."""

from collections.abc import Mapping
import enum
import re

from typing import Final


# Matches any atom.
REGEX_ATOMS: Final[re.Pattern] = re.compile(
  r"\[.*?\]|[()]|\d|(?:Br|Cl|Na|Li|Ca|Fe|Mg|Ti|Zn|Sn|Se|As)|[a-z]|[A-Z]|(?:=|~|-|#|%|\.|\+)"
)


class SpecialTokens(enum.StrEnum):
  """Represents special tokens.

  BOS: used to indicate start of SMILES.
  EOS: indicates end of SMILES.
  UNK: place holder for tokens not in the vocabulary.

  """

  BOS = "<BOS>"
  EOS = "<EOS>"
  UNK = "<UNK>"
  PAD = "<PAD>"


class SpecialTokensValues(enum.IntEnum):
  """Represents special tokens.

  BOS: used to indicate start of SMILES.
  EOS: indicates end of SMILES.
  UNK: place holder for tokens not in the vocabulary.

  """

  PAD = 0
  BOS = 1
  EOS = 2
  UNK = 3


SPECIAL_TOKENS_MAPPING: Final[Mapping[str, int]] = {
  SpecialTokens.BOS: SpecialTokensValues.BOS,
  SpecialTokens.EOS: SpecialTokensValues.EOS,
  SpecialTokens.UNK: SpecialTokensValues.UNK,
  SpecialTokens.PAD: SpecialTokensValues.PAD,
}
