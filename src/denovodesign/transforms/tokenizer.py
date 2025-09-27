"""Functions that convert molecules into different formats."""

import collections
from collections.abc import Sequence
import dataclasses
import re

from denovodesign import constants


@dataclasses.dataclass(kw_only=True, frozen=True)
class Tokens:
  """Represents tokens for a molecule."""

  raw_tokens: Sequence[str]

  @property
  def num_tokens(self) -> int:
    """Returns number of tokens."""
    return len(self.raw_tokens)

  @property
  def num_unique_tokens(self) -> int:
    """Returns number of unique tokens."""
    return len(set(self.raw_tokens))

  def get_smiles(self) -> str:
    """Returns the original SMILES."""
    return "".join(self.raw_tokens)

  def get_token(self) -> tuple[str, ...]:
    """Returns raw token appended with BOS and EOS tokens."""
    return (
      constants.SpecialTokens.BOS,
      *tuple(self.raw_tokens),
      constants.SpecialTokens.EOS,
    )

  def __repr__(self):
    initial_tokens = self.raw_tokens[:5]
    last_tokens = self.raw_tokens[-5:]
    return (
      f"<{self.__class__.__name__} "
      f"({initial_tokens}...{last_tokens}), "
      f"num_tokens={self.num_tokens}|unique={self.num_unique_tokens}>"
    )


def tokenize(
  smiles: str, pattern: re.Pattern = constants.REGEX_ATOMS
) -> tuple[str, ...]:
  """Generates tokens from a SMILES string.

  This function breaks the SMILES string down into tokens (representing
  individual atoms, bonds, rings, or other features) by matching the
  provided regular expression pattern. This process is essential for
  cheminformatics tasks like parsing and feature extraction.

  Args:
      smiles: The SMILES string (e.g., 'CC(=O)C') to be tokenized.
      pattern: The pre-compiled regular expression
          used for tokenization. Defaults to constants.REGEX_ATOMS.

  Returns:
    A tuple of tokens including beginning of sequence (BOS) and
    end of sequence (EOS) tokens. E.g. <BOS> <token1, token2, token3...> <EOS>
  """
  tokens = re.findall(pattern, smiles)
  if not tokens:
    raise ValueError(f"No tokens found in the SMILES string: {smiles}")
  return Tokens(raw_tokens=tokens)


class MolVocab:
  """Creates a vocabulary for a collection of tokens."""

  def __init__(self, tokens: Sequence[Tokens]):
    self.tokens = tokens

  @property
  def vocab(self) -> dict[str, int]:
    """Generates a vocabulary of tokens.

    This function takes a sequence of Tokens and generates unique
    integer ids for each individual token. Returns a dictionary
    mapping each token to a unique integer id.

    Returns:
      A mapping from tokens to integers

    """
    unique_tokens = set()

    for token in self.tokens:
      unique_tokens.update(set(token))

    mapping = collections.OrderedDict(
      {tok: idx for idx, tok in enumerate(sorted(unique_tokens), start=3)}
    )

    return {**constants.SPECIAL_TOKENS_MAPPING, **mapping}
