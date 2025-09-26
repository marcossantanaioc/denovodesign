"""Functions that convert molecules into different formats."""

from collections.abc import Sequence
import dataclasses
import re

from denovodesign import constants


@dataclasses.dataclass(kw_only=True, frozen=True)
class Tokens:
  """Represents tokens for a molecule."""
  raw_tokens: Sequence[str]

  def get_smiles(self) -> str:
    """Returns the original SMILES."""
    return "".join(self.raw_tokens)

  def get_token(self) -> tuple[str, ...]:
    """Returns raw token appended with BOS and EOS tokens."""
    return (constants.BOS, *tuple(self.raw_tokens), constants.EOS)


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
