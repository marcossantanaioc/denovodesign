"""Functions that convert molecules into different formats."""

from collections.abc import Iterator, Sequence
import dataclasses
import re

import torch
import tqdm

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

  def get_tokens(self) -> tuple[str, ...]:
    """Returns raw tokens appended with BOS and EOS tokens."""
    return (
      constants.SpecialTokens.BOS,
      *tuple(self.raw_tokens),
      constants.SpecialTokens.EOS,
    )

  def __repr__(self):
    tokens = self.raw_tokens
    size = len(tokens) // 2
    display_str = f"{''.join(tokens[:size])}...{''.join(tokens[-size:])}"

    return (
      f"<{self.__class__.__name__} "
      f"({display_str}), "
      f"num_tokens={self.num_tokens}|unique={self.num_unique_tokens}>"
    )


class MolTokenizer:
  """Creates a vocabulary for a collection of tokens."""

  def __init__(self, pattern: re.Pattern = constants.REGEX_ATOMS):
    self.pattern = pattern
    self._unique_tokens = set()

  @property
  def unique_tokens(self) -> set[str]:
    """Stores unique tokens in the vocabulary."""
    return self._unique_tokens

  @property
  def vocab(self) -> dict[str, int]:
    """Generates a vocabulary of tokens.

    This function takes a sequence of Tokens and generates unique
    integer ids for each individual token. Returns a dictionary
    mapping each token to a unique integer id.

    Returns:
      A mapping from tokens to integers

    """
    mapping = {
      tok: idx for idx, tok in enumerate(sorted(self._unique_tokens), start=3)
    }

    return {**constants.SPECIAL_TOKENS_MAPPING, **mapping}

  def _tokenize_one(self, smiles: str) -> Tokens:
    """Generates tokens from a SMILES string.

    This function breaks the SMILES string down into tokens (representing
    individual atoms, bonds, rings, or other features) by matching the
    provided regular expression pattern. This process is essential for
    cheminformatics tasks like parsing and feature extraction.

    Args:
        smiles: The SMILES string (e.g., 'CC(=O)C') to be tokenized.

    Returns:
      A tuple of tokens including beginning of sequence (BOS) and
      end of sequence (EOS) tokens. E.g. <BOS> <token1, token2, token3...> <EOS>
    """
    tokens = re.findall(self.pattern, smiles)
    if not tokens:
      raise ValueError(f"No tokens found in the SMILES string: {smiles}")
    return Tokens(raw_tokens=tokens)

  def tokenize(
    self, smiles: Sequence[str], show_progress: bool = False
  ) -> Iterator[Tokens]:
    """Tokenize a list of SMILES.

    Args:
      smiles: SMILES to tokenize.
      show_progress: Whether to display a progress bar.

    Yields:
      A generator of Tokens objects.

    """
    iterable = tqdm.tqdm(smiles, total=len(smiles)) if show_progress else smiles
    for smi in iterable:
      tokens = self._tokenize_one(smi)
      self._unique_tokens.update(tokens.raw_tokens)
      yield tokens

  def featurize(
    self, smiles: Sequence[str], show_progress_bar: bool = False
  ) -> Iterator[torch.Tensor]:
    """Generate features from a sequence of SMILES.

    This method tokenizes the input SMILES and yields
    tensors where tokens are represented as integers
    based on the vocabulary.

    Args:
      smiles: Input SMILES to featurize
      show_progress_bar: Whether to display progress bar.

    Yields:
      Tensor representations for smiles
    """
    tokens = self.tokenize(smiles=smiles, show_progress=show_progress_bar)
    for token in tokens:
      feat = tuple(
        self.vocab.get(
          tok, constants.SPECIAL_TOKENS_MAPPING[constants.SpecialTokens.UNK]
        )
        for tok in token.get_tokens()
      )
      if feat:
        yield torch.tensor(feat).long().view(1, -1)
