"""Test for tokenizer."""

import pytest
import torch

from denovodesign import constants
from denovodesign.transforms import moltokenizer


_INPUT_SMILES = (
  "C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl",
  "[CH-]1C=CC=C1",
  "N[C@@H](C)C(=O)O",
  "[I-].[Na+].C=CCBr",
  "N[C@]([H])(C)C(=O)O",
  "[O-][n+]1ccccc1",
  "n1ccccc1",
  "c1cc(CCCC(C=O))ccc1",
  "CCC12CCCN3CCC4(c5c(O)ccc(C6CC7(CC)CCCN8CCC9(C(=O)c%10ccccc%10N69)C87)c5NC4CC1)C32",
)


class TestTokenizer:
  @pytest.fixture
  def smiles(self) -> str:
    return _INPUT_SMILES[0]

  @pytest.fixture
  def tokens(self) -> tuple[str, ...]:
    return [
      "C",
      "1",
      "C",
      "O",
      "C",
      "C",
      "(",
      "=",
      "O",
      ")",
      "N",
      "1",
      "C",
      "2",
      "=",
      "C",
      "C",
      "=",
      "C",
      "(",
      "C",
      "=",
      "C",
      "2",
      ")",
      "N",
      "3",
      "C",
      "[C@@H]",
      "(",
      "O",
      "C",
      "3",
      "=",
      "O",
      ")",
      "C",
      "N",
      "C",
      "(",
      "=",
      "O",
      ")",
      "C",
      "4",
      "=",
      "C",
      "C",
      "=",
      "C",
      "(",
      "S",
      "4",
      ")",
      "Cl",
    ]

  @pytest.mark.parametrize("smiles", _INPUT_SMILES)
  def test_tokenize_one(self, smiles):
    """Test tokenization and ensure special tokens are included."""
    tokenizer = moltokenizer.MolTokenizer()
    tokens = tokenizer._tokenize_one(smiles=smiles)
    full_tokens = tokens.get_tokens()
    assert isinstance(tokens, moltokenizer.Tokens)
    assert tokens.get_smiles() == smiles
    assert constants.SpecialTokens.BOS in full_tokens
    assert constants.SpecialTokens.EOS in full_tokens

  def test_tokenize(self):
    """Test token generation for a sequence of SMILES."""
    tokenizer = moltokenizer.MolTokenizer()
    token_gen = tokenizer.tokenize(smiles=_INPUT_SMILES)

    # Ensure the tokenizer generates tokens for the given SMILES strings
    assert token_gen
    for tok in token_gen:
      assert isinstance(tok, moltokenizer.Tokens)

  def test_featurize(self, smiles):
    expected_features = torch.tensor(
      [
        1,
        11,
        6,
        11,
        14,
        11,
        11,
        4,
        10,
        14,
        5,
        13,
        6,
        11,
        7,
        10,
        11,
        11,
        10,
        11,
        4,
        11,
        10,
        11,
        7,
        5,
        13,
        8,
        11,
        16,
        4,
        14,
        11,
        8,
        10,
        14,
        5,
        11,
        13,
        11,
        4,
        10,
        14,
        5,
        11,
        9,
        10,
        11,
        11,
        10,
        11,
        4,
        15,
        9,
        5,
        12,
        2,
      ]
    )
    tokenizer = moltokenizer.MolTokenizer()
    feats = next(tokenizer.featurize(smiles=[smiles]))
    assert len(feats) == len(expected_features)
    assert torch.equal(feats, expected_features)
