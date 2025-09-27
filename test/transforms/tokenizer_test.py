"""Test for tokenizer."""

import pytest

from denovodesign import constants
from denovodesign.transforms import tokenizer


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
  @pytest.mark.parametrize("smiles", _INPUT_SMILES)
  def test_tokenizer(self, smiles):
    """Test molecules is_valid label returns correct value."""
    tokens = tokenizer.tokenize(smiles=smiles)
    full_token = tokens.get_token()
    assert isinstance(tokens, tokenizer.Tokens)
    assert tokens.get_smiles() == smiles
    assert constants.SpecialTokens.BOS in full_token
    assert constants.SpecialTokens.EOS in full_token

  def test_vocabulary(self):
    all_tokens = []
    for smiles in _INPUT_SMILES:
      all_tokens.append(tokenizer.tokenize(smiles=smiles).get_token())

    unique_tokens = set()
    for t in all_tokens:
      unique_tokens.update(set(t))
    vocab = tokenizer.MolVocab(tokens=all_tokens)
    assert unique_tokens.issubset(set(vocab.vocab.keys()))
    assert constants.SpecialTokens.UNK in vocab.vocab
