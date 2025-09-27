"""Test for tokenizer."""

import pytest

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

  # def test_vocabulary(self):
  #   all_tokens = []
  #   for smiles in _INPUT_SMILES:
  #     all_tokens.append(moltokenizer.tokenize(smiles=smiles))

  #   unique_tokens = set()
  #   for t in all_tokens:
  #     unique_tokens.update(set(t.raw_tokens))
  #   vocab = moltokenizer.MolVocab(tokens=all_tokens)
  #   assert unique_tokens.issubset(set(vocab.vocab.keys()))
  #   assert constants.SpecialTokens.UNK in vocab.vocab
