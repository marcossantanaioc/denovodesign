"""Test for tokenizer."""

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
"c1cc(CCCC(C=O))ccc1")


class TestMoleculeData:

  def test_molecule_data(self):
    """Test molecules is_valid label returns correct value."""
    for smiles in _INPUT_SMILES:
      tokens = tokenizer.tokenize(smiles=smiles)
      full_token = tokens.get_token()
      assert isinstance(tokens, tokenizer.Tokens)
      assert tokens.get_smiles() == smiles
      assert constants.BOS in full_token
      assert constants.EOS in full_token
