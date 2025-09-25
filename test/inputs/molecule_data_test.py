"""Test for molecule_data."""

import pytest

from denovodesign.inputs import molecule_data


class TestMoleculeData:
  @pytest.mark.parametrize(
    "smiles,label", [("fake", False), ("c1ccccc1", True), ("5", False)]
  )
  def test_molecule_data(self, smiles, label):
    """Test molecules is_valid label returns correct value."""
    mol = molecule_data.Molecule(smiles=smiles)
    assert mol.is_valid == label
