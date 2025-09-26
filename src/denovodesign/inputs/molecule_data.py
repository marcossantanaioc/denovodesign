"""Classes to handle molecular data."""

import pydantic
from rdkit import Chem


class Molecule(pydantic.BaseModel, frozen=True, arbitrary_types_allowed=True):
  """Represents a molecule."""

  smiles: str

  @pydantic.computed_field
  @property
  def tokens(self) -> tuple[str, ...]:
    """Store tokens for a molecule."""
    pass

  @pydantic.computed_field
  @property
  def molecule(self) -> Chem.Mol | None:
    """Convert smiles to Chem.Mol."""
    return Chem.MolFromSmiles(self.smiles)

  @pydantic.computed_field
  @property
  def is_valid(self) -> bool:
    """Label a molecule as valid.

    This method checks if self.molecule is a valid chemical entity by checking
    if self.smiles was parsed by RDKit and self.molecule has at least 1 atom.

    Returns:
        True if molecule is valid.
    """
    if isinstance(self.molecule, Chem.Mol) and self.molecule.GetNumAtoms() > 0:
      return True
    return False
