"""Functions used to tokenize molecules for model training."""

import abc


class BaseTokenizer(abc.ABC):
  """Abstract class for tokenizer."""

  @abc.abstractmethod
  def tokenize(self) -> tuple[str, ...]:
    """Base tokenize method."""
    pass
