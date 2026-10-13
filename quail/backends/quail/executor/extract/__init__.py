"""AI.EXTRACT on the Quail executor: the answer decoded over the document.

The vocabulary index (vocabulary.py) groups the model's tokens by
word; the candidates (candidates.py) rate every document position as
the answer's start and give each start's continuation; the decoder
(decode.py) applies the token rule to each readout row; and the
stages (stages.py) ask a pipeline's questions through the stage
scheduler.
"""

from quail.backends.quail.executor.extract.stages import ExtractionStages
from quail.backends.quail.executor.extract.vocabulary import load_vocabulary

__all__ = ["ExtractionStages", "load_vocabulary"]
