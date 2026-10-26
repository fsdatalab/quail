"""AI.EXTRACT on the Quail executor: the answer decoded over the document.

The candidates (candidates.py) rate every document position as the
answer's start and give each start's continuation; the decoder
(decoder.py) applies the token rule to each readout row; and the
stages (stages.py) ask a pipeline's questions through the stage
scheduler. The vocabulary index and the stage shape the operator
shares with AI.MAP live one level up (vocabulary.py, decoding.py).
"""

from quail.backends.quail.executor.extract.stages import ExtractionStages

__all__ = ["ExtractionStages"]
