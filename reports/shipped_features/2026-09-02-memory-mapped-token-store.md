# Memory mapped token store

Quail now writes tokenized input batches to a temporary Arrow file on the
compute worker. Planning reads document lengths from the file, and execution
reads token values when a document enters a model chunk. The worker no longer
keeps every source ID, document string, and token array in heap memory.

The same file stores columns needed by the final projection. Quail selects the
surviving positions from those columns instead of scanning the source again.
Multi GPU workers receive the token file path and document positions, so token
arrays are not copied through the parent process pipe.

Physical plans now store one contiguous document range per GPU. The plan size
therefore depends on the GPU count instead of the document count. Filter
admission also keeps pending positions in ranges or packed integer arrays, and
it creates document prefixes and KV keys only when it accesses a document.

The focused CPU suite covers file backed token access, source projection,
compact child process transfer, plan encoding, filter admission, and complete
query assembly.
