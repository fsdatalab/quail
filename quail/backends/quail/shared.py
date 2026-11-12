"""Values two processes share through a file both map.

``share`` pickles a value with its large buffers, such as the buffers
of Arrow tables and numpy arrays, written once to a file; ``opened``
maps the file and unpickles the value over the mapped pages, so the
reader copies no buffer. A file in ``/dev/shm`` is memory only. Rust
shares the tables of a stage the same way (``quail_python::shared``).
"""

import mmap
import pickle

# Arrow asks this alignment of every buffer; a mapping starts on a page.
_ALIGNMENT = 64


def share(value, path: str) -> bytes:
    """Write a value's buffers to a new file and return the message for it.

    Args:
        value: Any picklable value.
        path: Where the buffers go; the file must not exist.

    Returns:
        The bytes ``opened`` reads the value from, in any process that
        can open the file.
    """
    buffers = []
    data = pickle.dumps(value, protocol=5, buffer_callback=buffers.append)
    extents = []
    with open(path, "xb") as file:
        for buffer in buffers:
            raw = buffer.raw()
            file.write(b"\0" * (-file.tell() % _ALIGNMENT))
            extents.append((file.tell(), raw.nbytes))
            file.write(raw)
    return pickle.dumps((data, path, extents))


def opened(message: bytes):
    """Return the value of a ``share`` message, over the file's mapped pages.

    The pages are private to this process: writing a buffer changes
    this process's view only.
    """
    data, path, extents = pickle.loads(message)
    if not any(size for _, size in extents):
        return pickle.loads(data, buffers=[b""] * len(extents))
    with open(path, "rb") as file:
        view = memoryview(mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_COPY))
    return pickle.loads(
        data, buffers=[view[start:start + size] for start, size in extents])
