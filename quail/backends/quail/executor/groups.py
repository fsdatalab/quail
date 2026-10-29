"""Layer-group page layouts for answer decoding."""

import numpy as np

from quail.backends.quail.executor.chunk import DecodeRows, decode_groups


class FullAttentionGroup:
    """Full-attention page tables and borrowed-prefix reads by answer slot."""

    def __init__(self, num_slots, page_tokens):
        self.page_tokens = page_tokens
        self.pages = np.zeros((num_slots, 1), dtype=np.int32)
        self.num_pages = np.zeros(num_slots, dtype=np.int64)
        self.parent = np.arange(num_slots, dtype=np.int64)
        self.borrowed = np.zeros(num_slots, dtype=np.int64)
        self.lead = np.zeros(num_slots, dtype=np.int64)

    def start(self, document, pages, parent, shared_tokens):
        """Bind a branch to its pages and borrowed document prefix."""
        self.set_pages(document, pages)
        self.parent[document] = document
        self.borrowed[document] = 0
        self.lead[document] = 0
        if parent is not None and shared_tokens:
            self.parent[document] = parent
            self.borrowed[document] = shared_tokens
            self.lead[parent] = max(self.lead[parent], shared_tokens)

    def set_pages(self, document, pages):
        """Update one slot's table after allocation or growth."""
        pages = np.asarray(pages, dtype=np.int32)
        if len(pages) > self.pages.shape[1]:
            wider = np.zeros((len(self.parent), len(pages)), dtype=np.int32)
            wider[:, :self.pages.shape[1]] = self.pages
            self.pages = wider
        self.pages[document, :len(pages)] = pages
        self.num_pages[document] = len(pages)

    def add_pages(self, documents, pages) -> None:
        """Append one page to each listed slot's table."""
        width = int(self.num_pages[documents].max()) + 1
        if width > self.pages.shape[1]:
            wider = np.zeros((len(self.parent), width), dtype=np.int32)
            wider[:, :self.pages.shape[1]] = self.pages
            self.pages = wider
        self.pages[documents, self.num_pages[documents]] = pages
        self.num_pages[documents] += 1

    def detach(self, document):
        """Forget sharing identities after a branch is evicted."""
        children = np.flatnonzero(self.parent == document)
        self.parent[children] = children
        self.borrowed[children] = 0
        self.parent[document] = document
        self.borrowed[document] = 0
        self.lead.fill(0)
        np.maximum.at(self.lead, self.parent, self.borrowed)
        return np.union1d(children, [document])

    def rows(self, running, positions, share):
        """Build scatter rows and private and shared attention reads."""
        page = positions // self.page_tokens
        if (page >= self.num_pages[running]).any():
            raise AssertionError("an answer's row is past its pages")
        width = int(self.num_pages[running].max())
        table = self.pages[running, :width]
        index = np.arange(len(running))
        write_rows = (table[index, page].astype(np.int64)
                      * self.page_tokens + positions % self.page_tokens)
        rows = DecodeRows(positions=positions, write_rows=write_rows,
                          page_table=table, slots=running)
        if not share:
            return rows
        # a group is a slot and the slots that borrow the most from it,
        # whose first pages are the same; its rows read those pages once
        # together when two or more of them run. A slot both borrowing
        # and lent from takes the group with the longer shared pages
        groups, shared = decode_groups(running, self.parent, self.borrowed, self.lead)
        together = shared > 0
        if not together.any():
            return rows
        skip = np.where(together, shared // self.page_tokens, 0)
        own_width = int((self.num_pages[running] - skip).max())
        columns = skip[:, None] + np.arange(own_width)
        rows.own_table = np.where(
            columns < self.num_pages[running][:, None],
            table[index[:, None], np.minimum(columns, width - 1)], 0
        ).astype(np.int32)
        rows.own_lengths = positions + 1 - np.where(together, shared, 0)
        member = np.flatnonzero(together)
        order = member[np.argsort(groups[member], kind="stable")]
        starts = np.flatnonzero(np.r_[True, np.diff(groups[order]) != 0])
        rows.shared_rows = order
        rows.shared_cu = np.r_[starts, len(order)].astype(np.int32)
        firsts = order[starts]
        most = int((shared[firsts] // self.page_tokens).max())
        rows.shared_table = table[firsts, :most].astype(np.int32)
        rows.shared_used = shared[firsts].astype(np.int32)
        return rows

