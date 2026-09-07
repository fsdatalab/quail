# Remote source preparation

Quail can now send a logical query plan and remote source descriptions to its
compute provider. The default Modal provider opens S3 Parquet or Hugging Face
sources inside the Modal worker. The worker tokenizes the document column,
builds the physical plan, runs the model, projects the selected columns, and
returns the final Arrow table through a Modal Function call.

The client still reads Parquet metadata so it can check SQL column names. The
client does not scan or tokenize remote document rows. For an in memory Arrow
table, the Modal provider sends only the raw columns used by the query. The
Modal worker tokenizes those columns and creates the physical plan.

The tokenization loop creates one Arrow token chunk per input batch. It does not
create one Python list that contains every document's tokens before it builds
the Arrow column. The memory mapped token store shipped on September 2 removed
the remaining complete token column from worker heap memory.

CPU tests cover one compute provider request, remote source handling, local
Arrow tables, complete worker preparation, final projection, and the rule that
the client must not call `scan()` for a remote source.
