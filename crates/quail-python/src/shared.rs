//! Tables two processes share through a file both map.
//!
//! The writer puts a table in a file once, as an Arrow IPC file, and
//! sends the reader the file's path. The reader maps the file and reads
//! batches whose buffers are the mapped pages: no byte of the table
//! goes through a socket, and the reader copies none. A file in
//! `/dev/shm` is memory only.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::buffer::Buffer;
use arrow::ipc::convert::fb_to_schema;
use arrow::ipc::reader::{FileDecoder, read_footer_length};
use arrow::ipc::writer::{FileWriter, IpcWriteOptions};
use arrow::ipc::{MetadataVersion, root_as_footer};
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{DataFusionError, Result};
use memmap2::Mmap;

/// A table's schema and batches.
pub type Table = (SchemaRef, Vec<RecordBatch>);

/// The bytes after an IPC file's footer: the footer's length and the
/// magic string.
const TRAILER: usize = 10;

/// The alignment Arrow asks of every buffer, which a mapped file keeps
/// because a mapping starts on a page.
const ALIGNMENT: usize = 64;

fn failure(error: impl std::fmt::Display) -> DataFusionError {
    DataFusionError::Execution(error.to_string())
}

/// Write a table to a new file at `path`.
///
/// # Errors
///
/// The file exists or does not write.
pub fn write_table(path: &Path, (schema, batches): &Table) -> Result<()> {
    let file = File::create_new(path)?;
    let options = IpcWriteOptions::try_new(ALIGNMENT, false, MetadataVersion::V5)?;
    let mut writer = FileWriter::try_new_with_options(BufWriter::new(file), schema, options)?;
    for batch in batches {
        writer.write(batch)?;
    }
    writer.finish()?;
    Ok(())
}

/// Read the table at `path` over the file's mapped pages, and remove
/// the file's name; the pages live as long as a batch reads them.
///
/// # Errors
///
/// The file does not open or map, or is not an Arrow IPC file.
pub fn read_table(path: &Path) -> Result<Table> {
    let file = File::open(path)?;
    // the writer finished the file before it sent its path, and no one
    // writes it again
    let mapped = unsafe { Mmap::map(&file) }?;
    std::fs::remove_file(path)?;
    let length = mapped.len();
    let start = NonNull::new(mapped.as_ptr().cast_mut())
        .ok_or_else(|| failure("a shared table mapped at no address"))?;
    // the buffer owns the mapping, so the pages outlive every batch
    let buffer = unsafe { Buffer::from_custom_allocation(start, length, Arc::new(mapped)) };
    let malformed = || failure("a shared table is not an Arrow IPC file");
    let trailer = length.checked_sub(TRAILER).ok_or_else(malformed)?;
    let footer_length = read_footer_length(buffer[trailer..].try_into().map_err(|_| malformed())?)?;
    let footer_start = trailer.checked_sub(footer_length).ok_or_else(malformed)?;
    let footer = root_as_footer(&buffer[footer_start..trailer]).map_err(failure)?;
    let schema = Arc::new(fb_to_schema(footer.schema().ok_or_else(malformed)?));
    // a misaligned buffer is an error, not a silent copy
    let mut decoder =
        FileDecoder::new(Arc::clone(&schema), footer.version()).with_require_alignment(true);
    let block = |offset: i64, metadata: i32, body: i64| -> Result<Buffer> {
        let offset = usize::try_from(offset).map_err(|_| malformed())?;
        let size = usize::try_from(metadata)
            .ok()
            .zip(usize::try_from(body).ok())
            .and_then(|(metadata, body)| metadata.checked_add(body))
            .ok_or_else(malformed)?;
        if offset.checked_add(size).is_none_or(|end| end > length) {
            return Err(malformed());
        }
        Ok(buffer.slice_with_length(offset, size))
    };
    for found in footer.dictionaries().iter().flatten() {
        let data = block(found.offset(), found.metaDataLength(), found.bodyLength())?;
        decoder.read_dictionary(found, &data)?;
    }
    let mut batches = Vec::new();
    for found in footer.recordBatches().iter().flatten() {
        let data = block(found.offset(), found.metaDataLength(), found.bodyLength())?;
        batches.extend(decoder.read_record_batch(found, &data)?);
    }
    Ok((schema, batches))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::{Array, Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn directory(name: &str) -> std::path::PathBuf {
        let path = std::env::temp_dir().join(format!("quail-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn a_read_table_holds_the_written_rows_over_the_mapped_file() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("text", DataType::Utf8, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(0..1000)),
                Arc::new(StringArray::from_iter(
                    (0..1000).map(|id| Some(id.to_string())),
                )),
            ],
        )
        .unwrap();
        let path = directory("shared").join("table");
        write_table(&path, &(Arc::clone(&schema), vec![batch.clone()])).unwrap();
        let size = std::fs::metadata(&path).unwrap().len() as usize;
        let (found, batches) = read_table(&path).unwrap();
        assert_eq!(found, schema);
        assert_eq!(batches, vec![batch]);
        assert!(!path.exists());
        // each buffer lies inside one mapping the size of the file
        let ids = batches[0].column(0).to_data();
        let text = batches[0].column(1).to_data();
        let starts: Vec<usize> = [&ids.buffers()[0], &text.buffers()[0], &text.buffers()[1]]
            .iter()
            .map(|buffer| buffer.as_ptr() as usize)
            .collect();
        let span = starts.iter().max().unwrap() - starts.iter().min().unwrap();
        assert!(span < size);
        assert!(starts.iter().all(|start| start % ALIGNMENT == 0));
    }

    #[test]
    fn a_table_of_no_rows_keeps_its_schema() {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
        let path = directory("empty").join("table");
        write_table(&path, &(Arc::clone(&schema), vec![])).unwrap();
        let (found, batches) = read_table(&path).unwrap();
        assert_eq!(found, schema);
        assert!(batches.is_empty());
    }

    #[test]
    fn a_file_that_is_not_arrow_is_an_error() {
        let path = directory("bad").join("table");
        std::fs::write(&path, b"short").unwrap();
        assert!(read_table(&path).is_err());
        std::fs::write(&path, vec![7u8; 4096]).unwrap();
        assert!(read_table(&path).is_err());
    }
}
