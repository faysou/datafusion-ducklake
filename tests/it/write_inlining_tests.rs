#![cfg(feature = "write-duckdb")]

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{
    DuckLakeTableWriter, DuckLakeWriteOptions, DuckdbMetadataWriter, MetadataWriter,
};
use object_store::local::LocalFileSystem;
use tempfile::TempDir;

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_writer_persists_native_inlined_rows() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("metadata.duckdb");
    let data_path = temp.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();
    let writer = Arc::new(
        DuckdbMetadataWriter::new_with_init(catalog_path.to_string_lossy().into_owned()).unwrap(),
    );
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec![Some("one"), None])),
        ],
    )
    .unwrap();
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(2),
        ..Default::default()
    };
    let result = DuckLakeTableWriter::new(writer, Arc::new(LocalFileSystem::new()))
        .unwrap()
        .with_options(&options)
        .write_table("main", "items", &[batch])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    assert_eq!(result.records_written, 2);

    let connection = duckdb::Connection::open(&catalog_path).unwrap();
    let physical_name: String = connection
        .query_row(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            duckdb::params![result.table_id],
            |row| row.get(0),
        )
        .unwrap();
    let rows = connection
        .prepare(&format!(
            "SELECT row_id, begin_snapshot, end_snapshot, id, name
             FROM {physical_name} ORDER BY row_id"
        ))
        .unwrap()
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, i32>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })
        .unwrap()
        .collect::<std::result::Result<Vec<_>, _>>()
        .unwrap();
    let stats: (i64, i64, i64) = connection
        .query_row(
            "SELECT record_count, next_row_id, file_size_bytes
             FROM ducklake_table_stats WHERE table_id = ?",
            duckdb::params![result.table_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        rows,
        vec![
            (0, result.snapshot_id, None, 1, Some("one".to_string())),
            (1, result.snapshot_id, None, 2, None),
        ]
    );
    assert_eq!(stats, (2, 2, 0));
}
