#![cfg(feature = "write-duckdb")]

use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion_ducklake::{
    ColumnDef, DuckLakeTableWriter, DuckLakeWriteOptions, DuckdbMetadataWriter, MetadataWriter,
    WriteMode,
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

#[tokio::test(flavor = "multi_thread")]
async fn duckdb_multi_table_write_commits_parquet_and_inline_rows() {
    let temp = TempDir::new().unwrap();
    let catalog_path = temp.path().join("multi.duckdb");
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
    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("name", &DataType::Utf8, true).unwrap(),
    ];
    for table_name in ["data", "coverage"] {
        let setup = writer
            .begin_write_transaction("main", table_name, &columns, WriteMode::Append)
            .unwrap();
        writer
            .publish_snapshot(
                setup.table_id,
                "main",
                table_name,
                setup.snapshot_id,
                WriteMode::Append,
                setup.base_snapshot_id,
                &columns,
                &setup.column_ids,
            )
            .unwrap();
    }
    let table_writer =
        DuckLakeTableWriter::new(writer.clone(), Arc::new(LocalFileSystem::new())).unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_options(
            "main",
            "data",
            schema.as_ref(),
            WriteMode::Append,
            &[RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                    Arc::new(StringArray::from(vec!["one", "two", "three"])),
                ],
            )
            .unwrap()],
            &DuckLakeWriteOptions {
                data_inlining_row_limit: Some(0),
                ..Default::default()
            },
        )
        .await
        .unwrap();
    transaction
        .stage_write_with_options(
            "main",
            "coverage",
            schema.as_ref(),
            WriteMode::Append,
            &[RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int32Array::from(vec![1])), Arc::new(StringArray::from(vec!["one"]))],
            )
            .unwrap()],
            &DuckLakeWriteOptions {
                data_inlining_row_limit: Some(1),
                ..Default::default()
            },
        )
        .await
        .unwrap();

    let committed = transaction.commit().await.unwrap();

    assert_eq!(committed.len(), 2);
    assert_eq!(committed[0].snapshot_id, committed[1].snapshot_id);
    assert_eq!(committed[0].files_written, 1);
    assert_eq!(committed[1].files_written, 0);
    let connection = duckdb::Connection::open(&catalog_path).unwrap();
    let file_snapshot: i64 = connection
        .query_row(
            "SELECT begin_snapshot FROM ducklake_data_file WHERE table_id = ?",
            duckdb::params![committed[0].table_id],
            |row| row.get(0),
        )
        .unwrap();
    let inline_table: String = connection
        .query_row(
            "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
            duckdb::params![committed[1].table_id],
            |row| row.get(0),
        )
        .unwrap();
    let inline_snapshot: i64 = connection
        .query_row(
            &format!("SELECT begin_snapshot FROM \"{inline_table}\""),
            [],
            |row| row.get(0),
        )
        .unwrap();
    let changes: String = connection
        .query_row(
            "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
            duckdb::params![committed[0].snapshot_id],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(file_snapshot, committed[0].snapshot_id);
    assert_eq!(inline_snapshot, committed[0].snapshot_id);
    assert_eq!(
        changes,
        format!(
            "inserted_into_table:{},inserted_into_table:{}",
            committed[0].table_id, committed[1].table_id
        )
    );
}
