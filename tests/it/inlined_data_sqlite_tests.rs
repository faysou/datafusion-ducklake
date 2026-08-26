//! Integration tests for reading DuckLake **data inlining** on the SQLite
//! backend.
//!
//! DuckDB's ducklake extension stores small INSERTs directly in the catalog
//! database (in `ducklake_inlined_data_<tid>_<sv>` tables registered in
//! `ducklake_inlined_data_tables`) instead of Parquet. A reader that only scans
//! `ducklake_data_file` silently undercounts. These tests hand-craft inlined
//! tables exactly as DuckDB would and assert that `SELECT` / `COUNT(*)` include
//! the inlined rows, that inlined-row deletes (`end_snapshot`) are respected, and
//! that time travel is correct — while catalogs with no inlined data are
//! unaffected.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::process::Command;
use std::sync::Arc;

use arrow::array::{
    Array, BinaryViewArray, Float32Array, Float64Array, Int32Array, Int64Array,
    TimestampMicrosecondArray, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::AssertSqlSafe;
use sqlx::sqlite::SqlitePool;
use tempfile::TempDir;

use datafusion_ducklake::inlined_filter::{InlinedComparison, InlinedFilter, InlinedValue};
use datafusion_ducklake::{
    ColumnDef, DeleteFileEntry, DuckLakeCatalog, DuckLakeError, DuckLakeTableWriter,
    DuckLakeWriteOptions, InlinedRowRef, MetadataProvider, MetadataWriter, SnapshotCommitMetadata,
    SqliteMetadataProvider, SqliteMetadataWriter, TableWriteOptions, WriteMode,
};

fn table_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Int32, false),
    ]))
}

fn object_store() -> Arc<dyn object_store::ObjectStore> {
    Arc::new(LocalFileSystem::new())
}

fn rw_url(t: &TempDir) -> String {
    format!("sqlite:{}?mode=rwc", t.path().join("test.db").display())
}
fn ro_url(t: &TempDir) -> String {
    format!("sqlite:{}", t.path().join("test.db").display())
}

fn batch(ids: Vec<i32>, vals: Vec<i32>) -> RecordBatch {
    RecordBatch::try_new(
        table_schema(),
        vec![Arc::new(Int32Array::from(ids)), Arc::new(Int32Array::from(vals))],
    )
    .unwrap()
}

async fn make_writer(t: &TempDir) -> SqliteMetadataWriter {
    let data = t.path().join("data");
    std::fs::create_dir_all(&data).unwrap();
    let w = SqliteMetadataWriter::new_with_init(&rw_url(t))
        .await
        .unwrap();
    w.set_data_path(data.to_str().unwrap()).unwrap();
    w
}

fn create_empty_table(writer: &SqliteMetadataWriter, table_name: &str) {
    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
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

/// `(id, val)` from `main.t`, ascending, as of `snapshot` (or latest).
async fn read_rows(t: &TempDir, snapshot: Option<i64>) -> Vec<(i32, i32)> {
    let provider = SqliteMetadataProvider::new(&ro_url(t)).await.unwrap();
    let catalog = match snapshot {
        Some(s) => DuckLakeCatalog::with_snapshot(Arc::new(provider), s).unwrap(),
        None => DuckLakeCatalog::new(provider).unwrap(),
    };
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let mut rows = Vec::new();
    for b in &batches {
        let ids = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let vals = b.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            rows.push((ids.value(i), vals.value(i)));
        }
    }
    rows
}

/// Create the inlining registry + a physical inlined-insert table for `t`, laid
/// out exactly as DuckDB's extension would: `ducklake_inlined_data_<tid>_1(
/// row_id, begin_snapshot, end_snapshot, id, val)`.
async fn seed_inlined(
    pool: &SqlitePool,
    table_id: i64,
    rows: &[(i64, i64, Option<i64>, i32, i32)],
) {
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS ducklake_inlined_data_tables
             (table_id BIGINT, table_name VARCHAR, schema_version BIGINT)",
    )
    .execute(pool)
    .await
    .unwrap();
    let phys = format!("ducklake_inlined_data_{table_id}_1");
    sqlx::query(AssertSqlSafe(format!(
        "CREATE TABLE IF NOT EXISTS {phys}
             (row_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, id INTEGER, val INTEGER)"
    )))
    .execute(pool)
    .await
    .unwrap();
    sqlx::query(
        "INSERT INTO ducklake_inlined_data_tables (table_id, table_name, schema_version)
         VALUES (?, ?, 1)",
    )
    .bind(table_id)
    .bind(&phys)
    .execute(pool)
    .await
    .unwrap();
    for (row_id, begin, end, id, val) in rows {
        sqlx::query(AssertSqlSafe(format!(
            "INSERT INTO {phys} (row_id, begin_snapshot, end_snapshot, id, val) VALUES (?,?,?,?,?)"
        )))
        .bind(row_id)
        .bind(begin)
        .bind(*end)
        .bind(id)
        .bind(val)
        .execute(pool)
        .await
        .unwrap();
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_inlined_scan_pushes_supported_filters_and_falls_back_per_table() {
    let temp = TempDir::new().unwrap();
    let writer = make_writer(&temp).await;
    create_empty_table(&writer, "t");
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let table_id: i64 =
        sqlx::query_scalar("SELECT table_id FROM ducklake_table WHERE table_name = 't'")
            .fetch_one(&pool)
            .await
            .unwrap();
    let snapshot_id: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    seed_inlined(
        &pool,
        table_id,
        &[(0, snapshot_id, None, 1, 10), (1, snapshot_id, None, 2, 20)],
    )
    .await;
    let provider = SqliteMetadataProvider::new(&rw_url(&temp)).await.unwrap();
    let columns = provider.get_table_structure(table_id, snapshot_id).unwrap();
    let pushed = provider
        .scan_inlined_data(
            table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "id".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::I64(2),
            }),
        )
        .unwrap();
    assert_eq!(pushed.materialized_row_count, 1);

    let range = provider
        .scan_inlined_data(
            table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "id".to_string(),
                op: InlinedComparison::GtEq,
                value: InlinedValue::I64(2),
            }),
        )
        .unwrap();
    assert_eq!(range.materialized_row_count, 1);

    let fallback = provider
        .scan_inlined_data(
            table_id,
            snapshot_id,
            &columns,
            Some(&InlinedFilter::Comparison {
                column: "missing".to_string(),
                op: InlinedComparison::Eq,
                value: InlinedValue::I64(2),
            }),
        )
        .unwrap();
    assert_eq!(fallback.materialized_row_count, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_rows_are_unioned_into_reads_with_visibility_and_time_travel() {
    let t = TempDir::new().unwrap();
    // Parquet-backed rows: file1 at snapshot 1, file2 at snapshot 2.
    let w = Arc::new(make_writer(&t).await);
    DuckLakeTableWriter::new(w, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let w2 = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(w2, object_store())
        .unwrap()
        .append_table("main", "t", &[batch(vec![7, 8], vec![70, 80])])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let table_id: i64 = sqlx::query_scalar("SELECT table_id FROM ducklake_table LIMIT 1")
        .fetch_one(&pool)
        .await
        .unwrap();

    // Baseline (no inlined data yet): only the Parquet rows.
    assert_eq!(
        read_rows(&t, None).await,
        vec![(1, 10), (2, 20), (7, 70), (8, 80)]
    );

    // Inlined rows (as DuckDB would store them):
    //  - (3,30): live from snapshot 1 (end_snapshot NULL)
    //  - (5,50): inserted at snapshot 1, DELETED at snapshot 2 (end_snapshot = 2)
    seed_inlined(
        &pool,
        table_id,
        &[(100, 1, None, 3, 30), (101, 1, Some(2), 5, 50)],
    )
    .await;

    // At the latest snapshot (2): Parquet rows + the live inlined (3,30); the
    // inlined (5,50) is excluded because it was deleted at snapshot 2.
    assert_eq!(
        read_rows(&t, None).await,
        vec![(1, 10), (2, 20), (3, 30), (7, 70), (8, 80)],
        "inlined live row included; deleted inlined row excluded"
    );

    // Time travel to snapshot 1: only file1's Parquet rows, plus BOTH inlined
    // rows (neither deleted yet at snapshot 1; file2 not yet visible).
    assert_eq!(
        read_rows(&t, Some(1)).await,
        vec![(1, 10), (2, 20), (3, 30), (5, 50)],
        "time travel sees the inlined rows as of that snapshot"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_without_inlining_is_unaffected() {
    let t = TempDir::new().unwrap();
    let w = Arc::new(make_writer(&t).await);
    DuckLakeTableWriter::new(w, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2, 3], vec![10, 20, 30])])
        .await
        .unwrap();
    // No ducklake_inlined_data_tables exists -> get_inlined_data returns empty,
    // reads are exactly the Parquet rows.
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20), (3, 30)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn writer_inlines_at_limit_and_uses_parquet_outside_limit() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(2),
        ..Default::default()
    };
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    assert_eq!(result.records_written, 2);
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20)]);

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let physical_name: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(result.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let inline_rows: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT COUNT(*) FROM {physical_name} WHERE end_snapshot IS NULL"
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    let data_files: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file WHERE table_id = ?")
            .bind(result.table_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    let stats: (i64, i64, i64) = sqlx::query_as(
        "SELECT record_count, next_row_id, file_size_bytes
         FROM ducklake_table_stats WHERE table_id = ?",
    )
    .bind(result.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
    )
    .bind(result.snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(inline_rows, 2);
    assert_eq!(data_files, 0);
    assert_eq!(stats, (2, 2, 0));
    // The inline commit records the same composed ledger the Parquet path
    // records: the DDL entries for the snapshot plus the write change.
    assert_eq!(
        changes,
        format!(
            "created_schema:\"main\",created_table:\"main\".\"t\",inserted_into_table:{}",
            result.table_id
        )
    );

    let writer = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    let over_limit = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_rows(
            "main",
            "parquet_t",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1, 2, 3], vec![10, 20, 30])],
        )
        .await
        .unwrap();
    assert_eq!(over_limit.files_written, 1);
    assert_eq!(over_limit.records_written, 3);

    let disabled = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(0),
        ..Default::default()
    };
    let writer = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    let disabled_result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&disabled)
        .write_rows(
            "main",
            "disabled_t",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
        )
        .await
        .unwrap();
    assert_eq!(disabled_result.files_written, 1);
    assert_eq!(disabled_result.records_written, 1);
}

/// A fence-rejected multi-table commit is a DEFINITE rollback: nothing is
/// visible on either table and the staged Parquet objects are removed (only an
/// ambiguous commit failure leaves them to the guarded vacuum).
#[tokio::test(flavor = "multi_thread")]
async fn conflicted_multi_table_commit_leaves_no_partial_state_and_no_staged_files() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    create_empty_table(&writer, "data");
    create_empty_table(&writer, "coverage");
    let provider = SqliteMetadataProvider::new(&rw_url(&temp)).await.unwrap();
    let base = provider.get_current_snapshot().unwrap();

    let writer: Arc<dyn MetadataWriter> = writer;
    let table_writer = DuckLakeTableWriter::new(Arc::clone(&writer), object_store()).unwrap();
    let options = TableWriteOptions::new().with_expected_base_snapshot_id(base);
    let mut transaction = table_writer.transaction().with_options(&options);
    transaction
        .stage_write(
            "main",
            "data",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
        )
        .await
        .unwrap();

    // A concurrent commit moves the staged table's generation past the
    // fenced base (the fence is per staged table).
    let concurrent = DuckLakeTableWriter::new(Arc::clone(&writer), object_store()).unwrap();
    concurrent
        .append_table("main", "data", &[batch(vec![9], vec![90])])
        .await
        .unwrap();

    let error = transaction.commit().await.unwrap_err();
    assert!(
        matches!(error, DuckLakeError::Conflict(_)),
        "expected Conflict, got {error:?}"
    );
    // Only the concurrent append's file survives; the rejected stage's
    // Parquet object is removed (definite rollback -> cleanup).
    let staged_parquet = walkdir_parquet(&temp.path().join("data").join("main").join("data"));
    assert_eq!(
        staged_parquet, 1,
        "the fence-rejected stage's Parquet object must be removed"
    );
}

fn walkdir_parquet(dir: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                walkdir_parquet(&path)
            } else {
                usize::from(path.extension().is_some_and(|ext| ext == "parquet"))
            }
        })
        .sum()
}

/// One multi-table commit creating two tables in a fresh schema records the
/// schema's creation exactly once in the snapshot ledger.
#[tokio::test(flavor = "multi_thread")]
async fn multi_table_commit_records_created_schema_once() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let table_writer = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    let mut transaction = table_writer.transaction();
    for table_name in ["t1", "t2"] {
        transaction
            .stage_write(
                "s2",
                table_name,
                table_schema().as_ref(),
                WriteMode::Append,
                &[batch(vec![1], vec![10])],
            )
            .await
            .unwrap();
    }

    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 2);
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
    )
    .bind(results[0].snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(
        changes.matches("created_schema:\"s2\"").count(),
        1,
        "one commit records one schema creation: {changes}"
    );
    assert_eq!(
        changes,
        format!(
            "created_schema:\"s2\",created_table:\"s2\".\"t1\",created_table:\"s2\".\"t2\",\
             inserted_into_table:{},inserted_into_table:{}",
            results[0].table_id, results[1].table_id
        )
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_table_write_commits_parquet_and_inline_rows_in_one_snapshot() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    create_empty_table(&writer, "data");
    create_empty_table(&writer, "coverage");
    let data_options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(0),
        ..Default::default()
    };
    let coverage_options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(2),
        ..Default::default()
    };
    let table_writer = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_options(
            "main",
            "data",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
            &data_options,
        )
        .await
        .unwrap();
    transaction
        .stage_write_with_options(
            "main",
            "coverage",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
            &coverage_options,
        )
        .await
        .unwrap();

    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].snapshot_id, results[1].snapshot_id);
    assert_eq!(results[0].files_written, 1);
    assert_eq!(results[1].files_written, 0);
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let snapshots: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let files: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file")
        .fetch_one(&pool)
        .await
        .unwrap();
    let inlined_tables: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_data_tables")
            .fetch_one(&pool)
            .await
            .unwrap();
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes WHERE snapshot_id = ?",
    )
    .bind(results[0].snapshot_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(snapshots, 3);
    assert_eq!(files, 1);
    assert_eq!(inlined_tables, 1);
    assert_eq!(
        changes,
        format!(
            "inserted_into_table:{},inserted_into_table:{}",
            results[0].table_id, results[1].table_id
        )
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_table_fence_rejection_removes_staged_parquet_file() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    create_empty_table(&writer, "coverage");
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(2),
        ..Default::default()
    };
    let table_writer = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options);
    let initial = table_writer
        .append_table("main", "data", &[batch(vec![1, 2, 3], vec![10, 20, 30])])
        .await
        .unwrap();
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let (data_file_id, data_file_path): (i64, String) = sqlx::query_as(
        "SELECT data_file_id, path FROM ducklake_data_file
         WHERE table_id = ? AND end_snapshot IS NULL",
    )
    .bind(initial.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let delete = table_writer
        .write_delete_file("main", "data", &data_file_path, &[0])
        .await
        .unwrap();
    let deletes = [DeleteFileEntry {
        data_file_id,
        expected_prev_delete_file: None,
        delete,
    }];
    let transaction_options =
        TableWriteOptions::new().with_expected_base_snapshot_id(initial.snapshot_id);
    let mut transaction = table_writer
        .transaction()
        .with_options(&transaction_options);
    transaction
        .stage_write_with_deletes(
            "main",
            "data",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![4, 5, 6], vec![40, 50, 60])],
            &deletes,
            &[],
        )
        .await
        .unwrap();
    transaction
        .stage_write(
            "main",
            "coverage",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![1], vec![10])],
        )
        .await
        .unwrap();
    table_writer
        .append_table("main", "data", &[batch(vec![7, 8, 9], vec![70, 80, 90])])
        .await
        .unwrap();

    let error = transaction.commit().await.unwrap_err();

    assert!(error.to_string().contains("conflict"));
    let data_dir = temp.path().join("data/main/data");
    let parquet_files = std::fs::read_dir(data_dir)
        .unwrap()
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "parquet"))
        .count();
    assert_eq!(parquet_files, 2);
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let coverage_rows: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_inlined_data_tables tables
         JOIN ducklake_table table_meta ON table_meta.table_id = tables.table_id
         WHERE table_meta.table_name = 'coverage'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(coverage_rows, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn multi_table_delete_only_stage_ends_inline_row() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    create_empty_table(&writer, "data");
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(2),
        ..Default::default()
    };
    let table_writer = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options);
    let coverage = table_writer
        .append_table("main", "coverage", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let pool = SqlitePool::connect(&rw_url(&temp)).await.unwrap();
    let physical_name: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(coverage.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let row_id: i64 = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT row_id FROM {physical_name} WHERE id = 1 AND end_snapshot IS NULL"
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write(
            "main",
            "data",
            table_schema().as_ref(),
            WriteMode::Append,
            &[batch(vec![3, 4, 5], vec![30, 40, 50])],
        )
        .await
        .unwrap();
    transaction
        .stage_deletes(
            "main",
            "coverage",
            table_schema().as_ref(),
            &[],
            &[InlinedRowRef {
                table_name: physical_name.clone(),
                row_id,
            }],
        )
        .unwrap();

    let results = transaction.commit().await.unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].snapshot_id, results[1].snapshot_id);
    let live_ids: Vec<i64> = sqlx::query_scalar(AssertSqlSafe(format!(
        "SELECT id FROM {physical_name} WHERE end_snapshot IS NULL ORDER BY id"
    )))
    .fetch_all(&pool)
    .await
    .unwrap();
    assert_eq!(live_ids, vec![2]);
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_inlined_uint64_round_trips_text_storage() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(1),
        ..Default::default()
    };
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::UInt64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![Arc::new(UInt64Array::from(vec![u64::MAX]))],
    )
    .unwrap();
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .append_table("main", "uint64_values", &[batch])
        .await
        .unwrap();
    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));

    let batches = context
        .sql("SELECT value FROM ducklake.main.uint64_values")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    let values = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(values.value(0), u64::MAX);
}

#[tokio::test(flavor = "multi_thread")]
async fn inline_snapshot_column_uses_the_committed_snapshot() {
    let temp = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&temp).await);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("version", DataType::Int64, true),
    ]));
    let columns = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("version", &DataType::Int64, true).unwrap(),
    ];
    let setup = writer
        .begin_write_transaction("main", "versioned", &columns, WriteMode::Append)
        .unwrap();
    writer
        .publish_snapshot(
            setup.table_id,
            "main",
            "versioned",
            setup.snapshot_id,
            WriteMode::Append,
            setup.base_snapshot_id,
            &columns,
            &setup.column_ids,
        )
        .unwrap();
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(2),
        ..Default::default()
    };
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![None, Some(77)])),
        ],
    )
    .unwrap();
    let table_writer = DuckLakeTableWriter::new(writer, object_store()).unwrap();
    let mut transaction = table_writer.transaction();
    transaction
        .stage_write_with_snapshot_columns(
            "main",
            "versioned",
            schema.as_ref(),
            WriteMode::Append,
            &[batch],
            &options,
            &["version"],
        )
        .await
        .unwrap();

    let committed = transaction.commit().await.unwrap();

    let provider = SqliteMetadataProvider::new(&ro_url(&temp)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let context = SessionContext::new();
    context.register_catalog("ducklake", Arc::new(catalog));
    let batches = context
        .sql("SELECT version FROM ducklake.main.versioned ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    let versions = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(versions.values(), &[committed[0].snapshot_id, 77]);
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_ends_inlined_rows_and_updates_stats() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(10),
        ..Default::default()
    };
    let written = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    let writer = SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&rw_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let deleted = ctx
        .sql("DELETE FROM ducklake.main.t WHERE id = 2")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = deleted[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(count.value(0), 1);
    assert_eq!(read_rows(&t, None).await, vec![(1, 10)]);

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let physical_name: String = sqlx::query_scalar(
        "SELECT table_name FROM ducklake_inlined_data_tables WHERE table_id = ?",
    )
    .bind(written.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let ended: (Option<i64>, Option<i64>) = sqlx::query_as(AssertSqlSafe(format!(
        "SELECT MIN(end_snapshot), MAX(end_snapshot) FROM {physical_name} WHERE id = 2"
    )))
    .fetch_one(&pool)
    .await
    .unwrap();
    let stats: (i64, i64) = sqlx::query_as(
        "SELECT record_count, next_row_id FROM ducklake_table_stats WHERE table_id = ?",
    )
    .bind(written.table_id)
    .fetch_one(&pool)
    .await
    .unwrap();
    let changes: String = sqlx::query_scalar(
        "SELECT changes_made FROM ducklake_snapshot_changes ORDER BY snapshot_id DESC LIMIT 1",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    let snapshot: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(ended, (Some(snapshot), Some(snapshot)));
    assert_eq!(stats, (1, 2));
    assert_eq!(changes, format!("deleted_from_table:{}", written.table_id));
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_all_ends_every_inlined_row() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(10),
        ..Default::default()
    };
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let writer = SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&rw_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let deleted = ctx
        .sql("DELETE FROM ducklake.main.t")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = deleted[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(count.value(0), 2);
    assert!(read_rows(&t, None).await.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn delete_commits_parquet_and_inlined_rows_atomically() {
    let t = TempDir::new().unwrap();
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(2),
        ..Default::default()
    };
    let writer = Arc::new(make_writer(&t).await);
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();
    let writer = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .append_table("main", "t", &[batch(vec![3, 4, 5], vec![30, 40, 50])])
        .await
        .unwrap();

    let writer = SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&rw_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let deleted = ctx
        .sql("DELETE FROM ducklake.main.t WHERE id IN (2, 3)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let count = deleted[0]
        .column(0)
        .as_any()
        .downcast_ref::<UInt64Array>()
        .unwrap();
    assert_eq!(count.value(0), 2);
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (4, 40), (5, 50)]);

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let delete_snapshots: (i64, i64) =
        sqlx::query_as("SELECT MIN(begin_snapshot), MAX(begin_snapshot) FROM ducklake_delete_file")
            .fetch_one(&pool)
            .await
            .unwrap();
    let inlined_end: i64 =
        sqlx::query_scalar("SELECT end_snapshot FROM ducklake_inlined_data_1_1 WHERE id = 2")
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(delete_snapshots.0, delete_snapshots.1);
    assert_eq!(inlined_end, delete_snapshots.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn inlined_float_and_binary_view_columns_round_trip() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(10),
        ..Default::default()
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("double_val", DataType::Float64, true),
        Field::new("float_val", DataType::Float32, true),
        Field::new("payload", DataType::BinaryView, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(Float64Array::from(vec![Some(1.5), None])),
            Arc::new(Float32Array::from(vec![Some(-2.25_f32), None])),
            Arc::new(
                vec![Some(&[0x00_u8, 0xff][..]), None]
                    .into_iter()
                    .collect::<BinaryViewArray>(),
            ),
        ],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "typed", &[batch])
        .await
        .unwrap();
    assert_eq!(result.files_written, 0);
    assert_eq!(result.records_written, 2);

    let provider = SqliteMetadataProvider::new(&ro_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let batches = ctx
        .sql("SELECT id, double_val, float_val, payload FROM ducklake.main.typed ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    assert_eq!(batches.len(), 1);
    let batch = &batches[0];
    assert_eq!(batch.num_rows(), 2);
    let doubles = batch
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    let floats = batch
        .column(2)
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    let payloads = batch
        .column(3)
        .as_any()
        .downcast_ref::<BinaryViewArray>()
        .unwrap();
    assert_eq!(doubles.value(0), 1.5);
    assert!(doubles.is_null(1));
    assert_eq!(floats.value(0), -2.25_f32);
    assert!(floats.is_null(1));
    assert_eq!(payloads.value(0), &[0x00_u8, 0xff]);
    assert!(payloads.is_null(1));
}

#[tokio::test(flavor = "multi_thread")]
async fn timestamp_columns_fall_back_to_parquet() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(10),
        ..Default::default()
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("ts", DataType::Timestamp(TimeUnit::Microsecond, None), true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(TimestampMicrosecondArray::from(vec![Some(1_000_002)])),
        ],
    )
    .unwrap();
    let result = DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "events", &[batch])
        .await
        .unwrap();
    assert_eq!(
        result.files_written, 1,
        "a small write with a timestamp column must keep the Parquet path"
    );
    assert_eq!(result.records_written, 1);

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let inline_tables: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_data_tables")
            .fetch_one(&pool)
            .await
            .unwrap();
    let data_files: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_data_file")
        .fetch_one(&pool)
        .await
        .unwrap();
    assert_eq!(inline_tables, 0);
    assert_eq!(data_files, 1);
}

#[tokio::test(flavor = "multi_thread")]
async fn stale_expected_base_snapshot_conflicts_and_commits_nothing() {
    let t = TempDir::new().unwrap();
    let writer = make_writer(&t).await;
    let seed = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(seed, object_store())
        .unwrap()
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    let cols = vec![
        ColumnDef::from_arrow("id", &DataType::Int32, false).unwrap(),
        ColumnDef::from_arrow("val", &DataType::Int32, false).unwrap(),
    ];
    // Pin a base, then let a concurrent append publish a newer generation.
    let stale = writer
        .begin_write_transaction("main", "t", &cols, WriteMode::Append)
        .unwrap();
    let concurrent = Arc::new(SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap());
    DuckLakeTableWriter::new(concurrent, object_store())
        .unwrap()
        .append_table("main", "t", &[batch(vec![3], vec![30])])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    let head_before: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();

    let error = writer
        .register_inlined_data(
            stale.table_id,
            "main",
            "t",
            stale.snapshot_id,
            &[batch(vec![9], vec![90])],
            WriteMode::Append,
            stale.base_snapshot_id,
            &cols,
            &stale.column_ids,
            &SnapshotCommitMetadata::new(),
            Some(stale.base_snapshot_id),
        )
        .unwrap_err();
    assert!(matches!(error, DuckLakeError::Conflict(_)), "{error}");

    let head_after: i64 = sqlx::query_scalar("SELECT MAX(snapshot_id) FROM ducklake_snapshot")
        .fetch_one(&pool)
        .await
        .unwrap();
    let inline_tables: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM ducklake_inlined_data_tables")
            .fetch_one(&pool)
            .await
            .unwrap();
    let record_count: i64 =
        sqlx::query_scalar("SELECT record_count FROM ducklake_table_stats WHERE table_id = ?")
            .bind(stale.table_id)
            .fetch_one(&pool)
            .await
            .unwrap();
    assert_eq!(head_after, head_before, "conflict must commit no snapshot");
    assert_eq!(inline_tables, 0);
    assert_eq!(record_count, 3);
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20), (3, 30)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn update_refuses_tables_with_inlined_rows() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(10),
        ..Default::default()
    };
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    // Default write options: inlining stays enabled on the writable catalog.
    let writer = SqliteMetadataWriter::new(&rw_url(&t)).await.unwrap();
    let provider = SqliteMetadataProvider::new(&rw_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::with_writer(Arc::new(provider), Arc::new(writer)).unwrap();
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let error = match ctx
        .sql("UPDATE ducklake.main.t SET val = 99 WHERE id = 1")
        .await
    {
        Ok(df) => df.collect().await.expect_err("UPDATE must refuse"),
        Err(e) => e,
    };
    let message = error.to_string();
    assert!(
        message.contains("UPDATE on a table with inlined rows is not supported")
            && message.contains("flush inlined data to Parquet"),
        "{message}"
    );
    // Nothing changed.
    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20)]);
}

#[tokio::test(flavor = "multi_thread")]
async fn row_lineage_scan_refuses_tables_with_inlined_rows() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(10),
        ..Default::default()
    };
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    let provider = SqliteMetadataProvider::new(&ro_url(&t)).await.unwrap();
    let catalog = DuckLakeCatalog::new(provider)
        .unwrap()
        .with_row_lineage(true);
    let ctx = SessionContext::new();
    ctx.register_catalog("ducklake", Arc::new(catalog));
    let error = match ctx.sql("SELECT rowid, id FROM ducklake.main.t").await {
        Ok(df) => df.collect().await.expect_err("rowid scan must refuse"),
        Err(e) => e,
    };
    let message = error.to_string();
    assert!(
        message.contains("row-lineage (rowid) scan on a table with inlined rows is not supported")
            && message.contains("flush inlined data to Parquet"),
        "{message}"
    );

    // The same catalog still serves non-rowid reads, inlined rows included.
    let batches = ctx
        .sql("SELECT id, val FROM ducklake.main.t ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 2);
}

#[tokio::test(flavor = "multi_thread")]
async fn crate_and_duckdb_round_trip_inlined_rows() {
    let t = TempDir::new().unwrap();
    let writer = Arc::new(make_writer(&t).await);
    let options = DuckLakeWriteOptions {
        data_inlining_row_limit: Some(10),
        ..Default::default()
    };
    DuckLakeTableWriter::new(writer, object_store())
        .unwrap()
        .with_options(&options)
        .write_table("main", "t", &[batch(vec![1, 2], vec![10, 20])])
        .await
        .unwrap();

    let pool = SqlitePool::connect(&rw_url(&t)).await.unwrap();
    sqlx::query(
        "ALTER TABLE ducklake_snapshot
         ADD COLUMN next_catalog_id BIGINT NOT NULL DEFAULT 1000",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query(
        "ALTER TABLE ducklake_snapshot
         ADD COLUMN next_file_id BIGINT NOT NULL DEFAULT 1000",
    )
    .execute(&pool)
    .await
    .unwrap();
    sqlx::query("ALTER TABLE ducklake_schema ADD COLUMN schema_uuid VARCHAR")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("UPDATE ducklake_schema SET schema_uuid = '00000000-0000-0000-0000-000000000001'")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_table ADD COLUMN table_uuid VARCHAR")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("UPDATE ducklake_table SET table_uuid = '00000000-0000-0000-0000-000000000002'")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_data_file ADD COLUMN file_order BIGINT")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_data_file ADD COLUMN file_format VARCHAR")
        .execute(&pool)
        .await
        .unwrap();
    sqlx::query("ALTER TABLE ducklake_delete_file ADD COLUMN format VARCHAR")
        .execute(&pool)
        .await
        .unwrap();
    for ddl in [
        "CREATE TABLE ducklake_column_mapping (mapping_id BIGINT, table_id BIGINT, type VARCHAR)",
        "CREATE TABLE ducklake_column_tag (table_id BIGINT, column_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, key VARCHAR, value VARCHAR)",
        "CREATE TABLE ducklake_file_variant_stats (data_file_id BIGINT, table_id BIGINT, column_id BIGINT, variant_path VARCHAR, shredded_type VARCHAR, column_size_bytes BIGINT, value_count BIGINT, null_count BIGINT, min_value VARCHAR, max_value VARCHAR, contains_nan BOOLEAN, extra_stats VARCHAR)",
        "CREATE TABLE ducklake_macro (schema_id BIGINT, macro_id BIGINT, macro_name VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT)",
        "CREATE TABLE ducklake_macro_impl (macro_id BIGINT, impl_id BIGINT, dialect VARCHAR, sql VARCHAR, type VARCHAR)",
        "CREATE TABLE ducklake_macro_parameters (macro_id BIGINT, impl_id BIGINT, column_id BIGINT, parameter_name VARCHAR, parameter_type VARCHAR, default_value VARCHAR, default_value_type VARCHAR)",
        "CREATE TABLE ducklake_name_mapping (mapping_id BIGINT, column_id BIGINT, source_name VARCHAR, target_field_id BIGINT, parent_column BIGINT, is_partition BOOLEAN)",
        "CREATE TABLE ducklake_tag (object_id BIGINT, begin_snapshot BIGINT, end_snapshot BIGINT, key VARCHAR, value VARCHAR)",
        "CREATE TABLE IF NOT EXISTS ducklake_view (view_id BIGINT, view_uuid VARCHAR, begin_snapshot BIGINT, end_snapshot BIGINT, schema_id BIGINT, view_name VARCHAR, dialect VARCHAR, sql VARCHAR, column_aliases VARCHAR)",
    ] {
        sqlx::query(ddl).execute(&pool).await.unwrap();
    }
    pool.close().await;

    let attach = format!(
        "LOAD ducklake; LOAD sqlite; ATTACH 'ducklake:sqlite:{}' AS lake;",
        t.path().join("test.db").display()
    );
    let output = Command::new("duckdb")
        .args([
            "-csv",
            "-noheader",
            ":memory:",
            "-c",
            &format!("{attach} SELECT id, val FROM lake.main.t ORDER BY id;"),
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "DuckDB read failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(String::from_utf8(output.stdout).unwrap(), "1,10\n2,20\n");

    let output = Command::new("duckdb")
        .args([":memory:", "-c", &format!("{attach} INSERT INTO lake.main.t VALUES (3, 30);")])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "DuckDB insert failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    assert_eq!(read_rows(&t, None).await, vec![(1, 10), (2, 20), (3, 30)]);
}
