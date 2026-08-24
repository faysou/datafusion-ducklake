#![cfg(all(feature = "write-sqlite", feature = "write-postgres"))]

use std::sync::atomic::{AtomicBool, Ordering};

use datafusion_ducklake::{
    ColumnDef, DataFileInfo, DuckLakeError, MetadataProvider, MetadataWriter, MulticatalogManager,
    SnapshotCommitMetadata, SqliteMetadataProvider, SqliteMetadataWriter, WriteMode,
};
use tempfile::TempDir;
use testcontainers::runners::AsyncRunner;
use testcontainers_modules::postgres::Postgres;

fn columns() -> Vec<ColumnDef> {
    vec![ColumnDef::new("value", "BIGINT", false).unwrap()]
}

fn write_contract_data(writer: &dyn MetadataWriter, identity: &str) -> (i64, i64) {
    let setup = writer
        .begin_write_transaction("main", "events", &columns(), WriteMode::Replace)
        .unwrap();
    let commit = writer
        .register_data_file_with_commit_metadata(
            setup.table_id,
            "main",
            "events",
            setup.snapshot_id,
            &DataFileInfo::new("events.parquet", 128, 1),
            WriteMode::Replace,
            setup.base_snapshot_id,
            &columns(),
            &setup.column_ids,
            &SnapshotCommitMetadata::new()
                .with_author("contract")
                .with_message("metadata contract")
                .with_extra_info(identity),
            None,
        )
        .unwrap();
    (setup.table_id, commit.snapshot_id)
}

fn assert_metadata_contract(
    provider: &dyn MetadataProvider,
    writer: &dyn MetadataWriter,
    identity: &str,
    table_id: i64,
    snapshot_id: i64,
) {
    let changes = provider.list_snapshot_changes().unwrap();
    let change = changes
        .iter()
        .find(|change| change.snapshot_id == snapshot_id)
        .unwrap();
    assert_eq!(change.author.as_deref(), Some("contract"));
    assert_eq!(change.commit_message.as_deref(), Some("metadata contract"));
    assert_eq!(change.commit_extra_info.as_deref(), Some(identity));
    assert_eq!(
        provider
            .find_snapshot_by_commit_extra_info(identity)
            .unwrap(),
        Some(snapshot_id),
    );

    writer
        .set_table_setting(table_id, "data_inlining_row_limit", "42")
        .unwrap();
    assert_eq!(
        provider
            .get_metadata_settings(None, Some(table_id))
            .unwrap()
            .get("data_inlining_row_limit")
            .map(String::as_str),
        Some("42"),
    );

    let called = AtomicBool::new(false);
    writer
        .with_commit_lock(
            identity,
            Box::new(|| {
                called.store(true, Ordering::SeqCst);
                Ok(())
            }),
        )
        .unwrap();
    assert!(called.load(Ordering::SeqCst));

    let error = writer
        .with_commit_lock(
            identity,
            Box::new(|| Err(DuckLakeError::Internal("operation failed".to_string()))),
        )
        .unwrap_err();
    assert_eq!(error.to_string(), "Internal error: operation failed");

    // The failed operation released the lock: re-acquiring under the same
    // identity succeeds.
    writer
        .with_commit_lock(identity, Box::new(|| Ok(())))
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn sqlite_metadata_contract() {
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("catalog.sqlite");
    let url = format!("sqlite:{}?mode=rwc", path.display());
    let writer = SqliteMetadataWriter::new_with_init(&url).await.unwrap();
    writer.set_data_path(temp.path().to_str().unwrap()).unwrap();
    let provider = SqliteMetadataProvider::new(&url).await.unwrap();
    let (table_id, snapshot_id) = write_contract_data(&writer, "sqlite-contract");

    assert_metadata_contract(&provider, &writer, "sqlite-contract", table_id, snapshot_id);
}

#[tokio::test(flavor = "multi_thread")]
#[cfg_attr(all(feature = "skip-tests-with-docker", target_os = "macos"), ignore)]
async fn postgres_metadata_contract() {
    let container = Postgres::default().start().await.unwrap();
    let port = container.get_host_port_ipv4(5432).await.unwrap();
    let url = format!("postgresql://postgres:postgres@127.0.0.1:{port}/postgres");
    let manager = MulticatalogManager::connect(&url, 5).await.unwrap();
    let catalog_id = manager.create_catalog("metadata_contract").await.unwrap();
    let writer = manager.writer(catalog_id).await.unwrap();
    writer.set_data_path("/tmp/metadata-contract").unwrap();
    let provider = manager.provider(catalog_id).await.unwrap();
    let (table_id, snapshot_id) = write_contract_data(&writer, "postgres-contract");

    assert_metadata_contract(
        &provider,
        &writer,
        "postgres-contract",
        table_id,
        snapshot_id,
    );
}
