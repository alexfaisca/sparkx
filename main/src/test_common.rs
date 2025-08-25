use crate::{
    generic_edge::{TinyEdgeType, TinyLabelStandardEdge},
    generic_memory_map::GraphCache,
    utils::{FileType, H, cache_file_name_from_id},
};
#[allow(unused_imports)]
use crate::{
    shared_slice::AbstractedProceduralMemoryMut,
    utils::{CACHE_DIR, graph_id_from_cache_file_name},
};

use dashmap::DashMap;
use sha2::{Digest, Sha256};
use std::{
    fmt::Write,
    path::{Path, PathBuf},
    sync::OnceLock,
};

type TestCache = DashMap<String, OnceLock<GraphCache<TinyEdgeType, TinyLabelStandardEdge>>>;

static TEST_CACHE: OnceLock<TestCache> = OnceLock::new();

pub fn mem_cache() -> &'static TestCache {
    TEST_CACHE.get_or_init(DashMap::new)
}

#[allow(dead_code)]
fn cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if !Path::new(CACHE_DIR).exists() {
        std::fs::create_dir_all(CACHE_DIR)?;
    }
    Ok(PathBuf::from(CACHE_DIR))
}

#[allow(dead_code)]
fn cache_file_for(graph_path: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let filename = graph_path
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("error getting filename for dataset path {:?}", graph_path).into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            format!(
                "error getting filename for dataset path as str {:?}",
                graph_path
            )
            .into()
        })?;
    Ok(cache_file_name_from_id(
        FileType::Edges(H::H),
        id_from_filename(filename)?,
        None,
    ))
}

/// Create a stable 256-bit hex id from a filename (path & extension are ignored).
///
/// - Deterministic across runs and machines.
/// - Uses SHA-256.
/// - Returns a 64-char lowercase hex string.
pub fn id_from_filename(name: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    let digest = hasher.finalize();

    // hex-encode.
    let mut result = String::with_capacity(digest.len() * 2);
    digest
        .iter()
        .try_for_each(|b| -> Result<(), Box<dyn std::error::Error>> {
            write!(&mut result, "{:02x}", b)?;
            Ok(())
        })?;
    Ok(result)
}

#[allow(dead_code)]
fn graph_id_from_dataset_file_name(
    data_filename: &Path,
) -> Result<String, Box<dyn std::error::Error>> {
    let file_name = data_filename
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- path not found".into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- empty path".into()
        })?;

    let _parent_dir = data_filename.parent().unwrap_or_else(|| Path::new(""));
    id_from_filename(file_name)
}

#[allow(dead_code)]
pub fn get_or_init_dataset_cache_entry(
    graph_path: &Path,
) -> Result<GraphCache<TinyEdgeType, TinyLabelStandardEdge>, Box<dyn std::error::Error>> {
    let cache_id = graph_id_from_dataset_file_name(graph_path)?;

    // Get or create the per-key OnceLock
    let test_entry = mem_cache()
        .entry(cache_id.clone())
        .or_try_insert_with(|| -> Result<OnceLock<GraphCache<TinyEdgeType, TinyLabelStandardEdge>>, Box<dyn std::error::Error>> {
            let lock = OnceLock::new();
            Ok(lock)
        })?;
    let cache = test_entry.get_or_try_init(|| -> Result<GraphCache<TinyEdgeType, TinyLabelStandardEdge>, Box<dyn std::error::Error>> {
            let cache_filename = cache_file_for(graph_path)?;
            if Path::new(&cache_filename).exists() {
                eprintln!("found it {:?}", graph_path);
                GraphCache::<TinyEdgeType, TinyLabelStandardEdge>::open(cache_filename, None)
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting `GraphCache` instance for path {:?}: {:?}", graph_path, e).into()
                    })
            } else {
                eprintln!("built it {:?}", graph_path);
                GraphCache::<TinyEdgeType, TinyLabelStandardEdge>::from_file(
                    graph_path,
                    Some(cache_id),
                    None,
                    None,
                    )
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error creating `GraphCache` instance for path {:?}: {:?}", graph_path, e).into()
                    })
            }
    })?;

    Ok(cache.clone())
}
