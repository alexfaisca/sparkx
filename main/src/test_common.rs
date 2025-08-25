use crate::{shared_slice::AbstractedProceduralMemoryMut, utils::graph_id_from_cache_file_name};

use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, OnceLock},
};

static CACHE_DIR: &str = "./.test_cache/";

type CacheMap = HashMap<String, String>;

static TEST_MEM_CACHE: OnceLock<Mutex<CacheMap>> = OnceLock::new();

fn mem_cache() -> &'static Mutex<CacheMap> {
    TEST_MEM_CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    if !Path::new(CACHE_DIR).exists() {
        fs::create_dir_all(CACHE_DIR)?;
    }
    Ok(PathBuf::from(CACHE_DIR))
}

fn cache_file_for(graph_path: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
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
    Ok(cache_dir()?.join(filename).with_extension("mmap"))
}

/// Create a stable 256-bit hex id from a filename (path & extension are ignored).
///
/// - Deterministic across runs and machines.
/// - Uses SHA-256.
/// - Returns a 64-char lowercase hex string.
pub fn id_from_filename(name: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(name.as_bytes());
    let digest = hasher.finalize();

    // Take the first 16 bytes (128 bits) and hex-encode.
    digest.iter().map(|b| format!("{:02x}", b)).collect()
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
    let id = id_from_filename(file_name);

    Ok(id.to_string())
}

fn initialize_or_get_dataset_graph(
    graph_path: &Path,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let cache_id = graph_id_from_dataset_file_name(graph_path)?;
    Ok("".into())
}
