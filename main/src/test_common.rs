use crate::utils::{CACHE_DIR, EXACT_VALUE_CACHE_DIR, id_from_filename};
use crate::{
    generic_edge::{GenericEdge, GenericEdgeType, TinyEdgeType, TinyLabelStandardEdge},
    generic_memory_map::{GraphCache, GraphMemoryMap},
    shared_slice::SharedSliceMut,
    utils::{FileType, H, cache_file_name, cache_file_name_from_id},
};

use dashmap::DashMap;
use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
};

type TestCache = DashMap<String, OnceLock<GraphCache<TinyEdgeType, TinyLabelStandardEdge>>>;
type ExactValueCache = DashMap<String, OnceLock<String>>;

static TEST_CACHE: OnceLock<TestCache> = OnceLock::new();
static EXACT_VALUE_CACHE: OnceLock<ExactValueCache> = OnceLock::new();

pub(crate) fn mem_cache() -> &'static TestCache {
    TEST_CACHE.get_or_init(DashMap::new)
}

pub(crate) fn exact_value_cache() -> &'static ExactValueCache {
    EXACT_VALUE_CACHE.get_or_init(DashMap::new)
}

#[allow(dead_code)]
fn cache_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(PathBuf::from(CACHE_DIR))
}

#[allow(dead_code)]
fn exact_value_dir() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let dir_str = CACHE_DIR.to_string() + EXACT_VALUE_CACHE_DIR;
    Ok(PathBuf::from(dir_str))
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
        &id_from_filename(filename)?,
        None,
    ))
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
pub(crate) fn get_or_init_dataset_cache_entry(
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
                GraphCache::<TinyEdgeType, TinyLabelStandardEdge>::open(&cache_filename, None)
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

#[allow(dead_code)]
pub(crate) fn get_or_init_dataset_exact_value<
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
>(
    graph_path: &Path,
    graph: &GraphMemoryMap<EdgeType, Edge>,
    value_type: FileType,
) -> Result<String, Box<dyn std::error::Error>> {
    if value_type != FileType::ExactClosenessCentrality(H::H)
        && value_type != FileType::ExactHarmonicCentrality(H::H)
        && value_type != FileType::ExactLinCentrality(H::H)
    {
        return Err(format!(
            "error exact value files may only be of type {}, {} or {}",
            FileType::ExactClosenessCentrality(H::H),
            FileType::ExactHarmonicCentrality(H::H),
            FileType::ExactLinCentrality(H::H),
        )
        .into());
    }
    let cache_id = graph_id_from_dataset_file_name(graph_path)?;
    let e_fn = cache_file_name(&graph.cache_fst_filename(), value_type.clone(), None)?;

    // Get or create the per-key OnceLock
    let test_entry = exact_value_cache()
        .entry(cache_id.clone())
        .or_try_insert_with(|| -> Result<OnceLock<String>, Box<dyn std::error::Error>> {
            let lock = OnceLock::new();
            Ok(lock)
        })?;
    test_entry
        .get_or_try_init(|| -> Result<String, Box<dyn std::error::Error>> {
            if Path::new(&e_fn).exists() {
                eprintln!("found exact vals for {:?} at {e_fn}", graph_path);
                Ok(e_fn)
            } else {
                eprintln!("built exact vals for {:?}", graph_path);
                eprintln!("built in {:?} {e_fn}", exact_value_dir());
                let petgraph_export = graph.export_petgraph_stripped()?;
                match value_type {
                    FileType::ExactClosenessCentrality(H::H) => {
                        use rustworkx_core::centrality::closeness_centrality;
                        let node_count = graph.size().map_or(0, |s| s);
                        let exact = closeness_centrality(&petgraph_export, false)
                            .iter()
                            .map(|opt| opt.unwrap_or(0.))
                            .collect::<Vec<f64>>();
                        let mut e = SharedSliceMut::<f64>::abst_mem_mut(&e_fn, node_count, true)?;
                        for (idx, val) in exact.iter().enumerate() {
                            *e.get_mut(idx) = *val;
                        }
                        e.flush()?;
                    }
                    FileType::ExactHarmonicCentrality(H::H) => {
                        // FIXME: find out how!
                        return Err("error can't compute harmonic centrality values".into());
                    }
                    FileType::ExactLinCentrality(H::H) => {
                        // FIXME: find out how!
                        return Err("error can't compute lin's centrality values".into());
                    }
                    _ => {}
                };
                Ok(e_fn)
            }
        })
        .cloned()
}
