use glob::glob;
use regex::Regex;
use sha2::{Digest, Sha256};
use std::{
    any::type_name,
    fmt::Write,
    path::{Path, PathBuf},
};

#[cfg(not(any(test, feature = "bench")))]
pub static CACHE_DIR: &str = "./.cache/";
#[cfg(any(test, feature = "bench"))]
pub static CACHE_DIR: &str = "./.test_cache/";

#[cfg(any(test, feature = "bench"))]
pub static EXACT_VALUE_CACHE_DIR: &str = "exact_values/";

fn _type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

/// Create a stable 256-bit hex id from a filename (path & extension are ignored).
///
/// - Deterministic across runs and machines.
/// - Uses SHA-256.
/// - Returns a 64-char lowercase hex string.
///
/// # Arguments
///
/// * `key` --- the key to be hashed in order to obtain the hexadecimal id.
///
#[allow(dead_code)]
pub fn id_from_filename(key: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
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
pub fn graph_id_from_cache_file_name(
    cache_filename: String,
) -> Result<String, Box<dyn std::error::Error>> {
    let path = Path::new(cache_filename.as_str());

    let file_name = path
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- path not found".into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- empty path".into()
        })?;

    // extract id from filename
    let re = Regex::new(r#"^(?:[a-zA-Z0-9_]+_)(\w+)(\.[a-zA-Z0-9]+$)"#).map_err(
        |e| -> Box<dyn std::error::Error> {
            format!("error analyzing file name (regex failed): {e}").into()
        },
    )?;

    let caps = re
        .captures(file_name)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error capturing file name (regex captures failed: fail point 1)".into()
        })?;

    let id = caps
        .get(1)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error capturing file name (regex captures failed: fail point 2)".into()
        })?
        .as_str();

    Ok(id.to_string())
}

#[allow(dead_code)]
fn graph_id_and_dir_from_cache_file_name(
    cache_filename: &str,
) -> Result<(&str, PathBuf), Box<dyn std::error::Error>> {
    let path = Path::new(cache_filename);

    let file_name = path
        .file_name()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- path not found".into()
        })?
        .to_str()
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error invalid file path --- path empty".into()
        })?;

    let parent_dir = path.parent().unwrap_or_else(|| Path::new(""));

    // extract id from filename
    let re = Regex::new(r#"^(?:[a-zA-Z0-9_]+_)(\w+)(\.[a-zA-Z0-9]+$)"#).map_err(
        |e| -> Box<dyn std::error::Error> {
            format!("error analyzing file name (regex failed): {e}").into()
        },
    )?;

    let caps = re
        .captures(file_name)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error capturing file name (regex captures failed: fail point 1)".into()
        })?;

    let id = caps
        .get(1)
        .ok_or_else(|| -> Box<dyn std::error::Error> {
            "error capturing file name (regex captures failed: fail point 2)".into()
        })?
        .as_str();

    Ok((id, PathBuf::from(parent_dir)))
}

#[allow(dead_code)]
pub fn cache_file_name(
    original_filename: &str,
    target_type: FileType,
    sequence_number: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    #[cfg(any(test, feature = "bench"))]
    if target_type == FileType::Test(H::H) {
        return Ok(CACHE_DIR.to_string()
            + file_name_from_id_and_sequence_for_type(
                target_type,
                original_filename,
                sequence_number,
            )
            .as_str());
    }
    // If in test or bench mode store/lookup exact values in dir {cache}/exact_values/
    #[cfg(any(test, feature = "bench"))]
    if target_type == FileType::ExactClosenessCentrality(H::H)
        || target_type == FileType::ExactHarmonicCentrality(H::H)
        || target_type == FileType::ExactLinCentrality(H::H)
    {
        let (id, parent_dir) = graph_id_and_dir_from_cache_file_name(original_filename)?;
        let new_filename =
            file_name_from_id_and_sequence_for_type(target_type, id, sequence_number);
        return Ok(parent_dir
            .join(EXACT_VALUE_CACHE_DIR)
            .join(new_filename)
            .to_string_lossy()
            .into_owned());
    }
    let (id, parent_dir) = graph_id_and_dir_from_cache_file_name(original_filename)?;
    let new_filename = file_name_from_id_and_sequence_for_type(target_type, id, sequence_number);
    Ok(parent_dir.join(new_filename).to_string_lossy().into_owned())
}

pub fn id_for_subgraph_export(id: String, sequence_number: Option<usize>) -> String {
    // this isn't a filename --- it's an id to be used in filenames
    match sequence_number {
        Some(i) => format!("{}<{}>{}", "maskedexport", i, id),
        None => format!("{}{}", "maskedexport", id),
    }
}

/// Remove all cached `.tmp` files for a given `id` and [`FileType`] in the cache directory.
///
/// [`FileType`]: ./enum.FileType.html#
#[allow(dead_code)]
pub(crate) fn cleanup_cache(
    id: &str,
    target_type: FileType,
) -> Result<(), Box<dyn std::error::Error>> {
    let match_entries_to =
        CACHE_DIR.to_string() + suffix_for_file_type(target_type) + "*" + id + ".tmp";
    let cache_entries = glob(&match_entries_to).map_err(|e| -> Box<dyn std::error::Error> {
        format!("error cleaning up cache for entries with name {match_entries_to}: {e}").into()
    })?;
    for entry in cache_entries {
        std::fs::remove_file(entry.map_err(|e| -> Box<dyn std::error::Error> {
            format!("error cleaning up cache entry: {e}").into()
        })?)?;
    }
    Ok(())
}

/// Remove all cached `.tmp` in the cache directory.
///
#[allow(dead_code)]
pub(crate) fn remove_tmp_files_from_cache() -> Result<(), Box<dyn std::error::Error>> {
    let cache_entries = glob((CACHE_DIR.to_string() + "/*.tmp").as_str()).map_err(
        |e| -> Box<dyn std::error::Error> { format!("error cleaning up cache: {e}").into() },
    )?;
    for entry in cache_entries {
        std::fs::remove_file(entry.map_err(|e| -> Box<dyn std::error::Error> {
            format!("error cleaning up cache entry: {e}").into()
        })?)?;
    }
    Ok(())
}

/// Hides file types from users.
///
/// Short for "Hidden".
#[derive(Clone, Debug, PartialEq)]
pub enum H {
    H,
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code, private_interfaces, clippy::upper_case_acronyms)]
pub enum FileType {
    /// Only member visible to users
    General,
    Edges(H),
    Index(H),
    Metalabel(H),
    Helper(H),
    BFS(H),
    DFS(H),
    EulerTrail(H),
    KCoreBZ(H),
    KCoreLEA(H),
    KTrussBEA(H),
    KTrussPKT(H),
    ClusteringCoefficient(H),
    EdgeReciprocal(H),
    EdgeOver(H),
    HyperBall(H),
    HyperBallDistances(H),
    HyperBallInvDistances(H),
    HyperBallClosenessCentrality(H),
    HyperBallHarmonicCentrality(H),
    HyperBallLinCentrality(H),
    GVELouvain(H),
    #[cfg(any(test, feature = "bench"))]
    Test(H),
    ExactClosenessCentrality(H),
    ExactHarmonicCentrality(H),
    ExactLinCentrality(H),
}

pub fn cache_file_name_from_id(
    target_type: FileType,
    id: &str,
    sequence_number: Option<usize>,
) -> String {
    CACHE_DIR.to_string()
        + file_name_from_id_and_sequence_for_type(target_type, id, sequence_number).as_str()
}

fn file_name_from_id_and_sequence_for_type(
    target: FileType,
    id: &str,
    seq: Option<usize>,
) -> String {
    if target == FileType::HyperBallClosenessCentrality(H::H)
        || target == FileType::HyperBallHarmonicCentrality(H::H)
        || target == FileType::HyperBallLinCentrality(H::H)
    {
        return match seq {
            None => format!("{}_{}.{}", suffix_for_file_type(target), id, "mmap"),
            // sequenced files are temporary
            Some(i) => format!("{}_{}_{}.{}", suffix_for_file_type(target), i, id, "mmap"),
        };
    }
    match seq {
        None => format!("{}_{}.{}", suffix_for_file_type(target), id, "mmap"),
        // sequenced files are temporary
        Some(i) => format!("{}_{}_{}.{}", suffix_for_file_type(target), i, id, "tmp"),
    }
}

pub fn suffix_for_file_type(target_type: FileType) -> &'static str {
    static SUFFIX_FOR_GENERAL: &str = "miscelanious";
    static SUFFIX_FOR_EDGES: &str = "edges";
    static SUFFIX_FOR_INDEX: &str = "index";
    static SUFFIX_FOR_METALABEL: &str = "fst";
    static SUFFIX_FOR_HELPER: &str = "helper";
    static SUFFIX_FOR_BFS: &str = "bfs";
    static SUFFIX_FOR_DFS: &str = "dfs";
    static SUFFIX_FOR_EULER_TRAIL: &str = "eulertrail";
    static SUFFIX_FOR_KCORE_BZ: &str = "kcoresbz";
    static SUFFIX_FOR_KCORE_LEA: &str = "kcoreslea";
    static SUFFIX_FOR_KTRUSS_BEA: &str = "ktrusslea";
    static SUFFIX_FOR_KTRUSS_PKT: &str = "ktrusspkt";
    static SUFFIX_FOR_CLUSTERING_COEFFICIENT: &str = "clusteringcoefficient";
    static SUFFIX_FOR_EDGE_RECIPROCAL: &str = "edgereciprocal";
    static SUFFIX_FOR_EDGE_OVER: &str = "edgeover";
    static SUFFIX_FOR_HYPERBALL: &str = "hyperball";
    static SUFFIX_FOR_HYPERBALL_DISTANCES: &str = "hyperballdistances";
    static SUFFIX_FOR_HYPERBALL_INV_DISTANCES: &str = "hyperballinvdistances";
    static SUFFIX_FOR_HYPERBALL_CLOSENESS_CENTRALITY: &str = "hyperballcloseness";
    static SUFFIX_FOR_HYPERBALL_HARMONIC_CENTRALITY: &str = "hyperballharmonic";
    static SUFFIX_FOR_HYPERBALL_LIN_CENTRALITY: &str = "hyperballlin";
    static SUFFIX_FOR_GVE_LOUVAIN: &str = "louvain";
    static SUFFIX_FOR_EXACT_CLOSENESS_CENTRALITY: &str = "exactclosenesscentrality";
    static SUFFIX_FOR_EXACT_HARMONIC_CENTRALITY: &str = "exactharmoniccentrality";
    static SUFFIX_FOR_EXACT_LIN_CENTRALITY: &str = "exactlincentrality";
    #[cfg(any(test, feature = "bench"))]
    static SUFFIX_FOR_TEST: &str = "test";

    match target_type {
        FileType::General => SUFFIX_FOR_GENERAL,
        FileType::Edges(_) => SUFFIX_FOR_EDGES,
        FileType::Index(_) => SUFFIX_FOR_INDEX,
        FileType::Metalabel(_) => SUFFIX_FOR_METALABEL,
        FileType::Helper(_) => SUFFIX_FOR_HELPER,
        FileType::BFS(_) => SUFFIX_FOR_BFS,
        FileType::DFS(_) => SUFFIX_FOR_DFS,
        FileType::EulerTrail(_) => SUFFIX_FOR_EULER_TRAIL,
        FileType::KCoreBZ(_) => SUFFIX_FOR_KCORE_BZ,
        FileType::KCoreLEA(_) => SUFFIX_FOR_KCORE_LEA,
        FileType::KTrussBEA(_) => SUFFIX_FOR_KTRUSS_BEA,
        FileType::KTrussPKT(_) => SUFFIX_FOR_KTRUSS_PKT,
        FileType::ClusteringCoefficient(_) => SUFFIX_FOR_CLUSTERING_COEFFICIENT,
        FileType::EdgeReciprocal(_) => SUFFIX_FOR_EDGE_RECIPROCAL,
        FileType::EdgeOver(_) => SUFFIX_FOR_EDGE_OVER,
        FileType::HyperBall(_) => SUFFIX_FOR_HYPERBALL,
        FileType::HyperBallDistances(_) => SUFFIX_FOR_HYPERBALL_DISTANCES,
        FileType::HyperBallInvDistances(_) => SUFFIX_FOR_HYPERBALL_INV_DISTANCES,
        FileType::HyperBallClosenessCentrality(_) => SUFFIX_FOR_HYPERBALL_CLOSENESS_CENTRALITY,
        FileType::HyperBallHarmonicCentrality(_) => SUFFIX_FOR_HYPERBALL_HARMONIC_CENTRALITY,
        FileType::HyperBallLinCentrality(_) => SUFFIX_FOR_HYPERBALL_LIN_CENTRALITY,
        FileType::GVELouvain(_) => SUFFIX_FOR_GVE_LOUVAIN,
        FileType::ExactClosenessCentrality(_) => SUFFIX_FOR_EXACT_CLOSENESS_CENTRALITY,
        FileType::ExactHarmonicCentrality(_) => SUFFIX_FOR_EXACT_HARMONIC_CENTRALITY,
        FileType::ExactLinCentrality(_) => SUFFIX_FOR_EXACT_LIN_CENTRALITY,
        #[cfg(any(test, feature = "bench"))]
        FileType::Test(_) => SUFFIX_FOR_TEST,
    }
}

/// Given a path to `unitig_{name}.edges`,
/// return the companion `unitig_annotated_{name}.nodes`.
#[cfg(feature = "nodes_edges")]
pub(super) fn edges_to_nodes<P: AsRef<Path>>(edges_file: P) -> Option<PathBuf> {
    let path = edges_file.as_ref();

    let parent = path.parent();
    let stem = path.file_stem()?.to_str()?;
    let ext = path.extension()?.to_str()?;

    if ext != "edges" || !stem.starts_with("unitig_") {
        return None;
    }

    let name = &stem["unitig_".len()..];
    if name.is_empty() {
        return None;
    }

    let new_name = format!("unitig_annotated_{}.nodes", name);
    Some(parent.map_or_else(
        || PathBuf::from(new_name.clone()),
        |dir| dir.join(new_name.clone()),
    ))
}

/// Given a path to `unitig_annotated_{name}.nodes`,
/// return the companion `unitig_{name}.edges`.
#[cfg(feature = "nodes_edges")]
pub(super) fn nodes_to_edges<P: AsRef<Path>>(nodes_file: P) -> Option<PathBuf> {
    let path = nodes_file.as_ref();

    let parent = path.parent();
    let stem = path.file_stem()?.to_str()?;
    let ext = path.extension()?.to_str()?;

    if ext != "nodes" || !stem.starts_with("unitig_annotated_") {
        return None;
    }

    let name = &stem["unitig_annotated_".len()..];
    if name.is_empty() {
        return None;
    }

    let new_name = format!("unitig_{}.edges", name);
    Some(parent.map_or_else(
        || PathBuf::from(new_name.clone()),
        |dir| dir.join(new_name.clone()),
    ))
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FileType::General => "Miscelanious",
            FileType::Edges(_) => "Edges",
            FileType::Index(_) => "Index",
            FileType::Metalabel(_) => "Metalabel",
            FileType::Helper(_) => "Helper",
            FileType::BFS(_) => "BFS",
            FileType::DFS(_) => "DFS",
            FileType::EulerTrail(_) => "EulerTrail",
            FileType::KCoreBZ(_) => "KCoreBatageljZaversnik",
            FileType::KCoreLEA(_) => "KCoreLiuEtAl",
            FileType::KTrussBEA(_) => "KTrussBurkhardtEtAl",
            FileType::KTrussPKT(_) => "KTrussPKT",
            FileType::ClusteringCoefficient(_) => "ClusteringCoefficient",
            FileType::EdgeReciprocal(_) => "EdgeReciprocal",
            FileType::EdgeOver(_) => "EdgeOver",
            FileType::HyperBall(_) => "HyperBall",
            FileType::HyperBallDistances(_) => "HyperBallDistances",
            FileType::HyperBallInvDistances(_) => "HyperBallInverseDistances",
            FileType::HyperBallClosenessCentrality(_) => "HyperBallClosenessCentrality",
            FileType::HyperBallHarmonicCentrality(_) => "HyperBallHarmonicCentrality",
            FileType::HyperBallLinCentrality(_) => "HyperBallLinCentrality",
            FileType::GVELouvain(_) => "Louvain",
            FileType::ExactClosenessCentrality(_) => "ExactClosenessCentrality",
            FileType::ExactHarmonicCentrality(_) => "ExactHarmonicCentrality",
            FileType::ExactLinCentrality(_) => "ExactLinCentrality",
            #[cfg(any(test, feature = "bench"))]
            FileType::Test(_) => "Test",
        };
        write!(f, "FileType {{{}}}", s)
    }
}
