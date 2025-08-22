use glob::glob;
use regex::Regex;
use std::{
    any::type_name,
    path::{Path, PathBuf},
};

pub static CACHE_DIR: &str = "./cache/";
pub static TEMP_CACHE_DIR: &str = "./cache/tmp/";

fn _type_of<T>(_: T) -> &'static str {
    type_name::<T>()
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
    cache_filename: String,
) -> Result<(String, PathBuf), Box<dyn std::error::Error>> {
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

    Ok((id.to_string(), PathBuf::from(parent_dir)))
}

#[allow(dead_code)]
pub fn cache_file_name(
    original_filename: String,
    target_type: FileType,
    sequence_number: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    if target_type == FileType::Test {
        return Ok(CACHE_DIR.to_string()
            + file_name_from_id_and_sequence_for_type(
                target_type,
                original_filename,
                sequence_number,
            )
            .as_str());
    }
    let (id, parent_dir) = graph_id_and_dir_from_cache_file_name(original_filename)?;
    let new_filename = file_name_from_id_and_sequence_for_type(target_type, id, sequence_number);
    Ok(parent_dir.join(new_filename).to_string_lossy().into_owned())
}

pub fn id_for_subgraph_export(
    id: String,
    sequence_number: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
    // this isn't a filename --- it's an id to be used in filenames
    match sequence_number {
        Some(i) => Ok(format!("{}_{}_{}", "masked_export", i, id)),
        None => Ok(format!("{}_{}", "masked_export", id)),
    }
}

#[allow(dead_code)]
pub fn cleanup_cache() -> Result<(), Box<dyn std::error::Error>> {
    let cache_entries = glob((CACHE_DIR.to_string() + "/*.tmp").as_str()).map_err(
        |e| -> Box<dyn std::error::Error> { format!("error cleaning up cache: {e}").into() },
    )?;
    for entry in cache_entries {
        std::fs::remove_file(entry.map_err(|_e| -> Box<dyn std::error::Error> {
            format!("error cleaning up cache entry: {_e}").into()
        })?)?;
    }
    Ok(())
}

#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub enum FileType {
    Edges,
    Index,
    Fst,
    EulerPath,
    EulerTmp,
    KmerTmp,
    KmerSortedTmp,
    KCore,
    KTruss,
    EdgeReciprocal,
    EdgeOver,
    HyperBall,
    HyperBallDistances,
    HyperBallInvDistances,
    HyperBallClosenessCentrality,
    HyperBallHarmonicCentrality,
    HyperBallLinCentrality,
    GVELouvain,
    Test,
}

pub fn file_name_from_id_and_sequence_for_type(
    target_type: FileType,
    id: String,
    sequence_number: Option<usize>,
) -> String {
    match target_type {
        FileType::Edges => format!("{}_{}.{}", "edges", id, "mmap"),
        FileType::Index => format!("{}_{}.{}", "index", id, "mmap"),
        FileType::Fst => format!("{}_{}.{}", "fst", id, "fst"),
        FileType::EulerTmp => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "eulertmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "eulertmp", id, "tmp"),
        },
        FileType::EulerPath => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "eulerpath", i, id, "mmap"),
            None => format!("{}_{}.{}", "eulerpath", id, "mmap"),
        },
        FileType::KmerTmp => format!("{}_{}.{}", "kmertmpfile", id, "tmp"),
        FileType::KmerSortedTmp => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "kmersortedtmpfile", i, id, "tmp"),
            None => format!("{}_{}.{}", "kmersortedtmpfile", id, "mmap"),
        },
        FileType::KCore => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "kcore_tmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "kcores", id, "mmap"),
        },
        FileType::KTruss => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "ktruss_tmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "ktruss", id, "mmap"),
        },
        FileType::EdgeReciprocal => format!("{}_{}.{}", "edge_reciprocal", id, "mmap"),
        FileType::EdgeOver => format!("{}_{}.{}", "edge_over", id, "mmap"),
        FileType::HyperBall => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "hyperball", i, id, "tmp"),
            None => format!("{}_{}.{}", "hyperball", id, "mmap"),
        },
        FileType::HyperBallDistances => format!("{}_{}.{}", "hypeball_distances", id, "mmap"),
        FileType::HyperBallInvDistances => {
            format!("{}_{}.{}", "hyperball_inv_distances", id, "mmap")
        }
        FileType::HyperBallClosenessCentrality => {
            format!("{}_{}.{}", "hyperball_closeness", id, "mmap")
        }
        FileType::HyperBallHarmonicCentrality => {
            format!("{}_{}.{}", "hyperball_harmonic", id, "mmap")
        }
        FileType::HyperBallLinCentrality => {
            format!("{}_{}.{}", "hyperball_inv_lin", id, "mmap")
        }
        FileType::GVELouvain => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "louvaintmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "louvain", id, "mmap"),
        },
        FileType::Test => {
            let random_id = rand::random::<u128>().to_string();
            format!("{}_{}.{}", "test", random_id, "tmp")
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FileType::Edges => "Edges",
            FileType::Index => "Index",
            FileType::Fst => "Fst",
            FileType::EulerPath => "EulerPath",
            FileType::EulerTmp => "EulerTmp",
            FileType::KmerTmp => "KmerTmp",
            FileType::KmerSortedTmp => "KmerSortedTmp",
            FileType::KCore => "KCore",
            FileType::KTruss => "KTruss",
            FileType::EdgeReciprocal => "EdgeReciprocal",
            FileType::EdgeOver => "EdgeOver",
            FileType::HyperBall => "HyperBall",
            FileType::HyperBallDistances => "HyperBallDistances",
            FileType::HyperBallInvDistances => "HyperBallInverseDistances",
            FileType::HyperBallClosenessCentrality => "HyperBallClosenessCentrality",
            FileType::HyperBallHarmonicCentrality => "HyperBallHarmonicCentrality",
            FileType::HyperBallLinCentrality => "HyperBallLinCentrality",
            FileType::GVELouvain => "Louvain",
            FileType::Test => "Test",
        };
        write!(f, "{}", s)
    }
}

/// Checks if a `val` is a normal `f64`. Outputs a result with a custom error message.
///
/// # Arguments
///
/// * `val`: `f64` --- the value to be checked.
/// * `op_description`: `&str` --- the custom error message.
///
/// # Returns
///
/// `Ok(val)` if `val` is normal, or `Err(op_description.into())` if not.
///
#[inline(always)]
pub fn f64_is_nomal(val: f64, op_description: &str) -> Result<f64, Box<dyn std::error::Error>> {
    if !val.is_normal() {
        return Err(format!("error hk-relax abnormal value at {op_description} = {val}",).into());
    }
    Ok(val)
}

/// Safely converts an `f64` `val` into `usize`. Outputs an option.
///
/// Conversion is successful if `val` is normal, bigger than zero (not equal) and less than or equal to `usize::MAX`.
///
/// # Arguments
///
/// * `val`: `f64` --- the value to be cast.
///
/// # Returns
///
/// `Some(val as usize)` if successful, or None if not.
///
pub fn f64_to_usize_safe(val: f64) -> Option<usize> {
    if val.is_normal() && val > 0f64 && val <= usize::MAX as f64 {
        Some(val as usize) // truncates toward zero
    } else {
        None
    }
}
