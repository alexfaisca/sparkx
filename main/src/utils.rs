use glob::glob;
use regex::Regex;
use std::{
    any::type_name,
    path::{Path, PathBuf},
};

#[cfg(not(any(test, feature = "bench")))]
pub static CACHE_DIR: &str = "./.cache/";
#[cfg(any(test, feature = "bench"))]
pub static CACHE_DIR: &str = "./.test_cache/";

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
    #[cfg(test)]
    if target_type == FileType::Test(H::H) {
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

/// Hides file types from users.
///
/// Short for "Hidden".
#[derive(Debug, PartialEq)]
pub(crate) enum H {
    H,
}

#[derive(Debug, PartialEq)]
#[allow(dead_code)]
pub enum FileType {
    /// Only member visible to users
    General,
    Edges(H),
    Index(H),
    Fst(H),
    EulerPath(H),
    EulerTmp(H),
    KmerTmp(H),
    KmerSortedTmp(H),
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
    #[cfg(test)]
    ExactClosenessCentrality(H),
    #[cfg(test)]
    ExactHarmonicCentrality(H),
    #[cfg(test)]
    ExactLinCentrality(H),
    #[cfg(test)]
    Test(H),
}

pub fn cache_file_name_from_id(
    target_type: FileType,
    id: String,
    sequence_number: Option<usize>,
) -> String {
    CACHE_DIR.to_string()
        + file_name_from_id_and_sequence_for_type(target_type, id, sequence_number).as_str()
}

fn file_name_from_id_and_sequence_for_type(
    target_type: FileType,
    id: String,
    sequence_number: Option<usize>,
) -> String {
    match target_type {
        FileType::General => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "miscelanious", i, id, "tmp"),
            None => format!("{}_{}.{}", "miscelanious", id, "mmap"),
        },
        FileType::Edges(_) => format!("{}_{}.{}", "edges", id, "mmap"),
        FileType::Index(_) => format!("{}_{}.{}", "index", id, "mmap"),
        FileType::Fst(_) => format!("{}_{}.{}", "fst", id, "fst"),
        FileType::EulerTmp(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "eulertmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "eulertmp", id, "tmp"),
        },
        FileType::EulerPath(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "eulerpath", i, id, "mmap"),
            None => format!("{}_{}.{}", "eulerpath", id, "mmap"),
        },
        FileType::KmerTmp(_) => format!("{}_{}.{}", "kmertmpfile", id, "tmp"),
        FileType::KmerSortedTmp(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "kmersortedtmpfile", i, id, "tmp"),
            None => format!("{}_{}.{}", "kmersortedtmpfile", id, "mmap"),
        },
        FileType::KCoreBZ(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "kcorebz_tmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "kcoresbz", id, "mmap"),
        },
        FileType::KCoreLEA(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "kcorelea_tmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "kcoreslea", id, "mmap"),
        },
        FileType::KTrussBEA(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "ktrussbea_tmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "ktrussbea", id, "mmap"),
        },
        FileType::KTrussPKT(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "ktrusspkt_tmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "ktrusspkt", id, "mmap"),
        },
        FileType::ClusteringCoefficient(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "clusteringcoefficienttmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "clusteringcoefficient", id, "mmap"),
        },
        FileType::EdgeReciprocal(_) => format!("{}_{}.{}", "edge_reciprocal", id, "mmap"),
        FileType::EdgeOver(_) => format!("{}_{}.{}", "edge_over", id, "mmap"),
        FileType::HyperBall(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "hyperball", i, id, "tmp"),
            None => format!("{}_{}.{}", "hyperball", id, "mmap"),
        },
        FileType::HyperBallDistances(_) => format!("{}_{}.{}", "hypeball_distances", id, "mmap"),
        FileType::HyperBallInvDistances(_) => {
            format!("{}_{}.{}", "hyperball_inv_distances", id, "mmap")
        }
        FileType::HyperBallClosenessCentrality(_) => {
            format!("{}_{}.{}", "hyperball_closeness", id, "mmap")
        }
        FileType::HyperBallHarmonicCentrality(_) => {
            format!("{}_{}.{}", "hyperball_harmonic", id, "mmap")
        }
        FileType::HyperBallLinCentrality(_) => {
            format!("{}_{}.{}", "hyperball_inv_lin", id, "mmap")
        }
        FileType::GVELouvain(_) => match sequence_number {
            Some(i) => format!("{}_{}_{}.{}", "louvaintmp", i, id, "tmp"),
            None => format!("{}_{}.{}", "louvain", id, "mmap"),
        },
        #[cfg(test)]
        FileType::ExactClosenessCentrality(_) => {
            format!("{}_{}.{}", "exact_closeness_centarlity", id, "mmap")
        }
        #[cfg(test)]
        FileType::ExactHarmonicCentrality(_) => {
            format!("{}_{}.{}", "exact_harmonic_centarlity", id, "mmap")
        }
        #[cfg(test)]
        FileType::ExactLinCentrality(_) => {
            format!("{}_{}.{}", "exact_lin_centarlity", id, "mmap")
        }
        #[cfg(test)]
        FileType::Test(_) => {
            let random_id = rand::random::<u128>().to_string();
            format!("{}_{}.{}", "test", random_id, "tmp")
        }
    }
}

impl std::fmt::Display for FileType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            FileType::General => "Miscelanious",
            FileType::Edges(_) => "Edges",
            FileType::Index(_) => "Index",
            FileType::Fst(_) => "Fst",
            FileType::EulerPath(_) => "EulerPath",
            FileType::EulerTmp(_) => "EulerTmp",
            FileType::KmerTmp(_) => "KmerTmp",
            FileType::KmerSortedTmp(_) => "KmerSortedTmp",
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
            #[cfg(test)]
            FileType::ExactClosenessCentrality(_) => "ExactClosenessCentrality",
            #[cfg(test)]
            FileType::ExactHarmonicCentrality(_) => "ExactHarmonicCentrality",
            #[cfg(test)]
            FileType::ExactLinCentrality(_) => "ExactLinCentrality",
            #[cfg(test)]
            FileType::Test(_) => "Test",
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
