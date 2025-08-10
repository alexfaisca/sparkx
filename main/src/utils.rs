use glob::glob;
use regex::Regex;
use std::{
    any::type_name,
    io::Error,
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

    let file_name = match path.file_name() {
        Some(i) => match i.to_str() {
            Some(i) => i,
            None => {
                return Err(Box::new(Error::new(
                    std::io::ErrorKind::Other,
                    "error invalid file path --- empty path",
                )));
            }
        },
        None => {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error invalid file path --- path not found",
            )));
        }
    };

    // extract id from filename
    let re = match Regex::new(r#"^(?:[a-zA-Z0-9_]+_)(\w+)(\.[a-zA-Z0-9]+$)"#).ok() {
        Some(i) => i,
        None => {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error analyzing file name (regex failed)",
            )));
        }
    };
    let caps = match re.captures(file_name) {
        Some(i) => i,
        None => {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error capturing filnr name (regex captures failed: fail point 1)",
            )));
        }
    };
    let id = match caps.get(1) {
        Some(i) => i.as_str(),
        None => {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error capturing filne name (regex capture failed: fail point 2)",
            )));
        }
    };

    Ok(id.to_string())
}

#[allow(dead_code)]
fn graph_id_and_dir_from_cache_file_name(
    cache_filename: String,
) -> Result<(String, PathBuf), Box<dyn std::error::Error>> {
    let path = Path::new(cache_filename.as_str());

    let file_name = match path.file_name() {
        Some(i) => match i.to_str() {
            Some(i) => i,
            None => {
                return Err(Box::new(Error::new(
                    std::io::ErrorKind::Other,
                    "error invalid file path --- empty path",
                )));
            }
        },
        None => {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error invalid file path --- path not found",
            )));
        }
    };

    let parent_dir = path.parent().unwrap_or_else(|| Path::new(""));

    // extract id from filename
    let re = match Regex::new(r#"^(?:[a-zA-Z0-9_]+_)(\w+)(\.[a-zA-Z0-9]+$)"#).ok() {
        Some(i) => i,
        None => {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error analyzing file name (regex failed)",
            )));
        }
    };
    let caps = match re.captures(file_name) {
        Some(i) => i,
        None => {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error capturing filnr name (regex captures failed: fail point 1)",
            )));
        }
    };
    let id = match caps.get(1) {
        Some(i) => i.as_str(),
        None => {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error capturing filne name (regex capture failed: fail point 2)",
            )));
        }
    };

    Ok((id.to_string(), PathBuf::from(parent_dir)))
}

#[allow(dead_code)]
pub fn cache_file_name(
    original_filename: String,
    target_type: FileType,
    sequence_number: Option<usize>,
) -> Result<String, Box<dyn std::error::Error>> {
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
pub fn cleanup_cache() -> Result<(), Error> {
    match glob((CACHE_DIR.to_string() + "/*.tmp").as_str()) {
        Ok(entries) => {
            for entry in entries {
                match entry {
                    Ok(path) => std::fs::remove_file(path)?,
                    Err(e) => panic!("{}", e),
                };
            }
            Ok(())
        }
        Err(e) => Err(Error::new(
            std::io::ErrorKind::Other,
            format!("error cleaning up cache: {e}"),
        )),
    }
}

#[derive(Debug)]
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
            Some(i) => format!("{}_{}_{}.{}", "eulertmp", i, id, "mmap"),
            None => format!("{}_{}.{}", "eulertmp", id, "mmap"),
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
        };
        write!(f, "{}", s)
    }
}
