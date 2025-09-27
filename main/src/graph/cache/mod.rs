pub(crate) mod utils;
mod impl_metadata;
mod impl_sort;
#[cfg(feature = "nodes_edges")]
mod parser_nodes_edges;
#[cfg(feature = "mtx")]
mod parser_mtx;
#[cfg(feature = "ggcat")]
mod parser_ggcat;

use super::{CacheFile, GraphFile, E, N, IndexType};
use utils::{
    ensure_file_writable, cache_file_name, cache_file_name_from_id, cache_metadata_file_name_from_id, cleanup_cache, edges_to_nodes, graph_id_from_cache_file_name, id_for_subgraph_export, id_from_filename, nodes_to_edges, pers_cache_file_name, toml_cache_file_name, FileType, CACHE_DIR, H
};

#[cfg(any(test, feature = "bench"))]
use utils::EXACT_VALUE_CACHE_DIR;

use fst::{IntoStreamer, Map, MapBuilder, Streamer};
use memmap2::{Mmap, MmapOptions};
use static_assertions::const_assert;
use std::fmt::Debug;
use std::mem::ManuallyDrop;
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

#[derive(Debug, Clone)]
pub struct CacheMetadata {
    pub format_version: u16,       // bump if layout/semantics change
    pub dataset_name: String,      // e.g., "graph_0_5.lz4"
    pub cache_id: String,          // your 64-hex id
    pub nodes: usize,
    pub edges: usize,
    pub index_type: String,        // e.g., "u32" / "usize"
    pub index_size: usize,         // size_of::<Ix>()
    pub node_labeled: bool,
    pub edge_labeled: bool,
    pub has_fst: bool,
    pub node_label_type: String,   // e.g., "u32" / "usize"
    pub node_label_size: usize,    // 0 if not labeled
    pub edge_label_type: String,   // e.g., "u32" / "usize"
    pub edge_label_size: usize,    // 0 if not labeled
    pub created_unix_secs: u64,    // book keeping
    pub tool_version: Option<String>,
}

const_assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u64>());

#[allow(dead_code, clippy::upper_case_acronyms, non_camel_case_types)]
enum InputFileType {
#[cfg(feature = "ggcat")]
    GGCAT(&'static str),
#[cfg(feature = "mtx")]
    MTX(&'static str),
#[cfg(feature = "nodes_edges")]
    NODE_EDGE(&'static str),
}

pub(crate) struct GraphCache<N: super::N, E: super::E, Ix: IndexType> {
    pub neighbors_file: Arc<File>,
    pub offsets_file: Arc<File>,
    pub nodelabel_file: Arc<File>,
    pub edgelabel_file: Arc<File>,
    pub metalabel_file: Arc<File>,
    pub metadata_filename: String,
    pub neighbors_filename: String,
    pub offsets_filename: String,
    pub nodelabel_filename: String,
    pub edgelabel_filename: String,
    pub metalabel_filename: String,
    pub graph_bytes: usize,
    pub index_bytes: usize,
    pub readonly: bool,
    batch: Option<usize>,
    _marker1: PhantomData<N>,
    _marker2: PhantomData<E>,
    _marker3: PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: super::N, E: super::E, Ix: IndexType> GraphCache<N, E, Ix> {
    #[cfg(feature = "ggcat")]
    const EXT_COMPRESSED_LZ4: &str = "lz4";
    #[cfg(feature = "mtx")]
    const EXT_PLAINTEXT_MTX: &str = "mtx";
    #[cfg(feature = "ggcat")]
    const EXT_PLAINTEXT_TXT: &str = "txt";
    #[cfg(feature = "nodes_edges")]
    const EXT_PLAINTEXT_NODES: &str = "nodes";
    #[cfg(feature = "nodes_edges")]
    const EXT_PLAINTEXT_EDGES: &str = "edges";
    pub const DEFAULT_BATCHING_SIZE: usize = 50_000usize;
    /// in genetic graphs annotated with ff, fr, rf and rr directions maximum number of edges is 16
    /// |alphabet = 4| * |annotation set = 4| = 4 * 4 = 16
    const DEFAULT_MAX_EDGES: u16 = 16;
    const DEFAULT_CORRECTION_BUFFER_SIZE: usize = 1024;

    #[cfg(not(any(test, feature = "bench")))]
    fn guarantee_caching_dir() -> Result<(), Box<dyn std::error::Error>> {
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }
        Ok(())
    }

    #[cfg(any(test, feature = "bench"))]
    fn guarantee_caching_dir() -> Result<(), Box<dyn std::error::Error>> {
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }
        if !Path::new(CACHE_DIR).join(EXACT_VALUE_CACHE_DIR).exists() {
            let dir_str = CACHE_DIR.to_string() + EXACT_VALUE_CACHE_DIR;
            let dir_path = Path::new::<String>(&dir_str);
            std::fs::create_dir_all(dir_path)?;
        }
        Ok(())
    }

    pub(super) fn init_cache_file_from_id_or_random(
        graph_id: &str,
        target_type: FileType,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let id = id_from_filename(graph_id)?;

        Ok(
            match target_type {
                FileType::CacheMetadata(_) => cache_metadata_file_name_from_id(&FileType::CacheMetadata(H::H), &id, seq),
                FileType::Edges(_) => cache_file_name_from_id(&FileType::Edges(H::H), &id, seq),
                FileType::Index(H::H) => cache_file_name_from_id(&FileType::Index(H::H), &id, seq),
                FileType::NodeLabel(H::H) => {
                    cache_file_name_from_id(&FileType::NodeLabel(H::H), &id, seq)
                }
                FileType::EdgeLabel(H::H) => {
                    cache_file_name_from_id(&FileType::EdgeLabel(H::H), &id, seq)
                }
                FileType::MetaLabel(H::H) => {
                    cache_file_name_from_id(&FileType::MetaLabel(H::H), &id, seq)
                }
                FileType::Helper(H::H) => cache_file_name_from_id(&FileType::Helper(H::H), &id, seq),
                _ => {
                    return Err(format!(
                        "error unsupported file type for GraphCache: {target_type}"
                    )
                    .into());
                }
            }
        )
    }

    fn evaluate_input_file_type<P: AsRef<Path>>(
        p: P,
    ) -> Result<InputFileType, Box<dyn std::error::Error>> {
        let ext = p.as_ref().extension();

        // parse extension to decide on decoding
        if let Some(ext) = ext {
            let input_file_type = match ext.to_str() {
                #[cfg(feature = "ggcat")]
                Some(Self::EXT_COMPRESSED_LZ4) => InputFileType::GGCAT(Self::EXT_COMPRESSED_LZ4),
                #[cfg(feature = "ggcat")]
                Some(Self::EXT_PLAINTEXT_TXT) => InputFileType::GGCAT(Self::EXT_PLAINTEXT_TXT),
                #[cfg(feature = "mtx")]
                Some(Self::EXT_PLAINTEXT_MTX) => InputFileType::MTX(Self::EXT_PLAINTEXT_MTX),
                #[cfg(feature = "nodes_edges")]
                Some(Self::EXT_PLAINTEXT_NODES) => InputFileType::NODE_EDGE(Self::EXT_PLAINTEXT_NODES),
                #[cfg(feature = "nodes_edges")]
                Some(Self::EXT_PLAINTEXT_EDGES) => InputFileType::NODE_EDGE(Self::EXT_PLAINTEXT_EDGES),
                _ => {
                    return Err(format!(
                        "error ubknown extension {:?} (have you perhaps disabled any features that would allow parsing of this file type)",
                        ext,
                    )
                    .into());
                }
            };
            Ok(input_file_type)
        } else {
            Err("error input files must have an extension".into())
        }
    }

    fn read_input_file<P: AsRef<Path>>(path: P) -> Result<(Mmap, Option<String>), Box<dyn std::error::Error>> {
        let file = File::open(path.as_ref())?;
        let mut file_length = file.metadata()?.len().max(1024) * 3;
        let reader = BufReader::new(&file);
        let ext = path.as_ref().extension();

        let id = path.as_ref().file_name()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("error couldn't get file name from path {:?}", path.as_ref()).into()
            })?.to_str()
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("error couldn't get file name str from path {:?}", path.as_ref()).into()
            })?;

        // parse extension to decide on decoding
        if let Some(ext) = ext {
            Ok(match ext.to_str() {
                #[cfg(feature = "ggcat")]
                Some(Self::EXT_COMPRESSED_LZ4) => {
                    // prepare mmaped file to temporarily hold decoded contents
                    let h_fn = Self::init_cache_file_from_id_or_random(id, FileType::Helper(H::H), Some(0))?;
                    let helper_file = OpenOptions::new().create(true).truncate(true).read(true).write(true).open(&h_fn)?;
                    helper_file.set_len(file_length)?;
                    let mut mmap = unsafe { MmapOptions::new().map_mut(&helper_file)? };

                    // Decompress .lz4 using lz4 extern crate
                    let mut decoder = lz4::Decoder::new(reader)?;

                    let mut offset = 0;
                    loop {
                        if offset == file_length as usize {
                            file_length = (if mmap.is_empty() {1} else {mmap.len()} as u64).saturating_mul(2);
                            helper_file.set_len(file_length)?;
                            drop(mmap);
                            mmap = unsafe { MmapOptions::new().len(file_length as usize).map_mut(&helper_file)? };
                        }
                        let buf = mmap[offset..].as_mut();
                        if buf.is_empty() {
                            break;
                        }
                        let n = decoder.read(buf)?;
                         // check if EOF was reached
                        if n == 0 {
                            break;
                        }
                        offset += n;
                    }
                    // resize to input length
                    file_length = offset as u64;
                    helper_file.set_len(file_length)?;
                    drop(mmap);
                    mmap = unsafe { MmapOptions::new().len(file_length as usize).map_mut(&helper_file)? };

                    // ensure changes are flushed to disk
                    mmap.flush()?;

                    (mmap.make_read_only().map_err(|e| -> Box<dyn std::error::Error> {format!("error making tmp memmap reandonly: {e}").into()})?, Some(h_fn))
                }
                #[cfg(feature = "ggcat")]
                Some(Self::EXT_PLAINTEXT_TXT) => {
                    let file = OpenOptions::new().create(false).truncate(false).read(true).write(false).open(path.as_ref())?;
                    let file_length = file.metadata()?.len();
                    (unsafe { MmapOptions::new().len(file_length as usize).map(&file)? }, None)
                }
                #[cfg(feature = "nodes_edges")]
                Some(Self::EXT_PLAINTEXT_NODES) => {
                    let file = OpenOptions::new().create(false).truncate(false).read(true).write(false).open(path.as_ref())?;
                    let file_length = file.metadata()?.len();
                    (unsafe { MmapOptions::new().len(file_length as usize).map(&file)? }, None)
                }
                #[cfg(feature = "nodes_edges")]
                Some(Self::EXT_PLAINTEXT_EDGES) => {
                    let file = OpenOptions::new().create(false).truncate(false).read(true).write(false).open(path.as_ref())?;
                    let file_length = file.metadata()?.len();
                    (unsafe { MmapOptions::new().len(file_length as usize).map(&file)? }, None)
                }
                _ => {
                    return Err(format!("error ubknown file extension {:?}", ext,).into());
                }
            })
        } else {
            Err("error input files must have an extension".into())
        }
    }

    /// Initializes a [`GraphCache`] instance from a cache id[^1].
    ///
    /// [^1]: note that upon initialization, the instance is empty and must be populated with graphs nodes and edges (and optionally metalabels), checkout the [`from_ggcat_file`] or [`from_mtx_file`] methods to see an example of how to populate the [`GraphCache`] instance.
    ///
    /// # Arguments
    ///
    /// * `id` --- cache id to be used when creating cache files' filenames.
    /// * `batch_size` --- size of input chunking for fst (re)build(s)[^1][^2].
    ///
    /// [^1]: if `None` is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^2]: for more information on the functionality of input chunking, refer to [`build_fst_from_unsorted_file`]'s documentation footnote #2.
    ///
    /// [`from_ggcat_file`]: ./struct.GraphCache.html#method.from_ggcat_file
    /// [`from_mtx_file`]: ./struct.GraphCache.html#method.from_mtx_file
    /// [`build_fst_from_unsorted_file`]: ./struct.GraphCache.html#method.build_fst_from_unsorted_file
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    fn init_with_id(
        id: &str,
        batch: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;

        if id.is_empty() {
            return Err("error invalid cache id: id was `None`".into());
        }

        let metadata_filename = Self::init_cache_file_from_id_or_random(id, FileType::CacheMetadata(H::H), None)?;
        let neighbors_filename =
            Self::init_cache_file_from_id_or_random(id, FileType::Edges(H::H), None)?;
        let offsets_filename =
            Self::init_cache_file_from_id_or_random(id, FileType::Index(H::H), None)?;
        let nodelabel_filename =
            Self::init_cache_file_from_id_or_random(id, FileType::NodeLabel(H::H), None)?;
        let edgelabel_filename =
            Self::init_cache_file_from_id_or_random(id, FileType::EdgeLabel(H::H), None)?;
        let metalabel_filename =
            Self::init_cache_file_from_id_or_random(id, FileType::MetaLabel(H::H), Some(0))?;

        ensure_file_writable(&neighbors_filename)?;
        ensure_file_writable(&offsets_filename)?;
        ensure_file_writable(&nodelabel_filename)?;
        ensure_file_writable(&edgelabel_filename)?;
        ensure_file_writable(&metalabel_filename)?;

        let neighbors_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(&neighbors_filename)?,
        );
        let offsets_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(&offsets_filename)?,
        );
        let nodelabel_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(&nodelabel_filename)?,
        );
        let edgelabel_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(&edgelabel_filename)?,
        );
        let metalabel_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(&metalabel_filename)?,
        );

        Ok(Self {
            neighbors_file,
            offsets_file,
            nodelabel_file,
            edgelabel_file,
            metalabel_file,
            metadata_filename,
            neighbors_filename,
            offsets_filename,
            nodelabel_filename,
            edgelabel_filename,
            metalabel_filename,
            graph_bytes: 0,
            index_bytes: 0,
            readonly: false,
            batch: batch.map(|b| std::cmp::max(b, Self::DEFAULT_BATCHING_SIZE)),
            _marker1: PhantomData::<N>,
            _marker2: PhantomData::<E>,
            _marker3: PhantomData::<Ix>,
        })
    }

    /// Creates a [`GraphCache`] instance from an existing cached graph.
    ///
    /// # Arguments
    ///
    /// * `filename` --- filename of one of the graph's cache files[^1].
    /// * `batch_size` --- size of input chunking for fst (re)build(s)[^2][^3].
    ///
    /// [^1]: may be from any of the graph's cached files, the graph's cache id is then extracted from the filename and all necessary filename's necessary for the [`GraphCache`] instance are inferred from the cache id.
    /// [^2]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^3]: for more information on the functionality of input chunking, refer to [`build_fst_from_unsorted_file`]'s documentation footnote #2.
    ///
    /// [`build_fst_from_unsorted_file`]: ./struct.GraphCache.html#method.build_fst_from_unsorted_file
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    pub fn open<P: AsRef<Path>>(
        filename: P,
        batch: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;
        let filename = filename.as_ref().to_str().ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("error couldn't get cache entry path string for {:?}", filename.as_ref()).into()
        })?;

        let batch = Some(batch.map_or(Self::DEFAULT_BATCHING_SIZE, |b| b));
        let metadata_filename =
            Self::build_toml_graph_filename(filename, GraphFile::Metadata, None)?;
        let neighbors_filename =
            Self::build_graph_filename(filename, GraphFile::Neighbors, None)?;
        let offsets_filename =
            Self::build_graph_filename(filename, GraphFile::Offsets, None)?;
        let nodelabel_filename =
            Self::build_graph_filename(filename, GraphFile::NodeLabels, None)?;
        let edgelabel_filename =
            Self::build_graph_filename(filename, GraphFile::EdgeLabels, None)?;
        let metalabel_filename =
            Self::build_graph_filename(filename, GraphFile::MetaLabels, None)?;

        println!("reading metadata from {metadata_filename} {neighbors_filename} {filename}");
        let metadata = CacheMetadata::read_file(&metadata_filename)?;
        println!("Metadata for cache entry is: {}", metadata);

        // validate entry open request
        if N::is_labeled() {
            if !metadata.node_labeled {
                return Err(
                    format!(
                        "error cached entry for {:?} was built without unlabeled nodes (type {}), node labels can't be provided, please re-parse dataset with labeled nodes enablede",
                        filename,
                        metadata.node_label_type
                        ).into()
                    );
            } else if std::mem::size_of::<N>() != metadata.node_label_size {
                return Err(format!(
                        "error cached entry for {:?} was built without labeled nodes sized {}: node labels sized {} can't be provided, please re-parse dataset",
                        filename,
                        metadata.node_label_size,
                        std::mem::size_of::<N>()
                        ).into()
                    );
            }
        }

        if E::is_labeled() {
            if !metadata.edge_labeled {
                return Err(
                    format!(
                        "error cached entry for {:?} was built without unlabeled nodes (type {}), node labels can't be provided, please re-parse dataset with labeled nodes enablede",
                        filename,
                        metadata.edge_label_type
                        ).into()
                    );
            } else if std::mem::size_of::<E>() != metadata.edge_label_size {
                return Err(
                    format!(
                        "error cached entry for {:?} was built without labeled nodes sized {}: node labels sized {} can't be provided, please re-parse dataset",
                        filename,
                        metadata.edge_label_size,
                        std::mem::size_of::<N>()
                        ).into()
                    );
            }
        }

        let neighbors_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .open(&neighbors_filename)?,
        );

        let offsets_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .open(&offsets_filename)?,
        );

        let nodelabel_file = match OpenOptions::new()
            .read(true)
            .open(metalabel_filename.as_str())
        {
            Ok(file) => Arc::new(file),
            Err(_) => {
                if N::is_labeled() {
                    return Err(format!("error couldn't find a node labes' file in cache for {filename} but {} is labeled", crate::utils::type_of::<N>()).into());
                }
                // if graph has no nodelabel in cache no problem, build empty one and proceed
                Arc::new(
                    OpenOptions::new()
                        .create(true)
                        .truncate(false)
                        .write(true)
                        .open(&nodelabel_filename)?,
                )
            }
        };

        let edgelabel_file = match OpenOptions::new()
            .read(true)
            .open(&edgelabel_filename)
        {
            Ok(file) => Arc::new(file),
            Err(_) => {
                if E::is_labeled() {
                    return Err(format!("error couldn't find an edge labes' file in cache for {filename} but {} is labeled", crate::utils::type_of::<E>()).into());
                }
                // if graph has no edgelabel in cache no problem, build empty one and proceed
                Arc::new(
                    OpenOptions::new()
                        .create(true)
                        .truncate(false)
                        .write(true)
                        .open(&edgelabel_filename)?,
                )
            }
        };

        let metalabel_file = match OpenOptions::new()
            .read(true)
            .open(metalabel_filename.as_str())
        {
            Ok(file) => Arc::new(file),
            Err(_) => {
                // if graph has no fst in cache no problem, build empty one and proceed
                let mut empty_fst = Arc::new(
                    OpenOptions::new()
                        .create(true)
                        .truncate(true)
                        .write(true)
                        .open(&metalabel_filename)?,
                );
                let builder = MapBuilder::new(empty_fst.clone())?;
                builder.finish()?;
                // make it readonly :)
                Self::set_file_readonly(&empty_fst)?;
                empty_fst.flush()?;
                empty_fst
            }
        };

        let edges = neighbors_file.metadata().unwrap().len() as usize / std::mem::size_of::<usize>();
        let nodes = offsets_file.metadata().unwrap().len() as usize / std::mem::size_of::<usize>();

        Ok(Self {
            neighbors_file,
            offsets_file,
            nodelabel_file,
            edgelabel_file,
            metalabel_file,
            metadata_filename,
            neighbors_filename,
            offsets_filename,
            nodelabel_filename,
            edgelabel_filename,
            metalabel_filename,
            graph_bytes: edges,
            index_bytes: nodes,
            readonly: true,
            batch,
            _marker1: PhantomData::<N>,
            _marker2: PhantomData::<E>,
            _marker3: PhantomData::<Ix>,
        })
    }

    /// Parses a file input into a [`GraphCache`] instance.
    ///
    /// Input file is assumed to be either:
    ///  * [`MatrixMarket`](https://math.nist.gov/MatrixMarket/formats.html) file, in which case it must have extension .mtx and be of type matrix coordinate pattern/integer symmetric/skew-symmetric/general.
    ///  * [`GGCAT`](https://github.com/algbio/ggcat) output file, in which case it must have extension .lz4, if provided in compressed form using LZ4, or .txt if provided in plaintext, and follow [`GGCAT`](https://github.com/algbio/ggcat)'s output file format.
    ///
    /// # Arguments
    ///
    /// * `path` - input file path[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    /// * `batch` --- size of input chunking for fst rebuild[^3][^4].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^5].
    ///
    /// [^1]: for example, as a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if [`None`] is provided defaults to **NOT** storing any node's label.
    /// * `in_fst` - A function that receives a usize as input and returns a bool as output. For every node id it should return false, if its kmer is not to be included in the graph's metalabel fst or true, vice-versa.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    pub fn from_file<P: AsRef<Path>>(
        p: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;

        let mut cache = match Self::evaluate_input_file_type(p.as_ref())? {
            #[cfg(feature = "mtx")]
            InputFileType::MTX(_) => Self::from_mtx_file(p.as_ref(), id, batch),
            #[cfg(feature = "nodes_edges")]
            InputFileType::NODE_EDGE(ext) => 
            match ext {
                Self::EXT_PLAINTEXT_NODES => {
                    let edges_path = nodes_to_edges(p.as_ref())
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    "error getting edges path from nodes path".into()
                })?;
                    Self::from_node_edge_file(p.as_ref(), edges_path.as_ref(), id, batch, in_fst)
                },
                Self::EXT_PLAINTEXT_EDGES => {
                    let nodes_path = edges_to_nodes(p.as_ref())
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    "error getting nodes path from edges path".into()
                })?;
                    Self::from_node_edge_file(nodes_path.as_ref(), p.as_ref(), id, batch, in_fst)
                },
                _ => Err(format!(
                    "error ubknown extension for node/edge input file {:?}: must be of type .{} or .{}",
                    ext,
                    Self::EXT_PLAINTEXT_NODES,
                    Self::EXT_PLAINTEXT_EDGES,
                )
                .into()),
            }
            #[cfg(feature = "ggcat")]
            InputFileType::GGCAT(ext) => 
            match ext {
                Self::EXT_COMPRESSED_LZ4 => Self::from_ggcat_file(p.as_ref(), id),
                Self::EXT_PLAINTEXT_TXT => Self::from_ggcat_file(p.as_ref(), id),
                _ => Err(format!(
                    "error ubknown extension for GGCAT output file {:?}: must be of type .{} or .{}",
                    ext,
                    Self::EXT_PLAINTEXT_TXT,
                    Self::EXT_COMPRESSED_LZ4,
                )
                .into()),
            }
        }?;

        cache.finish(p.as_ref())?;
        cache.make_readonly(p.as_ref())?;
        Ok(cache)
    }

    /// Parses a [`GGCAT`](https://github.com/algbio/ggcat) output file input into a [`GraphCache`] instance.
    ///
    /// Input file is assumed to have file extension .lz4, if provided in compressed form using LZ4, or .txt, if provided in plaintext form. Furthermore, the file contents must follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `path` --- input file[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    /// * `batch`--- size of input chunking for fst rebuild[^3][^4].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^5].
    ///
    /// [^1]: for example, a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if [`None`] is provided defaults to **NOT** storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    #[cfg(feature = "nodes_edges")]
    pub fn from_node_edge_file<P: AsRef<Path>>(
        nodes_path: P,
        edges_path: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_node_edge_file_impl(nodes_path, edges_path, id, batch, in_fst)
    }

    /// Parses a [`GGCAT`](https://github.com/algbio/ggcat) output file input into a [`GraphCache`] instance.
    ///
    /// Input file is assumed to have file extension .lz4, if provided in compressed form using LZ4, or .txt, if provided in plaintext form. Furthermore, the file contents must follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `path` --- input file[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    ///
    /// [^1]: for example, a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    #[cfg(feature = "ggcat")]
    pub fn from_ggcat_file<P: AsRef<Path>>(
        path: P,
        id: Option<String>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_ggcat_file_impl(path, id)
    }

    /// Parses a [`GGCAT`](https://github.com/algbio/ggcat) output file input into a [`GraphCache`] instance, with meta labels.
    ///
    /// Input file is assumed to have file extension .lz4, if provided in compressed form using LZ4, or .txt, if provided in plaintext form. Furthermore, the file contents must follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `path` --- input file[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    /// * `batch`--- size of input chunking for fst rebuild[^3][^4].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^5].
    ///
    /// [^1]: for example, a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if [`None`] is provided defaults to **NOT** storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    #[cfg(feature = "ggcat")]
    pub fn from_ggcat_file_with_fst<P: AsRef<Path>>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_ggcat_file_with_fst_impl(path, id, batch, in_fst)
    }

    /// Parses a [`MatrixMarket`](https://math.nist.gov/MatrixMarket/formats.html) file input into a [`GraphCache`] instance.
    ///
    /// Input file is assumed have file extension .mtx and must be of type matrix coordinate pattern/integer symmetric/skew-symmetric/general.
    ///
    /// # Arguments
    ///
    /// * `path` - input file[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    /// * `batch` --- size of input chunking for fst rebuild[^3][^4].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^5].
    ///
    /// [^1]: for example, as a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if [`None`] is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    #[cfg(feature = "mtx")]
    pub fn from_mtx_file<P: AsRef<Path>>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::from_mtx_file_impl(path, id, batch)
    }

    /// Creates a [`GraphCache`] instance from an existing cached graph.
    ///
    /// # Arguments
    ///
    /// * `filename` --- filename of one of the graph's cache files[^1].
    /// * `batch` --- size of input chunking for fst rebuild[^2][^3].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^4].
    ///
    /// [^1]: may be from any of the graph's cached files, the graph's cache id is then extracted from the filename and all necessary filename's necessary for the [`GraphCache`] instance are inferred from the cache id.
    /// [^2]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^3]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^4]: if [`None`] is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    #[cfg(feature = "ggcat")]
    pub fn rebuild_fst_from_ggcat_file_file<P: AsRef<Path>>(
        &mut self,
        path: P,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        Self::rebuild_fst_from_ggcat_file_impl(self, path.as_ref(), batch, in_fst)?;
        self.finish(path.as_ref())?;
        Ok(())
    }

    fn write_node(
        &mut self,
        node_id: usize,
        data: &[usize],
        label: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected_id = self.index_bytes;
        match node_id == expected_id {
            true => {
                // write label
                writeln!(self.metalabel_file, "{}\t{}", node_id, label)?;

                self.offsets_file.write_all(bytemuck::bytes_of(&self.graph_bytes)).map_err(
                    |e| -> Box<dyn std::error::Error> {
                        format!("error writing index for {node_id}: {e}").into()
                    }
                )?;
                self.index_bytes += 1;

                self.neighbors_file.write_all(bytemuck::cast_slice(data)).map_err(
                    |e| -> Box<dyn std::error::Error> {
                        format!("error writing edges for {node_id}: {e}").into()
                    }
                )?;
                self.graph_bytes += data.len();

                Ok(())
            }
            false => Err(
                format!("error nodes must be mem mapped in ascending order, (id: {node_id}, expected_id: {expected_id})"
                ).into()),
        }
    }

    fn write_unlabeled_node(
        &mut self,
        node_id: usize,
        data: &[usize],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected_id = self.index_bytes;
        match node_id == expected_id {
            true => {
                self.offsets_file.write_all(bytemuck::bytes_of(&self.graph_bytes)).map_err(
                    |e| -> Box<dyn std::error::Error> {
                        format!("error writing index for {node_id}: {e}").into()
                    }
                )?;
                self.index_bytes += 1;


                self.neighbors_file.write_all(bytemuck::cast_slice(data)).map_err(
                    |e| -> Box<dyn std::error::Error> {
                        format!("error writing edges for {node_id}: {e}").into()
                    }
                )?;
                self.graph_bytes += data.len();
                Ok(())
            }
            false => Err(format!(
                    "error nodes must be mem mapped in ascending order, (id: {node_id}, expected_id: {expected_id})"
                ).into()),
        }
    }

    fn external_sort_by_content(temp: &str, sorted: &str) -> std::io::Result<()> {
        // sort based on 2nd column (content), not line number
        Command::new("sort")
            .args(["-k2", temp, "-o", sorted])
            .status()?;
        Ok(())
    }

    fn build_fst_from_sorted_file(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        // Build finite state tranducer for k-mer to node id
        let fst_filename =
            cache_file_name(&self.metalabel_filename, &FileType::MetaLabel(H::H), None)?;
        let sorted_file =
            cache_file_name(&self.metalabel_filename, &FileType::MetaLabel(H::H), Some(1))?;

        Self::external_sort_by_content(&self.metalabel_filename, &sorted_file)?;

        self.metalabel_file = match File::create(&fst_filename) {
            Ok(i) => {
                self.metalabel_filename = fst_filename;
                Arc::new(i)
            }
            Err(e) => {
                return Err(format!("error couldn't create metalabel fst file: {e}").into());
            }
        };

        let sorted_file = OpenOptions::new()
            .read(true)
            .create(false)
            .open(sorted_file)?;
        // As reandonly is false, no clones exist -> safe to take file ownership from Arc
        let mut build =
            MapBuilder::new(&*self.metalabel_file).map_err(|e| -> Box<dyn std::error::Error> {
                format!("error couldn't initialize builder: {e}").into()
            })?;

        let mut reader = BufReader::new(sorted_file);
        let mut line = Vec::new();

        loop {
            if reader
                .read_until(b'\n', &mut line)
                .map_err(|e| -> Box<dyn std::error::Error> {
                    format!("error reading file: {e}").into()
                })?
                == 0
            {
                break;
            }

            if let Ok(text) = std::str::from_utf8(&line) {
                let mut parts = text.trim_end().split('\t');
                if let (Some(id_value), Some(metalabel)) = (parts.next(), parts.next()) {
                    let id = id_value.parse::<u64>()?;
                    build.insert(metalabel, id).map_err(|e| -> Box<dyn std::error::Error> {
                        format!(
                            "error couldn't insert metalabel for node (id: {id_value} metalabel: {metalabel}): {e}"
                            ).into()
                    })?;
                }
            }
            line.clear();
        }

        build.finish().map_err(|e| -> Box<dyn std::error::Error> {
            format!("error couldn't finish fst build: {e}").into()
        })
    }

    /// Builds an fst for the [`GraphCache`] instance from an unsorted entries file[^1].
    ///
    /// [^1]: entries should be formatted ad "{node_id}\t{node_label}\n".
    ///
    /// # Arguments
    ///
    /// * `batch_size` --- size of input chunking[^2].
    ///
    /// [^2]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub fn build_fst_from_unsorted_file(
        &mut self,
        batch_size: usize, // e.g. 10_000
    ) -> Result<(), Box<dyn std::error::Error>> {
        let input = OpenOptions::new()
            .read(true)
            .create(false)
            .open(&self.metalabel_filename)?;
        let mut contents = vec![];
        let mut reader = BufReader::new(input);
        reader.read_to_end(&mut contents)?;

        let mut batch_num: usize = 0usize;
        let mut batches = Vec::new();
        let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);

        for line in contents.split(|b| *b == b'\n') {
            let mut parts = line.split(|c| *c == b'\t');
            if let (Some(node_id), Some(label)) = (parts.next(), parts.next()) {
                let id = std::str::from_utf8(node_id)?.parse::<u64>()?;

                current_batch.push((label, id));
                if current_batch.len() >= batch_size {
                    let tmp_fst = self.build_batch_fst(&mut current_batch, batch_num)?;
                    batches.push(tmp_fst);
                    current_batch.clear();
                    batch_num += 1;
                }
            }
        }

        // Process the last batch if not empty
        if !current_batch.is_empty() {
            let tmp_fst = self.build_batch_fst(&mut current_batch, batch_num)?;
            batches.push(tmp_fst);
        }

        // Now merge all batch FSTs into one
        self.merge_fsts(&batches)?;

        // Cleanup temp batch files
        for batch_file in batches {
            std::fs::remove_file(batch_file)?;
        }

        Ok(())
    }

    /// Builds an FST for a single sorted batch and writes it to a temp file.
    fn build_batch_fst_vec(
        &self,
        batch: &mut [(Vec<u8>, u64)],
        batch_num: usize,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        // Sort lexicographically by key (label)
        batch.sort_by(|a, b| a.0.cmp(&b.0));
        let tempfst_fn = cache_file_name(
            &self.metalabel_filename,
            &FileType::MetaLabel(H::H),
            Some(batch_num + 1),
        )?;

        let mut wtr = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tempfst_fn)?;
        let mut builder = MapBuilder::new(&mut wtr)?;

        for (k, v) in batch.iter() {
            builder.insert(k, *v)?;
        }
        builder.finish()?;
        Ok(PathBuf::from(tempfst_fn))
    }

    /// Builds an FST for a single sorted batch and writes it to a temp file.
    fn build_batch_fst(
        &self,
        batch: &mut [(&[u8], u64)],
        batch_num: usize,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        // Sort lexicographically by key (label)
        batch.sort_by(|a, b| a.0.cmp(b.0));
        let tempfst_fn = cache_file_name(
            &self.metalabel_filename,
            &FileType::MetaLabel(H::H),
            Some(batch_num + 1),
        )?;

        let mut wtr = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&tempfst_fn)?;
        let mut builder = MapBuilder::new(&mut wtr)?;

        for (k, v) in batch.iter() {
            builder.insert(k, *v)?;
        }
        builder.finish()?;
        Ok(PathBuf::from(tempfst_fn))
    }

    /// Merge multiple FST batch files into a final FST.
    fn merge_fsts(&mut self, batch_paths: &[PathBuf]) -> Result<(), Box<dyn std::error::Error>> {
        // open output FST file for writing
        let out_fn = cache_file_name(&self.metalabel_filename, &FileType::MetaLabel(H::H), None)?;
        ensure_file_writable(&out_fn)?;
        let out = Arc::new(
            OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&out_fn)?,
        );
        let mut builder = MapBuilder::new(out.clone())?;

        // if empty an empty fst is built and stored
        if !batch_paths.is_empty() {
            // Open all batch maps
            let maps: Result<Vec<_>, _> = batch_paths
                .iter()
                .map(|p| {
                    let mmap = unsafe {
                        MmapOptions::new()
                            .map(&OpenOptions::new().read(true).create(false).open(p)?)?
                    };
                    Map::new(mmap)
                })
                .collect();
            let maps = maps?;

            let mut op_builder = maps[0].op();
            if maps.len() > 1 {
                for map in &maps[1..] {
                    op_builder = op_builder.add(map.stream());
                }
            }

            // perform a union (merges sorted keys across all maps)
            let union_stream = op_builder.union();

            // iterate through merged keys and select values
            let mut stream = union_stream.into_stream();
            while let Some((key, vals)) = stream.next() {
                // each `vals` is a Vec<IndexedValue>, one per input map
                if let Some(val) = vals.to_vec().first() {
                    builder.insert(key, val.value)?;
                }
            }
        }

        builder.finish()?;
        self.metalabel_file = out;
        self.metalabel_filename = out_fn;

        Ok(())
    }

    fn build_metadata_simple<P: AsRef<Path>>(
        &self,
        dataset: P,
        cache_id: &str,
        has_fst: bool,
        tool_version: Option<String>,
    ) -> Result<CacheMetadata, Box<dyn std::error::Error>> {
        let edges = self.neighbors_file.metadata()?.len() as usize / size_of::<usize>();
        let nodes = self.offsets_file.metadata()?.len() as usize / size_of::<usize>();
        let dataset_str = dataset.as_ref().to_str().ok_or_else(|| -> Box<dyn std::error::Error> {
            format!("error getting dataset str from {:?}", dataset.as_ref()).into()
        })?;

        Ok(CacheMetadata::now::<N, E, Ix>(
            dataset_str,
            cache_id,
            nodes,
            edges,
            size_of::<Ix>(),
            N::is_labeled(),
            E::is_labeled(),
            has_fst,
            if N::is_labeled() { size_of::<N>() } else { 0 },
            if E::is_labeled() { size_of::<E>() } else { 0 },
            tool_version,
        ))
    }

    fn set_file_readonly(file: &File)  -> Result<(), Box<dyn std::error::Error>> {
        let mut permissions = file.metadata()?.permissions();
        permissions.set_readonly(true);
        Ok(file.set_permissions(permissions)?)
    }

    fn finish<P: AsRef<Path>>(&mut self, dataset: P) -> Result<(), Box<dyn std::error::Error>> {
        // make all files read-only and cleanup
        for file in [
            &mut self.offsets_file,
            &mut self.neighbors_file,
            &mut self.edgelabel_file,
            &mut self.nodelabel_file,
            &mut self.metalabel_file,
        ] {
            Self::set_file_readonly(file)?;
            // flush needed because in multithreaded accesses wihtout it memory is in undefined state
            file.flush()?;
        }

    // Write metadata (text, no external deps)
    let has_fst = self.metalabel_file.metadata()?.len() > 0;

    let meta = self.build_metadata_simple(
        dataset.as_ref(),                 // store these when creating the cache
        &self.cache_id()?,
        has_fst,
        Some(env!("CARGO_PKG_VERSION").to_string()),
    )?;

    meta.write_file(self.metadata_filename())?;

    // Make metadata read-only as well
    let mut mf = OpenOptions::new().read(true).open(self.metadata_filename())?;
    Self::set_file_readonly(&mf)?;
    mf.flush()?;
        // remove any tmp files that may have been used to (re)build `GraphCache` instance
        self.cleanup_cache_by_target(FileType::MetaLabel(H::H))?;
        self.readonly = true;
        Ok(())
    }

    /// Make the [`GraphCache`] instance readonly. Allows the user to [`Clone`] the struct.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub fn make_readonly<P: AsRef<Path>>(&mut self, p: P) -> Result<(), Box<dyn std::error::Error>> {
        if self.readonly {
            return Ok(());
        }

        // complete index file if not set to readonly
        if !self.offsets_file.metadata()?.permissions().readonly() {
            self.offsets_file
                .write_all(bytemuck::bytes_of(&self.graph_bytes))
                .map_err(|e| -> Box<dyn std::error::Error> {
                    format!("error couldn't finish index: {e}").into()
                })?;
            self.index_bytes += 1;
        }

        if let Some(batch_size) = self.batch {
            self.build_fst_from_unsorted_file(batch_size).map_err(
                |e| -> Box<dyn std::error::Error> {
                    format!("error couldn't build fst from unsorted file in bacthes of {batch_size}: {e}")
                        .into()
                },
            )?;
        } else {
            self.build_fst_from_sorted_file()?;
        }

        self.finish(p.as_ref())?;
        self.make_readonly(p.as_ref())
    }

    /// Returns the neighbors' file's filename.
    #[inline]
    pub(super) fn metadata(&self) -> Result<CacheMetadata, Box<dyn std::error::Error>> {
        CacheMetadata::read_file(&self.metadata_filename)
    }

    /// Returns the neighbors' file's filename.
    #[inline]
    pub fn metadata_filename(&self) -> String {
        self.metadata_filename.clone()
    }

    /// Returns the neighbors' file's filename.
    #[inline]
    pub fn neighbors_filename(&self) -> String {
        self.neighbors_filename.clone()
    }

    /// Returns the offsets' file's filename.
    #[inline]
    pub fn offsets_filename(&self) -> String {
        self.offsets_filename.clone()
    }

    /// Returns the nodelabels' file's filename.
    #[inline]
    pub fn nodelabels_filename(&self) -> String {
        self.nodelabel_filename.clone()
    }

    /// Returns the edgelabels' file's filename.
    #[inline]
    pub fn edgelabels_filename(&self) -> String {
        self.edgelabel_filename.clone()
    }

    /// Returns the fst (metalabel-to-node map) file's filename.
    #[inline]
    pub fn fst_filename(&self) -> String {
        self.metalabel_filename.clone()
    }

    /// Returns the graph's cache id.
    #[inline]
    pub fn cache_id(&self) -> Result<String, Box<dyn std::error::Error>> {
        graph_id_from_cache_file_name(self.offsets_filename.clone())
    }

    /// Returns the graph's cache id.
    #[inline]
    pub(super) fn build_subgraph_cache_id(&self, sequence: usize) -> Result<String, Box<dyn std::error::Error>> {
        Ok(id_for_subgraph_export(self.cache_id()?, Some(sequence)))
    }

    fn convert_cache_file(file_type: CacheFile) -> FileType {
        match file_type {
            CacheFile::General => FileType::General,
            CacheFile::BFS => FileType::BFS(H::H),
            CacheFile::DFS => FileType::DFS(H::H),
            CacheFile::EulerIndex => FileType::EulerIndex(H::H),
            CacheFile::EulerTrail => FileType::EulerTrail(H::H),
            CacheFile::KCoreBZ => FileType::KCoreBZ(H::H),
            CacheFile::KCoreLEA => FileType::KCoreLEA(H::H),
            CacheFile::Triangles => FileType::Triangles(H::H),
            CacheFile::KTrussBEA => FileType::KTrussBEA(H::H),
            CacheFile::KTrussPKT => FileType::KTrussPKT(H::H),
            CacheFile::ClusteringCoefficient => FileType::ClusteringCoefficient(H::H),
            CacheFile::EdgeReciprocal => FileType::EdgeReciprocal(H::H),
            CacheFile::EdgeOver => FileType::EdgeOver(H::H),
            CacheFile::HyperBall => FileType::HyperBall(H::H),
            CacheFile::HyperBallDistances => FileType::HyperBallDistances(H::H),
            CacheFile::HyperBallInvDistances => FileType::HyperBallInvDistances(H::H),
            CacheFile::HyperBallClosenessCentrality => FileType::HyperBallClosenessCentrality(H::H),
            CacheFile::HyperBallHarmonicCentrality => FileType::HyperBallHarmonicCentrality(H::H),
            CacheFile::HyperBallLinCentrality => FileType::HyperBallLinCentrality(H::H),
            CacheFile::GVELouvain => FileType::GVELouvain(H::H),
            CacheFile::ExactCloseness => FileType::ExactClosenessCentrality(H::H),
            CacheFile::ExactHarmonic => FileType::ExactHarmonicCentrality(H::H),
            CacheFile::ExactLin => FileType::ExactLinCentrality(H::H),
        }
    }

    fn convert_graph_file(file_type: GraphFile) -> FileType {
        match file_type {
            GraphFile::Metadata => FileType::CacheMetadata(H::H),
            GraphFile::Neighbors => FileType::Edges(H::H),
            GraphFile::Offsets => FileType::Index(H::H),
            GraphFile::NodeLabels => FileType::NodeLabel(H::H),
            GraphFile::EdgeLabels => FileType::EdgeLabel(H::H),
            GraphFile::MetaLabels => FileType::MetaLabel(H::H),
        }
    }

    /// Build a cached (`.mmap`) file of a given [`CacheFile`] type for the [`GraphCache`] instance.
    ///
    /// [`CacheFile`]: ./enum.CacheFile.html#
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub(super) fn build_pers_cache_filename(
        &self,
        target: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        pers_cache_file_name(&self.neighbors_filename, &Self::convert_cache_file(target), seq)
    }

    /// Build a cached (either `.mmap` or `.tmp`) file of a given [`CacheFile`] type for the [`GraphCache`] instance.
    ///
    /// [`CacheFile`]: ./enum.CacheFile.html#
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub(super) fn build_cache_filename(
        &self,
        target: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        cache_file_name(&self.neighbors_filename, &Self::convert_cache_file(target), seq)
    }

    /// Build a cached (`.toml`) file of a given [`GraphFile`] type for the [`GraphCache`] instance.
    ///
    /// [`CacheFile`]: ./enum.CacheFile.html#
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub(super) fn build_toml_graph_filename(
        fname: &str,
        target: GraphFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        toml_cache_file_name(fname, &Self::convert_graph_file(target), seq)
    }

    /// Build a cached (either `.mmap` or `.tmp`) file of a given [`GaraphFile`] type for the [`GraphCache`] instance.
    ///
    /// [`GraphFile`]: ./enum.CacheFile.html#
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub(super) fn build_graph_filename(
        fname: &str,
        target: GraphFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        cache_file_name(fname, &Self::convert_graph_file(target), seq)
    }

    /// Build a cached `.tmp` file of type [`FileType`]::Helper(_) for the [`GraphCache`] instance.
    ///
    /// [`FileType`]: ../utils/enum.FileType.html#
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub(super) fn build_helper_filename(
        &self,
        seq: usize,
    ) -> Result<String, Box<dyn std::error::Error>> {
        cache_file_name(&self.neighbors_filename, &FileType::Helper(H::H), Some(seq))
    }

    /// Remove [`GraphCache`] instance's cached `.tmp` files of type [`FileType`]::Helper(_).
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`FileType`]: ../utils/enum.FileType.html#
    pub(super) fn cleanup_helpers(&self) -> Result<(), Box<dyn std::error::Error>> {
        cleanup_cache(&self.cache_id()?, &FileType::Helper(H::H))
    }

    /// Remove [`GraphCache`] instance's cached `.tmp` files for a given [`FileType`].
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`FileType`]: ../utils/enum.FileType.html#
    pub(super) fn cleanup_cache_by_target(
        &self,
        target: FileType,
    ) -> Result<(), Box<dyn std::error::Error>> {
        cleanup_cache(&self.cache_id()?, &target)
    }

    /// Remove [`GraphCache`] instance's cached `.tmp` files for a given [`CacheFile`] in the cache directory.
    ///
    /// [`CacheFile`]: ./enum.CacheFile.html#
    pub(super) fn cleanup_cache(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>> {
        cleanup_cache(&self.cache_id()?, &Self::convert_cache_file(target))
    }

    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        std::fs::remove_file(this.metadata_filename())?;
        std::fs::remove_file(this.neighbors_filename())?;
        std::fs::remove_file(this.offsets_filename())?;
        std::fs::remove_file(this.nodelabels_filename())?;
        std::fs::remove_file(this.edgelabels_filename())?;
        std::fs::remove_file(this.fst_filename())?;
        Ok(())
    }
}


impl<N: super::N, E: super::E, Ix: IndexType> Clone for GraphCache<N, E, Ix> {
    fn clone(&self) -> Self {
        if !self.readonly {
            panic!(
                "can't clone GraphCache before it is readonly, fst needs file ownership to map metalabels to node_ids"
            );
        }
        Self {
            neighbors_file: self.neighbors_file.clone(),
            offsets_file: self.offsets_file.clone(),
            nodelabel_file: self.nodelabel_file.clone(),
            edgelabel_file: self.edgelabel_file.clone(),
            metalabel_file: self.metalabel_file.clone(),
            metadata_filename: self.metadata_filename.clone(),
            neighbors_filename: self.neighbors_filename.clone(),
            offsets_filename: self.offsets_filename.clone(),
            nodelabel_filename: self.nodelabel_filename.clone(),
            edgelabel_filename: self.edgelabel_filename.clone(),
            metalabel_filename: self.metalabel_filename.clone(),
            graph_bytes: self.graph_bytes,
            index_bytes: self.index_bytes,
            readonly: true,
            batch: self.batch,
            _marker1: self._marker1,
            _marker2: self._marker2,
            _marker3: self._marker3,
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Debug
    for GraphCache<N, E, Ix>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n\tneighbours filename: {}\n\toffsets filename: {}\n\tnodelabels filename: {}\n\tedgelabels filename: {}\n\tmetalabel filename: {}\n}}",
            self.neighbors_filename(),
            self.offsets_filename(),
            self.nodelabels_filename(),
            self.edgelabels_filename(),
            self.fst_filename()
        )
    }
}

type MultithreadedParserIndexBounds = Box<[(usize, usize)]>;
