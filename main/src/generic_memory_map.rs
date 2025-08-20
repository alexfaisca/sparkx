use crate::generic_edge::{GenericEdge, GenericEdgeType};
use crate::shared_slice::{
    AbstractedProceduralMemory, AbstractedProceduralMemoryMut, SharedSlice, SharedSliceMut,
};
use crate::utils::{
    CACHE_DIR, FileType, TEMP_CACHE_DIR, cache_file_name, cleanup_cache,
    graph_id_from_cache_file_name, id_for_subgraph_export,
};

use crossbeam::thread;
use fst::{IntoStreamer, Map, MapBuilder, Streamer};
use memmap2::{Mmap, MmapOptions};
use ordered_float::OrderedFloat;
use rustworkx_core::petgraph::graph::{DiGraph, NodeIndex};
use static_assertions::const_assert;
use std::io::{Seek, SeekFrom};
use std::{
    collections::HashSet,
    fmt::Debug,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    marker::PhantomData,
    path::{Path, PathBuf},
    process::Command,
    slice,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
};
use zerocopy::*;

const_assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u64>());

pub struct GraphCache<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    pub graph_file: Arc<File>,
    pub index_file: Arc<File>,
    pub kmer_file: Arc<File>,
    pub graph_filename: String,
    pub index_filename: String,
    pub kmer_filename: String,
    pub graph_bytes: usize,
    pub index_bytes: usize,
    pub readonly: bool,
    batch: Option<usize>,
    _marker1: PhantomData<Edge>,
    _marker2: PhantomData<EdgeType>,
}

#[allow(dead_code)]
impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> GraphCache<EdgeType, Edge> {
    const EXT_COMPRESSED_LZ4: &str = "lz4";
    const EXT_MTX: &str = "mtx";
    const EXT_PLAINTEXT: &str = "txt";
    const DEFAULT_BATCHING_SIZE: usize = 50_000usize;
    /// in genetic graphs annotated with ff, fr, rf and rr directions maximum number of edges is 16
    /// |alphabet = 4| * |annotation set = 4| = 4 * 4 = 16
    const DEFAULT_MAX_EDGES: u16 = 16;
    const DEFAULT_CORRECTION_BUFFER_SIZE: usize = 1024;

    pub(self) fn init_cache_file_from_id_or_random(
        graph_id: Option<String>,
        target_type: FileType,
    ) -> Result<(String, String), Box<dyn std::error::Error>> {
        let id = match graph_id {
            Some(i) => i,
            None => rand::random::<u64>().to_string(),
        };
        Ok((
            match target_type {
                FileType::Edges => format!("{}{}_{}.{}", CACHE_DIR, "edges", id, "mmap"),
                FileType::Index => format!("{}{}_{}.{}", CACHE_DIR, "index", id, "mmap"),
                FileType::Fst => format!("{}{}_{}.{}", CACHE_DIR, "fst", id, "fst"),
                FileType::KmerTmp => format!("{}{}_{}.{}", CACHE_DIR, "kmertmpfile", id, "tmp"),
                FileType::KmerSortedTmp => {
                    format!("{}{}_{}.{}", CACHE_DIR, "kmersortedtmpfile", id, "tmp")
                }
                _ => {
                    return Err(format!(
                        "error unsupported file type for GraphCache: {target_type}"
                    )
                    .into());
                }
            },
            id,
        ))
    }

    fn read_input_file<P: AsRef<Path> + Clone>(
        path: P,
    ) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
        let mut reader = BufReader::new(File::open(path.clone())?);
        let ext = path.as_ref().extension();
        let mut contents = Vec::new();

        // parse extension to decide on decoding
        if let Some(ext) = ext {
            match ext.to_str() {
                Some(Self::EXT_COMPRESSED_LZ4) => {
                    // Decompress .lz4 using lz4 extern crate
                    let mut decoder = lz4::Decoder::new(reader)?;
                    decoder.read_to_end(&mut contents)?;
                }
                Some(Self::EXT_PLAINTEXT) => {
                    reader.read_to_end(&mut contents)?;
                }
                _ => {
                    return Err(format!(
                        "error ubknown extension {:?} (must be of type .{} or .{})",
                        ext,
                        Self::EXT_PLAINTEXT,
                        Self::EXT_COMPRESSED_LZ4
                    )
                    .into());
                }
            };
            Ok(contents)
        } else {
            Err("error input files must have an extension".into())
        }
    }

    /// Parses a ggcat output edge direction
    ///
    /// # Arguments
    ///
    /// * `orig` - The edge's origin endpoint annotation. Must be either '+' or '-'.
    /// * `destiny` - The edge's destiny endpoint annotation. Must be either '+' or '-'.
    ///
    /// # Returns
    ///
    /// A `usize` that encodes the edge direction:
    /// * `0` --- for a forward-forward edge.
    /// * `1` --- for a forward-reverse edge.
    /// * `2` --- for a reverse-forward edge.
    /// * `3` --- for a reverse-reverse edge.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * An unknown annotation is given as input.
    ///
    /// # Examples
    ///
    /// ```
    /// // encoding for forward-forward edge
    /// let ff_edge = Self::parse_ggcat_direction("+", "+");
    /// ```
    /// ```
    /// // encoding for forward-reverse edge
    /// let fr_edge = Self::parse_ggcat_direction("+", "-");
    /// ```
    /// ```
    /// // encoding for reverse-forward edge
    /// let rf_edge = Self::parse_ggcat_direction("-", "+");
    /// ```
    /// ```
    /// // encoding for reverse-reverse edge
    /// let rr_edge = Self::parse_ggcat_direction(".", "-");
    /// ```
    ///
    fn parse_ggcat_direction(orig: &str, dest: &str) -> Result<u64, Box<dyn std::error::Error>> {
        match (orig, dest) {
            ("+", "+") => Ok(0u64),
            ("+", "-") => Ok(1u64),
            ("-", "+") => Ok(2u64),
            ("-", "-") => Ok(3u64),
            _ => Err(format!("error ubknown edge direction annotations (supported are '+' && '-'): (orig: {orig}, dest: {dest})").into()),
        }
    }

    /// Parses a ggcat output file input into a [`GraphCache`] instance.
    ///
    /// Input is assumed to be of type UTF-8 and to follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `input` - Input bytes.
    /// * `max_edges` - Ascribes a maximum number of edges for a node. If None is provided, defaults to 16.
    /// * `in_fst` - A function that receives a usize as input and returns a bool as output. For every node id it should return false, if its kmer / label is not to be included in the graph's fst or true, vice-versa.
    ///
    /// # Returns
    ///
    /// Empty Ok().
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * In the input slice no k-mer / label sequence is found for some node.
    /// * An error occurs parsing a line of the input from UTF-8.
    /// * An error occurs parsing a node's identifier.
    /// * An error occurs parsing an edge's destiny node identifier or direction annototation.
    /// * An error occurs writing the node into the memmapped cache files.
    ///
    /// # Examples
    ///
    /// ```
    /// // for every node record at most 16 (default maximum value) edges and always add its label
    /// // to the grapph's finite state transducer
    /// let () = mut_graph_cache.parse_ggcat_bytes_mmap(input_bytes, None, |_id: usize| {true})?;
    /// ```
    /// ```
    /// // for every node record at most 8 edges and add its label to the grapph's finite state
    /// // transducer if the node id is 123432 or 912383 or it its id is divisible by 7
    /// let () = mut_graph_cache.parse_ggcat_bytes_mmap(
    ///     input_bytes,
    ///     Some(8),
    ///     |id: usize| { if id == 123432 || id == 912383 || id % 7 == 0 { true } else { false }}
    /// )?;
    /// ```
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    fn parse_ggcat_bytes_mmap(
        &mut self,
        input: &[u8],
        in_fst: fn(usize) -> bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // this assumes UTF-8 but avoids full conversion
        let mut lines = input.split(|&b| b == b'\n');
        let mut edges = vec![];
        while let Some(line) = lines.next() {
            if line.is_empty() {
                continue;
            }
            // convert each line to str temporarily && cut off ">" char
            let line_str = std::str::from_utf8(&line[1..])?;

            let sequence_line = match lines.next() {
                None => {
                    return Err(format!("error no k-mer sequence for node {line_str}").into());
                }
                Some(i) => i,
            };
            // convert each line to str temporarily -> cut off ">" char
            let line_str = std::str::from_utf8(&line[1..line.len()])?;
            let node = line_str.split_whitespace().collect::<Vec<&str>>();
            let mut node = node.iter().peekable();

            let id: usize = node.next().unwrap().parse()?;
            let _node_lengh = node.next(); // length 
            let _node_color = node.next(); // color value
            for link in node {
                let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                edges.push(Edge::new(
                    link_slice[1].parse()?,
                    Self::parse_ggcat_direction(link_slice[0], link_slice[2])?,
                ));
            }
            edges.sort_unstable_by_key(|e| e.dest());

            if in_fst(id) {
                self.write_node(
                    id,
                    edges.as_slice(),
                    std::str::from_utf8(&sequence_line[0..])?,
                )?;
            } else {
                self.write_unlabeled_node(id, edges.as_slice())?;
            }
            edges.clear();
        }

        Ok(())
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
    /// [^1]: if `None` is provided defaults to 10'000.
    /// [^2]: for more information on the functionality of input chunking, refer to [`build_fst_from_unsorted_file`]'s documentation footnote #2.
    ///
    /// [`from_ggcat_file`]: ./struct.GraphCache.html#method.from_ggcat_file
    /// [`from_mtx_file`]: ./struct.GraphCache.html#method.from_mtx_file
    /// [`build_fst_from_unsorted_file`]: ./struct.GraphCache.html#method.build_fst_from_unsorted_file
    /// [`GraphCache`]: ./struct.GraphCache.html#
    fn init_with_id(
        id: String,
        batch: Option<usize>,
    ) -> Result<GraphCache<EdgeType, Edge>, Box<dyn std::error::Error>> {
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }
        if id.is_empty() {
            return Err("error invalid cache id: id was `None`".into());
        }

        let (graph_filename, id) =
            Self::init_cache_file_from_id_or_random(Some(id), FileType::Edges)?;
        let (index_filename, id) =
            Self::init_cache_file_from_id_or_random(Some(id), FileType::Index)?;
        let (kmer_filename, _) =
            Self::init_cache_file_from_id_or_random(Some(id), FileType::KmerTmp)?;

        let graph_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(graph_filename.as_str())?,
        );

        let index_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(index_filename.as_str())?,
        );

        let kmer_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .write(true)
                .truncate(true)
                .create(true)
                .open(kmer_filename.as_str())?,
        );

        Ok(GraphCache::<EdgeType, Edge> {
            graph_file,
            index_file,
            kmer_file,
            graph_filename,
            index_filename,
            kmer_filename,
            graph_bytes: 0,
            index_bytes: 0,
            readonly: false,
            batch: if let Some(n) = batch {
                Some(std::cmp::max(n, Self::DEFAULT_BATCHING_SIZE))
            } else {
                batch
            },
            _marker1: PhantomData::<Edge>,
            _marker2: PhantomData::<EdgeType>,
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
    /// [^2]: if `None` is provided defaults to 10'000.
    /// [^3]: for more information on the functionality of input chunking, refer to [`build_fst_from_unsorted_file`]'s documentation footnote #2.
    ///
    /// [`build_fst_from_unsorted_file`]: ./struct.GraphCache.html#method.build_fst_from_unsorted_file
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub fn open(
        filename: String,
        batch: Option<usize>,
    ) -> Result<GraphCache<EdgeType, Edge>, Box<dyn std::error::Error>> {
        let batch = Some(batch.map_or(Self::DEFAULT_BATCHING_SIZE, |b| b));
        let graph_filename = cache_file_name(filename.clone(), FileType::Edges, None)?;
        let index_filename = cache_file_name(filename.clone(), FileType::Index, None)?;
        let kmer_filename = cache_file_name(filename.clone(), FileType::Fst, None)?;

        let graph_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .open(graph_filename.as_str())?,
        );
        let index_file = Arc::new(
            OpenOptions::new()
                .read(true)
                .open(index_filename.as_str())?,
        );
        let kmer_file = match OpenOptions::new().read(true).open(kmer_filename.as_str()) {
            Ok(file) => Arc::new(file),
            Err(_) => {
                // if graph has no fst in cache no problem, build empty one and proceed
                let empty_fst = Arc::new(
                    OpenOptions::new()
                        .create(true)
                        .truncate(true)
                        .write(true)
                        .open(&kmer_filename)?,
                );
                let builder = MapBuilder::new(empty_fst.clone())?;
                builder.finish()?;
                // make it readonly :)
                let mut permissions = empty_fst.metadata()?.permissions();
                permissions.set_readonly(true);
                empty_fst.set_permissions(permissions)?;
                empty_fst
            }
        };

        let edges = graph_file.metadata().unwrap().len() as usize / std::mem::size_of::<usize>();
        let nodes = index_file.metadata().unwrap().len() as usize / std::mem::size_of::<Edge>();

        Ok(GraphCache::<EdgeType, Edge> {
            graph_file,
            index_file,
            kmer_file,
            graph_filename,
            index_filename,
            kmer_filename,
            graph_bytes: edges,
            index_bytes: nodes,
            readonly: true,
            batch,
            _marker1: PhantomData::<Edge>,
            _marker2: PhantomData::<EdgeType>,
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
    /// [^1]: for example, as a `String`.
    /// [^2]: if `None` is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if `None` is provided defaults to 10'000.
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if `None` is provided defaults to **NOT** storing any node's label.
    /// * `in_fst` - A function that receives a usize as input and returns a bool as output. For every node id it should return false, if its kmer / label is not to be included in the graph's fst or true, vice-versa.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub fn from_file<P: AsRef<Path> + Clone>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<GraphCache<EdgeType, Edge>, Box<dyn std::error::Error>> {
        // make sure cache directory exists, if not attempt to create it
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }

        // make sure tmp directory exists, if not attempt to create it
        if !Path::new(TEMP_CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }

        // parse optional inputs && fallback to defaults for the Nones found
        // parse extension to decide on decoding
        let ext = path.as_ref().extension();
        if let Some(ext) = ext {
            match ext.to_str() {
                Some(Self::EXT_COMPRESSED_LZ4) => {
                    Ok(Self::from_ggcat_file(path, id, batch, in_fst)?)
                }
                Some(Self::EXT_PLAINTEXT) => Ok(Self::from_ggcat_file(path, id, batch, in_fst)?),
                Some(Self::EXT_MTX) => Ok(Self::from_mtx_file(path, id, batch)?),
                _ => Err(format!(
                    "error ubknown extension {:?}: must be of type .{}, .{} .{}",
                    ext,
                    Self::EXT_PLAINTEXT,
                    Self::EXT_MTX,
                    Self::EXT_COMPRESSED_LZ4
                )
                .into()),
            }
        } else {
            Err("error input files must have an extension".into())
        }
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
    /// [^1]: for example, a `String`.
    /// [^2]: if `None` is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if `None` is provided defaults to 10'000.
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if `None` is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub fn from_ggcat_file<P: AsRef<Path> + Clone>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<GraphCache<EdgeType, Edge>, Box<dyn std::error::Error>> {
        // make sure cache directory exists, if not attempt to create it
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }

        // make sure tmp directory exists, if not attempt to create it
        if !Path::new(TEMP_CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }

        // parse optional inputs && fallback to defaults for the Nones found
        let id = id.map_or(rand::random::<u64>().to_string(), |id| id);
        let batching = Some(batch.map_or(Self::DEFAULT_BATCHING_SIZE, |b| b));
        let in_fst = match in_fst {
            Some(f) => f,
            None => |_id: usize| -> bool { false },
        };
        let input = Self::read_input_file(path)?;

        // init cache
        let mut cache = Self::init_with_id(id, batching)?;

        // parse and cache input
        cache.parse_ggcat_bytes_mmap(input.as_slice(), in_fst)?;

        // make cache readonly (for now only serves to allow clone() on instances)
        cache.make_readonly()?;

        Ok(cache)
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
    /// [^1]: for example, as a `String`.
    /// [^2]: if `None` is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if `None` is provided defaults to 10'000.
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if `None` is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub fn from_mtx_file<P: AsRef<Path> + Clone>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
    ) -> Result<GraphCache<EdgeType, Edge>, Box<dyn std::error::Error>> {
        if std::mem::size_of::<Edge>() % std::mem::size_of::<usize>() != 0 {
            return Err(format!("Error type `{}` has a  size of {}, which will lead to unaligned memory read/write for graph edges. Currently we don't support this type of memory access for graphs from .matx files. :(", std::any::type_name::<Edge>(), std::mem::size_of::<Edge>()).into());
        }

        // make sure cache directory exists, if not attempt to create it
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }

        // make sure tmp directory exists, if not attempt to create it
        if !Path::new(TEMP_CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }
        let id = id.map_or(rand::random::<u64>().to_string(), |id| id);

        let (nr, _nc, _nnz_declared) = Self::parse_mtx_header(path.clone())?;

        let (edge_fn, id) = Self::init_cache_file_from_id_or_random(Some(id), FileType::Edges)?;
        let (index_fn, _id) = Self::init_cache_file_from_id_or_random(Some(id), FileType::Index)?;
        let offset_fn = cache_file_name(edge_fn.clone(), FileType::EulerTmp, Some(20))?;
        let mut index = SharedSliceMut::<usize>::abst_mem_mut(index_fn, nr + 1, true)?;
        let mut index_offset = SharedSliceMut::<usize>::abst_mem_mut(offset_fn, nr, true)?;

        // accumulate node degrees on index
        let mut edge_count = 0;
        {
            Self::parse_mtx_with(path.clone(), |u, _v, _w| {
                edge_count += 1;
                *index.get_mut(u) += 1;
            })?;
        }
        let mut edges = SharedSliceMut::<Edge>::abst_mem_mut(edge_fn.clone(), edge_count, true)?;
        // build offset vector from degrees
        let mut sum = 0;
        let mut max_degree = 0;
        // this works beacause after aloc memmaped files are zeroed (so index[nr] = 0)
        for u in 0..=nr {
            let deg_u = *index.get(u);
            if deg_u > max_degree {
                max_degree = deg_u;
            }
            *index.get_mut(u) = sum;
            sum += deg_u;
        }

        if max_degree >= u8::MAX as usize {
            return Err(format!("Error graph has a max_degree of {max_degree} which, unforturnately, is bigger than {}, our current maximum supported size. If you feel a mistake has been made or really need this feature, please contact the developer team. We sincerely apologize.", u8::MAX).into());
        }

        // write edges
        let mut wrote = 0;
        {
            Self::parse_mtx_with(path.clone(), |u, v, w| {
                wrote += 1;
                *edges.get_mut(*index.get(u) + *index_offset.get(u)) = Edge::new(v, w);
                *index_offset.get_mut(u) += 1;
            })?;
        }

        edges.flush()?;
        index.flush()?;
        Self::open(edge_fn.clone(), batch)
    }

    fn parse_mtx_header<P: AsRef<Path>>(path: P) -> Result<MTXHeader, Box<dyn std::error::Error>> {
        let f = File::open(path)?;
        let mut rdr = BufReader::new(f);

        // read and parse the header line
        let mut header = String::new();
        rdr.read_line(&mut header)?;
        if !header.starts_with("%%MatrixMarket") {
            return Err("Invalid MatrixMarket header".into());
        }
        // tokens: 0=%%MatrixMarket, 1=matrix, 2=coordinate, 3=field, 4=symmetry
        let toks: Vec<_> = header.split_whitespace().collect();
        if toks.len() < 5 {
            return Err("Malformed MatrixMarket header".into());
        }
        if !toks[1].eq_ignore_ascii_case("matrix")
            || !toks[2].eq_ignore_ascii_case("coordinate")
            || !(toks[3].eq_ignore_ascii_case("pattern") || toks[3].eq_ignore_ascii_case("integer"))
            || !(toks[4].eq_ignore_ascii_case("symmetric")
                || toks[4].eq_ignore_ascii_case("skew-symmetric")
                || toks[4].eq_ignore_ascii_case("general"))
        {
            return Err(
                "Only 'matrix coordinate pattern/integer symmetric' format is supported".into(),
            );
        }

        // skip comment lines (%) to find the size line: "nrows ncols nnz"
        let (nrows, ncols, nnz_declared) = {
            let mut line = String::new();
            loop {
                line.clear();
                let n = rdr.read_line(&mut line)?;
                if n == 0 {
                    return Err("Unexpected EOF before reading size line".into());
                }
                // ignore comment lines
                let line_trim = line.trim();
                if line_trim.is_empty() || line_trim.starts_with('%') {
                    continue;
                }
                // parse sizes
                let mut it = line_trim.split_whitespace();
                let nr: usize = it.next().ok_or("Missing nrows")?.parse()?;
                let nc: usize = it.next().ok_or("Missing ncols")?.parse()?;
                let nnz: usize = it.next().ok_or("Missing nnz")?.parse()?;
                if nr != nc {
                    return Err(format!(
                        "Only symmetric matrices are supported but {{nr = {nr}}} != {{nc = {nc}}}"
                    )
                    .into());
                }
                break (nr, nc, nnz);
            }
        };
        Ok((nrows, ncols, nnz_declared))
    }

    fn parse_mtx_with<F, P: AsRef<Path>>(
        path: P,
        mut emit: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, u64, u64),
    {
        let f = File::open(path)?;
        let mut rdr = BufReader::new(f);

        // read and parse the header line
        let mut header = String::new();
        rdr.read_line(&mut header)?;
        if !header.starts_with("%%MatrixMarket") {
            return Err("Invalid MatrixMarket header".into());
        }
        // tokens: 0=%%MatrixMarket, 1=matrix, 2=coordinate, 3=field, 4=symmetry
        let toks: Vec<_> = header.split_whitespace().collect();
        if toks.len() < 5 {
            return Err("Malformed MatrixMarket header".into());
        }
        if !toks[1].eq_ignore_ascii_case("matrix")
            || !toks[2].eq_ignore_ascii_case("coordinate")
            || !(toks[3].eq_ignore_ascii_case("pattern") || toks[3].eq_ignore_ascii_case("integer"))
            || !(toks[4].eq_ignore_ascii_case("symmetric")
                || toks[4].eq_ignore_ascii_case("skew-symmetric")
                || toks[4].eq_ignore_ascii_case("general"))
        {
            return Err(
                "Only 'matrix coordinate pattern/integer symmetric' format is supported".into(),
            );
        }
        let pattern = toks[3].eq_ignore_ascii_case("pattern");
        let symmetric = !toks[4].eq_ignore_ascii_case("general");

        // skip comment lines (%) to find the size line: "nrows ncols nnz"
        let (nrows, ncols, nnz_declared) = {
            let mut line = String::new();
            loop {
                line.clear();
                let n = rdr.read_line(&mut line)?;
                if n == 0 {
                    return Err("Unexpected EOF before reading size line".into());
                }
                // ignore comment lines
                let line_trim = line.trim();
                if line_trim.is_empty() || line_trim.starts_with('%') {
                    continue;
                }
                // parse sizes
                let mut it = line_trim.split_whitespace();
                let nr: usize = it.next().ok_or("Missing nrows")?.parse()?;
                let nc: usize = it.next().ok_or("Missing ncols")?.parse()?;
                let nnz: usize = it.next().ok_or("Missing nnz")?.parse()?;
                if nr != nc {
                    return Err(format!(
                        "Only symmetric matrices are supported but {{nr = {nr}}} != {{nc = {nc}}}"
                    )
                    .into());
                }
                break (nr, nc, nnz);
            }
        };

        // stream the nnz entries
        // For coordinate:
        // - pattern:          i j
        // - integer/real:     i j val
        // - complex/hermitian i j real imag  (we’ll ignore values and just read i j)
        //
        // MatrixMarket is 1-based; convert to 0-based.
        // We’ll be forgiving: we keep reading non-comment non-empty lines until we see nnz_declared entries.
        let mut seen = 0usize;
        let mut line = String::new();
        while seen < nnz_declared {
            line.clear();
            let n = rdr.read_line(&mut line)?;
            if n == 0 {
                // Some files under-report nnz in the header; we stop if EOF reached
                break;
            }
            let t = line.trim();
            if t.is_empty() || t.starts_with('%') {
                continue;
            }
            let mut it = t.split_whitespace();

            // 1st two tokens must be i j (1-based)
            let i1: usize = match it.next() {
                Some(s) => s.parse()?,
                None => continue, // malformed line; skip
            };
            let j1: u64 = match it.next() {
                Some(s) => s.parse()?,
                None => continue, // malformed line; skip
            };
            // Convert to 0-based, also ensure in-bounds
            if i1 == 0 || j1 == 0 {
                return Err("MatrixMarket indices are 1-based; found 0".into());
            }
            let i = i1 - 1;
            let j = j1 - 1;

            let w: u64 = if pattern {
                0
            } else {
                match it.next() {
                    Some(s) => s.parse()?,
                    None => 0, // malformed line; skip
                }
            };
            if i >= nrows || j as usize >= ncols {
                return Err(
                    format!("Entry out of bounds: ({i},{j}) not in [{nrows}x{ncols}]").into(),
                );
            }

            // Emit edges:
            // one directed edge i --(w)--> j
            emit(i, j, w);
            if symmetric && i != j as usize {
                emit(j as usize, i as u64, w);
            }

            seen += 1;
        }

        Ok(())
    }

    #[inline(always)]
    fn process_ggcat_entry(
        id_line: String,
        label_line: Vec<u8>,
        in_fst: fn(usize) -> bool,
    ) -> Result<ProcessGGCATEntry, Box<dyn std::error::Error>> {
        let node = id_line.split_whitespace().collect::<Vec<&str>>();
        let id: usize = node.first().unwrap().parse()?;

        if in_fst(id) {
            Ok(Some((label_line.clone(), id as u64)))
        } else {
            Ok(None)
        }
    }

    fn process_chunk_with_reader<R: BufRead>(
        &mut self,
        reader: &mut R,
        end: usize,
        batch_num: Arc<AtomicUsize>,
        batch_size: usize,
        in_fst: fn(usize) -> bool,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut current_batch: Vec<(Vec<u8>, u64)> = Vec::with_capacity(batch_size);
        let mut batches = Vec::new();

        let mut id_line: String = String::new();
        let mut label_line: Vec<u8> = Vec::new();

        reader.read_until(b'>', &mut label_line)?;
        label_line.clear();

        // first iteration
        let r_i = reader.read_line(&mut id_line)?;
        label_line.clear();
        let r_l = reader.read_until(b'\n', &mut label_line)?;
        if r_i == 0 || r_l == 0 {
            return Err(format!("error empty label for node {id_line}").into());
        }
        let mut read = r_i + r_l;

        let mut node_str = &id_line.clone()[..];
        // continue until end of strem or read >= end
        loop {
            // convert each line to str temporarily && cut off ">" char
            let _ = label_line.pop(); // pop '\n'

            if let Some(entry) =
                Self::process_ggcat_entry(node_str.to_string(), label_line.clone(), in_fst)?
            {
                current_batch.push(entry);
                if current_batch.len() >= batch_size {
                    println!("wrote {}", batch_num.load(Ordering::Relaxed));
                    let tmp_fst = self.build_batch_fst_vec(
                        &mut current_batch,
                        batch_num.fetch_add(1, Ordering::Relaxed),
                    )?;
                    batches.push(tmp_fst);
                    current_batch.clear();
                }
            }

            id_line.clear();
            let r_i = reader.read_line(&mut id_line)?;
            node_str = &id_line[1..];
            label_line.clear();
            let r_l = reader.read_until(b'\n', &mut label_line)?;
            if r_i == 0 || r_l == 0 {
                return Err(format!("error empty label for node {id_line}").into());
            }

            read += r_i;
            read += r_l;

            if read >= end {
                break;
            }
        }

        // Process the last batch if not empty
        if !current_batch.is_empty() {
            let tmp_fst = self.build_batch_fst_vec(
                &mut current_batch,
                batch_num.fetch_add(1, Ordering::Relaxed),
            )?;
            batches.push(tmp_fst);
        }

        Ok(batches)
    }

    fn parallel_fst_from_ggcat_with_reader<P: AsRef<Path> + Clone>(
        &mut self,
        file_path: P,
        batch_size: usize,
        in_fst: fn(usize) -> bool,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.readonly {
            return Err("error cache must be readonly to build fst in parallel".into());
        }
        let file = File::open(file_path.clone())?;
        let file_len = file.metadata()?.len(); // Get the size of the file
        let ext = file_path.as_ref().extension();
        if ext.is_none() {
            return Err("error file extension was None".into());
        }
        let ext = ext.unwrap().to_str();

        let chunk_size = file_len / threads as u64;
        let batch_num = Arc::new(AtomicUsize::new(0));

        let batches = thread::scope(|s| -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
            let mut batch_handles = Vec::new();

            for i in 0..threads {
                let mut cache = self.clone();
                let batch_num = Arc::clone(&batch_num);

                let path = file_path.as_ref();
                let r = match ext {
                    Some(Self::EXT_COMPRESSED_LZ4) => 1,
                    Some(Self::EXT_PLAINTEXT) => 2,
                    _ => 0,
                };

                let start = i as u64 * chunk_size;
                let end = std::cmp::min(
                    start + chunk_size + Self::DEFAULT_CORRECTION_BUFFER_SIZE as u64,
                    file_len,
                );

                let handle = s.spawn(
                    move |_| -> Result<Vec<PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
                        let mut file = File::open(path)?; // Open the file separately in each thread
                        let mut reader = BufReader::new(&mut file);
                        reader.seek(SeekFrom::Start(start))?;
                        let mut input_chunk = reader.take(end - start);

                        let res = match r {
                            1 => {
                                let decoder = lz4::Decoder::new(input_chunk)?;
                                let mut reader = BufReader::new(decoder);
                                cache.process_chunk_with_reader(
                                    &mut reader,
                                    end as usize,
                                    batch_num,
                                    batch_size,
                                    in_fst,
                                )
                            }
                            2 => cache.process_chunk_with_reader(
                                &mut input_chunk,
                                end as usize,
                                batch_num,
                                batch_size,
                                in_fst,
                            ),
                            _ => {
                                return Err(
                                    format!("error unknown file extension {:?}", ext).into()
                                );
                            }
                        };

                        match res {
                            Ok(i) => Ok(i),
                            Err(e) => Err(format!("error occured for thread {i}: {e}").into()),
                        }
                    },
                );

                batch_handles.push(handle);
            }

            let mut all_batches = Vec::new();
            for handle in batch_handles {
                match handle.join() {
                    Ok(Ok(batches)) => {
                        all_batches.extend(batches);
                    }
                    Ok(Err(e)) => {
                        return Err(e);
                    }
                    Err(e) => {
                        return Err(format!("Error in thread: {:?}", e).into());
                    }
                }
            }
            Ok(all_batches)
        })
        .unwrap()?;

        // Now merge all batch FSTs into one option
        self.merge_fsts(&batches)?;

        // Clean up temporary batch files
        for batch_file in batches {
            let _ = std::fs::remove_file(batch_file);
        }

        Ok(())
    }

    fn process_chunk(
        &mut self,
        input: &[u8],
        end: usize,
        batch_num: Arc<AtomicUsize>,
        batch_size: usize,
        in_fst: fn(usize) -> bool,
    ) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
        let mut begin_pos = 0;
        let mut end_pos = 0;

        let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);
        let mut batches = Vec::new();

        // find beginning of next node entry (marked by '>')
        while begin_pos < input.len() && input[begin_pos] != b'>' {
            begin_pos += 1;
        }

        // find beginning of next node entry after end of slice (marked by '\n>')
        while end + end_pos + 1 < input.len() && input[end + end_pos + 1] != b'>' {
            end_pos += 1;
        }

        // truncate input
        let input = &input[begin_pos..=std::cmp::min(input.len() - 1, end + end_pos)];

        let mut lines = input.split(|&b| b == b'\n');

        while let Some(line) = lines.next() {
            if line.is_empty() {
                continue;
            }

            // convert each line to str temporarily && cut off ">" char
            let line_str = std::str::from_utf8(&line[1..])?;
            let label_line = match lines.next() {
                None => {
                    return Err(format!("error no label for node {line_str}").into());
                }
                Some(i) => i,
            };

            let line_str = std::str::from_utf8(&line[1..line.len()])?;
            let node = line_str.split_whitespace().collect::<Vec<&str>>();
            let id: usize = node.first().unwrap().parse()?;

            if in_fst(id) {
                current_batch.push((label_line, id as u64));

                if current_batch.len() >= batch_size {
                    println!("wrote {}", batch_num.load(Ordering::Relaxed));
                    let tmp_fst = self.build_batch_fst(
                        &mut current_batch,
                        batch_num.fetch_add(1, Ordering::Relaxed),
                    )?;
                    batches.push(tmp_fst);
                    current_batch.clear();
                }
            }
        }

        // Process the last batch if not empty
        if !current_batch.is_empty() {
            let tmp_fst = self.build_batch_fst(
                &mut current_batch,
                batch_num.fetch_add(1, Ordering::Relaxed),
            )?;
            batches.push(tmp_fst);
        }

        Ok(batches)
    }

    fn parallel_fst_from_ggcat_bytes(
        &mut self,
        input: &[u8],
        batch_size: usize,
        in_fst: fn(usize) -> bool,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.readonly {
            return Err("error cache must be readonly to build fst in parallel".into());
        }

        let input_size = input.len();
        let chunk_size = input_size / threads;
        let batch_num = Arc::new(AtomicUsize::new(0));

        let shared_slice = SharedSlice::from_slice(input);

        let batches = thread::scope(|s| -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
            let mut batches = Vec::new();
            for i in 0..threads {
                let mut cache = self.clone();

                let batch_num = Arc::clone(&batch_num);

                let start = std::cmp::min(i * chunk_size, input_size);
                let end = std::cmp::min(
                    start + chunk_size + Self::DEFAULT_CORRECTION_BUFFER_SIZE,
                    input_size,
                );

                batches.push(s.spawn(
                    move |_| -> Result<Vec<PathBuf>, Box<dyn std::error::Error + Send + Sync>> {
                        let input = match shared_slice.slice(start, end) {
                            Some(slice) => slice,
                            None => {
                                return Err("error occured while chunking input in slices".into());
                            }
                        };

                        match cache.process_chunk(input, chunk_size, batch_num, batch_size, in_fst)
                        {
                            Ok(b) => Ok(b),
                            Err(e) => Err(format!(
                                "error occured while batching fst build in parallel: {e}"
                            )
                            .into()),
                        }
                    },
                ));
            }

            let mut b = Vec::new();
            for batch in batches {
                match batch.join() {
                    Ok(i) => match i {
                        Ok(paths) => {
                            for path in paths {
                                b.push(path);
                            }
                        }
                        Err(e) => return Err(format!("error is thread {e}").into()),
                    },
                    Err(e) => {
                        return Err(format!("error is thread {:?}", e).into());
                    }
                }
            }
            Ok(b)
        })
        .unwrap()?;

        // if graph cache was used to build a graph then the graph's fst
        //  holds kmer_filename open, hence, it must be removed so that the new fst may
        //  be built
        std::fs::remove_file(self.kmer_filename.clone())?;

        // Now merge all batch FSTs into option
        self.merge_fsts(&batches)?;

        // Cleanup temp batch files
        for batch_file in batches {
            let _ = std::fs::remove_file(batch_file);
        }

        Ok(())
    }

    /// Parses a [`GGCAT`](https://github.com/algbio/ggcat) output file given as input and builds an fst for the graph according to the input.
    /// parameters.
    ///
    /// Input is assumed to be of type UTF-8 and to follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `input` - Input bytes.
    /// * `in_fst` - A function that receives a usize as input and returns a bool as output. For every node id it should return false, if its kmer / label is not to be included in the graph's fst or true, vice-versa.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * In the input slice no k-mer / label sequence is found for some node.
    /// * An error occurs parsing a line of the input from UTF-8.
    /// * An error occurs parsing a node's identifier.
    /// * An error occurs parsing an edge's destiny node identifier or direction annototation.
    /// * An error occurs writing the node into the memmapped cache files.
    ///
    /// # Examples
    ///
    /// ```
    /// // for every node record at most 16 (default maximum value) edges and always add its label
    /// // to the grapph's finite state transducer
    /// let () = mut_graph_cache.parse_ggcat_bytes_mmap(input_bytes, None, |_id: usize| {true})?;
    /// ```
    /// ```
    /// // for every node record at most 8 edges and add its label to the grapph's finite state
    /// // transducer if the node id is 123432 or 912383 or it its id is divisible by 7
    /// let () = mut_graph_cache.parse_ggcat_bytes_mmap(
    ///     input_bytes,
    ///     Some(8),
    ///     |id: usize| { if id == 123432 || id == 912383 || id % 7 == 0 { true } else { false }}
    /// )?;
    /// ```
    ///
    ///
    #[deprecated]
    fn fst_from_ggcat_bytes(
        &mut self,
        input: &[u8],
        batch_size: usize,
        in_fst: fn(usize) -> bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // this assumes UTF-8 but avoids full conversion

        let mut batch_num: usize = 0usize;

        let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);
        let mut batches = Vec::new();

        let mut lines = input.split(|&b| b == b'\n');

        while let Some(line) = lines.next() {
            if line.is_empty() {
                continue;
            }
            // convert each line to str temporarily && cut off ">" char
            let line_str = std::str::from_utf8(&line[1..])?;

            let sequence_line = match lines.next() {
                None => {
                    return Err(format!("error no k-mer sequence for node {line_str}").into());
                }
                Some(i) => i,
            };

            let line = std::str::from_utf8(&line[1..])?;
            let node = line.split_whitespace().collect::<Vec<&str>>();
            let id: usize = node.first().unwrap().parse()?;

            if in_fst(id) {
                current_batch.push((sequence_line, id as u64));
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

        // if graph cache was used to build a graph then the graph's fst
        // holds kmer_filename open, hence, it must be removed so that the new fst may
        // be built
        std::fs::remove_file(self.kmer_filename.clone())?;
        // Now merge all batch FSTs into one
        self.merge_fsts(&batches)?;

        // Cleanup temp batch files
        for batch_file in batches {
            let _ = std::fs::remove_file(batch_file);
        }

        Ok(())
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
    /// [^2]: if `None` is provided defaults to 10'000.
    /// [^3]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^4]: if `None` is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub fn rebuild_fst_from_ggcat_file<P: AsRef<Path> + Clone>(
        &mut self,
        path: P,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batching = batch.map_or(Self::DEFAULT_BATCHING_SIZE, |b| b);
        let in_fst = match in_fst {
            Some(f) => f,
            None => |_id: usize| -> bool { true },
        };

        // self.parallel_fst_from_ggcat_with_reader(path, batching, in_fst, get_physical())?;
        let input = Self::read_input_file(path)?;
        // FIXME: Out Of Memory
        self.parallel_fst_from_ggcat_bytes(
            input.as_slice(),
            batching,
            in_fst,
            num_cpus::get_physical(),
        )?;
        // self.fst_from_ggcat_bytes(input.as_slice(), batching, in_fst)?;
        //
        Ok(())
    }

    fn write_node(
        &mut self,
        node_id: usize,
        data: &[Edge],
        label: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected_id = self.index_bytes;
        match node_id == expected_id {
            true => {
                // write label
                writeln!(self.kmer_file, "{}\t{}", node_id, label)?;

                match self.index_file.write_all(self.graph_bytes.as_bytes()) {
                    Ok(_) => self.index_bytes += 1,
                    Err(e) => {
                        return Err(format!("error writing index for {node_id}: {e}").into());
                    }
                };

                match self.graph_file.write_all(bytemuck::cast_slice(data)) {
                    Ok(_) => {
                        self.graph_bytes += data.len();
                        Ok(())
                    }
                    Err(e) => Err(format!("error writing edges for {node_id}: {e}").into()),
                }
            }
            false => Err(
                format!("error nodes must be mem mapped in ascending order, (id: {node_id}, expected_id: {expected_id})"
                ).into()),
        }
    }

    fn write_unlabeled_node(
        &mut self,
        node_id: usize,
        data: &[Edge],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let expected_id = self.index_bytes;
        match node_id == expected_id {
            true => {
                match self.index_file.write_all(self.graph_bytes.as_bytes()) {
                    Ok(_) => self.index_bytes += 1,
                    Err(e) => {
                        return Err(format!("error writing index for {node_id}: {e}").into());
                    }
                };

                match self.graph_file.write_all(bytemuck::cast_slice(data)) {
                    Ok(_) => {
                        self.graph_bytes += data.len();
                        Ok(())
                    }
                    Err(e) => Err(format!("error writing edges for {node_id}: {e}").into()),
                }
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
        let fst_filename = cache_file_name(self.kmer_filename.clone(), FileType::Fst, None)?;
        let sorted_file =
            cache_file_name(self.kmer_filename.clone(), FileType::KmerSortedTmp, None)?;

        Self::external_sort_by_content(self.kmer_filename.as_str(), sorted_file.as_str())?;

        self.kmer_file = match File::create(fst_filename.clone()) {
            Ok(i) => {
                self.kmer_filename = fst_filename;
                Arc::new(i)
            }
            Err(e) => {
                return Err(format!("error couldn't create mer .fst file: {e}").into());
            }
        };

        let sorted_file = OpenOptions::new()
            .read(true)
            .create(false)
            .open(sorted_file)?;
        // As reandonly is false, no clones exist -> safe to take file ownership from Arc
        let mut build = match MapBuilder::new(&*self.kmer_file) {
            Ok(i) => i,
            Err(e) => {
                return Err(format!("error couldn't initialize builder: {e}").into());
            }
        };

        let mut reader = BufReader::new(sorted_file);
        let mut line = Vec::new();

        loop {
            let () = match reader.read_until(b'\n', &mut line) {
                Ok(i) => {
                    if i == 0 {
                        break;
                    }
                }
                Err(e) => {
                    return Err(format!("error reading file: {e}").into());
                }
            };
            if let Ok(text) = std::str::from_utf8(&line) {
                let mut parts = text.trim_end().split('\t');
                if let (Some(id_value), Some(kmer)) = (parts.next(), parts.next()) {
                    let id = id_value.parse::<u64>()?;
                    match build.insert(kmer, id) {
                        Ok(i) => i,
                        Err(e) => {
                            return Err(format!(
                                    "error couldn't insert kmer for node (id: {id_value} label: {kmer}): {e}"
                                ).into());
                        }
                    };
                }
            }
            line.clear();
        }

        match build.finish() {
            Ok(_) => Ok(()),
            Err(e) => Err(format!("error couldn't finish fst build: {e}").into()),
        }
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
            .open(&self.kmer_filename)?;
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
            let _ = std::fs::remove_file(batch_file);
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
            self.kmer_filename.clone(),
            FileType::KmerSortedTmp,
            Some(batch_num),
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
            self.kmer_filename.clone(),
            FileType::KmerSortedTmp,
            Some(batch_num),
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
        let out_fn = cache_file_name(self.kmer_filename.clone(), FileType::Fst, None)?;
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
        self.kmer_file = out;
        self.kmer_filename = out_fn;
        Ok(())
    }

    /// Make the [`GraphCache`] instance readonly. Allows the user to `Clone` the struct.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    pub fn make_readonly(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.readonly {
            return Ok(());
        }

        // complete index file
        match self.index_file.write_all(self.graph_bytes.as_bytes()) {
            Ok(_) => self.index_bytes += 1,
            Err(e) => {
                return Err(format!("error couldn't finish index: {e}").into());
            }
        };

        if let Some(batch_size) = self.batch {
            match self.build_fst_from_unsorted_file(batch_size) {
                Ok(_) => {}
                Err(e) => {
                    return Err(format!(
                        "error couldn't build fst from unsorted file in bacthes of 10'000: {e}"
                    )
                    .into());
                }
            };
        } else {
            self.build_fst_from_sorted_file()?;
        }
        // make all files read-only and cleanup
        for file in [&self.index_file, &self.graph_file, &self.kmer_file] {
            let mut permissions = file.metadata()?.permissions();
            permissions.set_readonly(true);
            file.set_permissions(permissions)?;
        }
        self.readonly = true;
        // cleanup_cache()
        Ok(())
    }

    /// Returns the edge file's filename.
    #[inline]
    pub fn edges_filename(&self) -> String {
        self.graph_filename.clone()
    }

    /// Returns the offsets file's filename.
    #[inline]
    pub fn index_filename(&self) -> String {
        self.graph_filename.clone()
    }

    /// Returns the fst (metalabel-to-node map) file's filename.
    #[inline]
    pub fn fst_filename(&self) -> String {
        self.graph_filename.clone()
    }

    /// Returns the graph's cache id.
    #[inline]
    pub fn cache_id(&self) -> Result<String, Box<dyn std::error::Error>> {
        graph_id_from_cache_file_name(self.graph_filename.clone())
    }

    /// Remove all of cache directory's `.tmp` files.
    pub fn cleanup_cache(&self) -> Result<(), Box<dyn std::error::Error>> {
        cleanup_cache()
    }
}

#[derive(Clone)]
pub struct GraphMemoryMap<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: Arc<Mmap>,
    index: Arc<Mmap>,
    kmers: Arc<Map<Mmap>>,
    graph_cache: GraphCache<EdgeType, Edge>,
    edge_size: usize,
    thread_count: u8,
    exports: u8,
}

#[allow(dead_code)]
impl<EdgeType, Edge> GraphMemoryMap<EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    /// Initializes a [`GraphMemoryMap`] instance from a [`GraphCache`] instance[^1][^2].
    ///
    /// [^1]: [`GraphCache`] instance must be reandonly, dynamic graphs are not supported.
    /// [^2]: despite being readonly, the [`GraphCache`] instance's fst may be rebuilt *a posteriori* if proven necessary, checkout [`GraphCache`]'s documentation for more information on how to perform an fst rebuild.
    ///
    /// # Arguments
    ///
    /// * `cache` --- [`GraphCache`] instance to be used.
    /// * `thread_count`--- user suggested number of threads to be used when computing algorithms on the graph.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn init(
        cache: GraphCache<EdgeType, Edge>,
        thread_count: u8,
    ) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
        if cache.readonly {
            let mmap = unsafe { MmapOptions::new().map(&File::open(&cache.kmer_filename)?)? };
            let thread_count = thread_count.max(1);
            return Ok(GraphMemoryMap {
                graph: unsafe { Arc::new(Mmap::map(&cache.graph_file)?) },
                index: unsafe { Arc::new(Mmap::map(&cache.index_file)?) },
                kmers: Arc::new(Map::new(mmap)?),
                graph_cache: cache,
                edge_size: std::mem::size_of::<Edge>(),
                thread_count,
                exports: 0,
            });
        }

        Err("error graph cache must be readonly to be memmapped".into())
    }

    #[inline(always)]
    pub(crate) fn index_ptr(&self) -> *const usize {
        self.index.as_ptr() as *const usize
    }

    #[inline(always)]
    pub(crate) fn edges_ptr(&self) -> *const Edge {
        self.graph.as_ptr() as *const Edge
    }

    /// Returns the (suggested) number of threads being used for computations on the graph.
    #[inline(always)]
    pub fn thread_num(&self) -> usize {
        self.thread_count.max(1) as usize
    }

    /// Returns the graph's cache id.
    #[inline(always)]
    pub fn graph_id(&self) -> Result<String, Box<dyn std::error::Error>> {
        self.graph_cache.cache_id()
    }

    /// Returns the graph's edge file's filename.
    #[inline(always)]
    pub fn cache_edges_filename(&self) -> String {
        self.graph_cache.edges_filename()
    }

    /// Returns the graph's offsets file's filename.
    #[inline(always)]
    pub fn cache_index_filename(&self) -> String {
        self.graph_cache.index_filename()
    }

    /// Returns the graph's fst file's filename.
    #[inline(always)]
    pub fn cache_fst_filename(&self) -> String {
        self.graph_cache.fst_filename()
    }

    /// Returns the given (by id) node's degree.
    #[inline(always)]
    pub fn node_degree(&self, node_id: usize) -> usize {
        assert!(node_id < self.size() - 1);
        unsafe {
            let ptr = (self.index.as_ptr() as *const usize).add(node_id);
            let begin = ptr.read_unaligned();
            ptr.add(1).read_unaligned() - begin
        }
    }

    /// Returns the given (by id) node's metalabel if it exists and was stored in the graph's fst.
    #[inline(always)]
    pub fn node_id_from_kmer(&self, kmer: &str) -> Result<u64, Box<dyn std::error::Error>> {
        if let Some(val) = self.kmers.get(kmer) {
            Ok(val)
        } else {
            Err(format!("error k-mer {kmer} not found").into())
        }
    }

    /// Returns the given (by id) node's edges' offsets.
    #[inline(always)]
    pub fn index_node(&self, node_id: usize) -> std::ops::Range<usize> {
        assert!(node_id < self.size() - 1);
        unsafe {
            let ptr = (self.index.as_ptr() as *const usize).add(node_id);
            ptr.read_unaligned()..ptr.add(1).read_unaligned()
        }
    }

    /// Returns a [`NeighbourIter`] iterator over the given (by id) node's neighbours.
    ///
    /// [`NeighbourIter`]: ./struct.NeighbourIter.html#
    pub fn neighbours(
        &self,
        node_id: usize,
    ) -> Result<NeighbourIter<EdgeType, Edge>, Box<dyn std::error::Error>> {
        if node_id >= self.size() - 1 {
            return Err(format!(
                "error {node_id} must be smaller than |V| = {}",
                self.size() - 1
            )
            .into());
        }

        Ok(NeighbourIter::<EdgeType, Edge>::new(
            self.graph.as_ptr() as *const Edge,
            self.index.as_ptr() as *const usize,
            node_id,
        ))
    }

    /// Returns an [`EdgeIter`] iterator over all of the graph's edges.
    ///
    /// [`EdgeIter`]: ./struct.EdgeIter.html#
    pub fn edges(&self) -> Result<EdgeIter<EdgeType, Edge>, Box<dyn std::error::Error>> {
        Ok(EdgeIter::<EdgeType, Edge>::new(
            self.graph.as_ptr() as *const Edge,
            self.index.as_ptr() as *const usize,
            0,
            self.size(),
        ))
    }

    /// Returns an [`EdgeIter`] iterator over the graph's edges in a given range.
    ///
    /// # Arguments
    ///
    /// * `begin_node` --- id of the node whose offset begin marks the beginning of the iterator's range.
    /// * `end_node` ---  id of the node whose offset end marks the end of the iterator's range.
    ///
    /// [`EdgeIter`]: ./struct.EdgeIter.html#
    pub fn edges_in_range(
        &self,
        begin_node: usize,
        end_node: usize,
    ) -> Result<EdgeIter<EdgeType, Edge>, Box<dyn std::error::Error>> {
        if begin_node > end_node {
            return Err("error invalid range, beginning after end".into());
        }
        if begin_node > self.size() || end_node > self.size() {
            return Err("error invalid range".into());
        }

        Ok(EdgeIter::<EdgeType, Edge>::new(
            self.graph.as_ptr() as *const Edge,
            self.index.as_ptr() as *const usize,
            begin_node,
            end_node,
        ))
    }

    /// Performs a sweep cut over a given diffusion vector[^1] by partition conductance.
    ///
    /// # Arguments
    ///
    /// * `diffusion` --- the diffusion vector[^1].
    /// * `target_size` --- the partition's target size[^2].
    /// * `target_volume` --- the partition's target volume[^3].
    ///
    /// [^1]: diffusion vector entries must be of type `(node_id: usize, heat: f64)`.
    /// [^2]: if `None` is provided defaults to `|V|`, effectively, the overall best partition by conducatance is returned independent on the number of nodes in it.
    /// [^3]: if `None` is provided defaults to `|E|`, effectively, the overall best partition by conducatance is returned independent on the number of edges in it.
    pub fn sweep_cut_over_diffusion_vector_by_conductance(
        &self,
        diffusion: &mut [(usize, f64)],
        target_size: Option<usize>,
        target_volume: Option<usize>,
    ) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        diffusion.sort_unstable_by_key(|(_, mass)| std::cmp::Reverse(OrderedFloat(*mass)));
        // debug
        // println!("descending-ordered mass vector {:?}", s);
        let target_size = target_size.map_or(self.size() - 1, |s| s);
        let target_volume = target_volume.map_or(self.width(), |s| s);

        let mut vol_s = 0usize;
        let mut vol_v_minus_s = self.width();
        let mut cut_s = 0usize;
        let mut community: HashSet<usize> = HashSet::new();
        let mut best_conductance = 1f64;
        let mut best_community: Vec<(usize, f64)> = Vec::new();
        let mut best_size = 0usize;
        let mut best_width = 0usize;

        for (idx, (u, _)) in diffusion.iter().enumerate() {
            let u_n = match self.neighbours(*u) {
                Ok(u_n) => {
                    vol_s = match vol_s.overflowing_add(u_n.remaining_neighbours()) {
                        (r, false) => r,
                        (_, true) => {
                            return Err(format!(
                                "error sweep cut overflow_add in vol_s at node {u}"
                            )
                            .into());
                        }
                    };
                    vol_v_minus_s = match vol_v_minus_s.overflowing_sub(u_n.remaining_neighbours())
                    {
                        (r, false) => r,
                        (_, true) => {
                            return Err(format!(
                                "error sweep cut overflow_add in vol_v_minus_s at node {u}"
                            )
                            .into());
                        }
                    };
                    u_n
                }
                Err(e) => {
                    return Err(format!("error sweep cut couldn't get {u} neighbours: {e}").into());
                }
            };
            match community.get(u) {
                Some(_) => {
                    return Err(format!(
                        "error sweepcut diffusion vector: {u} is present multiple times"
                    )
                    .into());
                }
                None => community.insert(*u),
            };
            for v in u_n {
                // if edge is (u, u) it doesn't influence delta(S)
                if v.dest() == *u {
                    continue;
                }
                if community.contains(&v.dest()) {
                    cut_s = match cut_s.overflowing_sub(1) {
                        (r, false) => r,
                        (_, true) => {
                            return Err(format!(
                                "error sweepcut overflow_sub at node {u} in neighbour {}",
                                v.dest()
                            )
                            .into());
                        }
                    };
                } else {
                    cut_s = match cut_s.overflowing_add(1) {
                        (r, false) => r,
                        (_, true) => {
                            return Err(format!(
                                "error sweepcut overflow_add at node {u} in neighbour {}",
                                v.dest()
                            )
                            .into());
                        }
                    };
                }
            }

            let conductance = (cut_s as f64) / (std::cmp::min(vol_s, vol_v_minus_s) as f64);
            if conductance < best_conductance {
                best_conductance = conductance;
                best_community = diffusion[0..=idx].to_vec();
                best_width = vol_s;
                best_size = community.len();
            }

            // truncate sweep if vol or size go over double the target value
            if community.len() > target_size * 2 || vol_s > target_volume * 2 {
                println!(
                    "Sweep cut truncated with size: {} and volume {}\n\tTarget size: {target_size}\n\tTarget volume: {target_volume}",
                    community.len(),
                    vol_s
                );
                break;
            }
        }

        Ok(Community {
            nodes: best_community,
            size: best_size,
            width: best_width,
            conductance: best_conductance,
        })
    }

    #[inline(always)]
    fn is_neighbour(&self, u: usize, v: usize) -> Option<usize> {
        assert!(u < self.size());
        let mut floor = unsafe { (self.index.as_ptr() as *const usize).add(u).read() };
        let mut ceil = unsafe { (self.index.as_ptr() as *const usize).add(u + 1).read() };
        // binary search on neighbours w, where w < v
        loop {
            // may happen if u + 1 overflows
            if floor > ceil {
                return None;
            }
            let m = floor + (ceil - floor).div_floor(2);
            let dest = unsafe { (self.graph.as_ptr() as *const Edge).add(m).read().dest() };
            match dest.cmp(&v) {
                std::cmp::Ordering::Greater => ceil = m - 1,
                std::cmp::Ordering::Less => floor = m + 1,
                _ => break Some(m),
            }
        }
    }

    fn is_triangle(&self, u: usize, v: usize, w: usize) -> Option<(usize, usize)> {
        let mut index_a = None;
        let mut index_b = None;
        let switch = v < u;

        if let Ok(mut iter) = self.neighbours(w) {
            loop {
                if let Some((index, n)) = iter._next_with_offset() {
                    if index_a.is_none() {
                        match (if switch { v } else { u }).cmp(&n.dest()) {
                            std::cmp::Ordering::Less => {
                                return None;
                            }
                            std::cmp::Ordering::Equal => {
                                if let Some(b) = index_b {
                                    return Some((index, b));
                                }
                                index_a = Some(index);
                            }
                            _ => {}
                        };
                    } else {
                        match (if switch { u } else { v }).cmp(&n.dest()) {
                            std::cmp::Ordering::Less => {
                                return None;
                            }
                            std::cmp::Ordering::Equal => {
                                if let Some(a) = index_a {
                                    return Some(if switch { (index, a) } else { (a, index) });
                                }
                            }
                            _ => {}
                        };
                    }
                } else {
                    return None;
                }
                if let Some((index, n)) = iter._next_back_with_offset() {
                    if index_b.is_none() {
                        match (if switch { u } else { v }).cmp(&n.dest()) {
                            std::cmp::Ordering::Greater => return None,
                            std::cmp::Ordering::Equal => {
                                if let Some(a) = index_a {
                                    return Some((a, index));
                                }
                                index_b = Some(index);
                            }
                            _ => {}
                        };
                    } else {
                        match (if switch { v } else { u }).cmp(&n.dest()) {
                            std::cmp::Ordering::Greater => return None,
                            std::cmp::Ordering::Equal => {
                                if let Some(b) = index_b {
                                    return Some(if switch { (b, index) } else { (index, b) });
                                }
                            }
                            _ => {}
                        };
                    }
                } else {
                    return None;
                }
            }
        }
        None
    }

    /// Returns number of entries in the offset file[^1].
    ///
    /// [^1]: this is equivalent to `|V| + 1`, as there is an extrea offset file entry to mark the end of edges' offsets.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.graph_cache.index_bytes // num nodes
    }

    /// Returns number of entries in edge file[^1].
    ///
    /// [^1]: this is equivalent to `|E|`.
    #[inline(always)]
    pub fn width(&self) -> usize {
        self.graph_cache.graph_bytes // num edges
    }

    fn exports_fetch_increment(&mut self) -> Result<usize, Box<dyn std::error::Error>> {
        self.exports = match self.exports.overflowing_add(1) {
            (r, false) => r,
            (_, true) => {
                self.exports = u8::MAX;
                return Err(
                    "error overflowed export count var in graph struct, please provide an identifier for your export".into()
                );
            }
        };
        Ok((self.exports - 1) as usize)
    }

    /// Applies the mask: fn(usize) -> bool function to each node id and returns the resulting subgraph.
    ///
    /// The resulting subgraph is that wherein only the set of nodes, `S ⊂ V`, of the nods for whose the output of mask is true, as well as, only the set of edges coming from and going to nodes in `S`[^1].
    ///
    /// [^1]: the node ids of the subgraph may not, and probably will not, correspond to the original node identifiers, efffectively, it will be a whole new graph.
    pub fn apply_mask_to_nodes(
        &self,
        mask: fn(usize) -> bool,
        identifier: Option<String>,
    ) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
        let node_count = self.size();
        if node_count < 2 {
            return Err("error can't mask empty graph".into());
        }
        let c_fn = CACHE_DIR.to_string() + "edge_count.tmp";
        let i_fn = CACHE_DIR.to_string() + "node_index.tmp";

        // allocate |V| + 1 usize's to store the beginning and end offsets for each node's edges
        let mut edge_count = SharedSliceMut::<usize>::abst_mem_mut(c_fn, node_count, true)?;
        // allocate |V| usize's to store each node's new id if present in the subgraph
        let mut node_index = SharedSliceMut::<usize>::abst_mem_mut(i_fn, node_count - 1, true)?;

        let mut curr_node_index: usize = 0;
        let mut curr_edge_count: usize = 0;
        *edge_count.get_mut(0) = curr_edge_count;
        // iterate over |V|
        for u in 0..node_count - 1 {
            if mask(u) {
                *node_index.get_mut(u) = curr_node_index;
                curr_node_index += 1;
                let neighbours = self.neighbours(u)?.filter(|x| mask(x.dest())).count();
                curr_edge_count += neighbours;
                *edge_count.get_mut(u + 1) = curr_edge_count;
            } else {
                *node_index.get_mut(u) = usize::MAX;
                *edge_count.get_mut(u + 1) = curr_edge_count;
            }
        }

        let mut kmer_stream = self.kmers.stream();
        let id = match identifier {
            Some(id) => id,
            None => id_for_subgraph_export(
                self.graph_id()?,
                Some(self.clone().exports_fetch_increment()?),
            )?,
        };

        // prepare files for subgraph
        let (edges_fn, id) = GraphCache::<EdgeType, Edge>::init_cache_file_from_id_or_random(
            Some(id),
            FileType::Edges,
        )?;
        let (index_fn, id) = GraphCache::<EdgeType, Edge>::init_cache_file_from_id_or_random(
            Some(id),
            FileType::Index,
        )?;
        let (kmers_fn, _) = GraphCache::<EdgeType, Edge>::init_cache_file_from_id_or_random(
            Some(id),
            FileType::Fst,
        )?;
        let mut edges = SharedSliceMut::<Edge>::abst_mem_mut(edges_fn, curr_edge_count, true)?;
        let mut index = SharedSliceMut::<usize>::abst_mem_mut(index_fn, curr_node_index + 1, true)?;
        let kmer_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(kmers_fn.clone())?;
        let mut build = MapBuilder::new(&kmer_file)?;

        // write nodes in order of lexicographically ordered kmers to avoid sorting kmers
        // FIXME: what is more costly random page accesses or sorting build merging kmer fst?
        *index.get_mut(0) = 0;
        while let Some((kmer, node_id)) = kmer_stream.next() {
            let id = node_id as usize;
            if mask(id) {
                // write index file for next node (id + 1)
                let new_id = *node_index.get(id);
                *index.get_mut(new_id + 1) = *edge_count.get(id + 1);
                // write edge file node
                edges
                    .write_slice(
                        *edge_count.get(id),
                        self.neighbours(id)?
                            .filter(|x| mask(x.dest()))
                            .collect::<Vec<Edge>>()
                            .as_slice(),
                    )
                    .ok_or("error writing edges for node {id}")?;
                // write fst for node
                build.insert(kmer, new_id as u64)?;
            }
        }
        self.cleanup_cache()?;
        let cache: GraphCache<EdgeType, Edge> = GraphCache::open(kmers_fn, None)?;
        GraphMemoryMap::init(cache, self.thread_count)
    }

    /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format keeping all edge and node labelings[^1].
    ///
    /// [^1]: if none of the edge or node labeling is wanted consider using [`export_petgraph_stripped`].
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    /// [`export_petgraph_stripped`]: ./struct.GraphMemoryMap.html#method.export_petgraph_stripped
    pub fn export_petgraph(
        &self,
    ) -> Result<DiGraph<NodeIndex<usize>, EdgeType>, Box<dyn std::error::Error>> {
        let mut graph = DiGraph::<NodeIndex<usize>, EdgeType>::new();
        let node_count = self.size() - 1;

        (0..node_count).for_each(|u| {
            graph.add_node(NodeIndex::new(u));
        });
        (0..node_count)
            .filter_map(|u| match self.neighbours(u) {
                Ok(neighbours_of_u) => Some((u, neighbours_of_u)),
                Err(e) => {
                    eprint!("error getting neihghbours of {u} (proceeding anyways): {e}");
                    None
                }
            })
            .for_each(|(u, u_n)| {
                u_n.for_each(|v| {
                    graph.add_edge(NodeIndex::new(u), NodeIndex::new(v.dest()), v.e_type());
                });
            });

        Ok(graph)
    }

    /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format stripping any edge or node labelings whatsoever.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn export_petgraph_stripped(&self) -> Result<DiGraph<(), ()>, Box<dyn std::error::Error>> {
        let mut graph = DiGraph::<(), ()>::new();
        let node_count = self.size() - 1;

        (0..node_count).for_each(|_| {
            graph.add_node(());
        });
        (0..node_count)
            .filter_map(|u| match self.neighbours(u) {
                Ok(neighbours_of_u) => Some((u, neighbours_of_u)),
                Err(e) => {
                    eprint!("error getting neihghbours of {u} (proceeding anyways): {e}");
                    None
                }
            })
            .for_each(|(u, u_n)| {
                u_n.for_each(|v| {
                    graph.add_edge(NodeIndex::new(u), NodeIndex::new(v.dest()), ());
                });
            });

        Ok(graph)
    }

    fn init_procedural_memory_build_reciprocal(
        &self,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<usize>,
            AbstractedProceduralMemoryMut<usize>,
        ),
        Box<dyn std::error::Error>,
    > {
        let edge_count = self.width();
        let node_count = self.size() - 1;

        let template_fn = self.graph_cache.graph_filename.clone();
        let er_fn = cache_file_name(template_fn.clone(), FileType::EdgeReciprocal, None)?;
        let eo_fn = cache_file_name(template_fn.clone(), FileType::EdgeOver, None)?;

        let edge_reciprocal = SharedSliceMut::<usize>::abst_mem_mut(er_fn, edge_count, true)?;
        let edge_out = SharedSliceMut::<usize>::abst_mem_mut(eo_fn, node_count, true)?;

        Ok((edge_reciprocal, edge_out))
    }

    fn build_reciprocal_edge_index(
        self,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<usize>,
            AbstractedProceduralMemoryMut<usize>,
        ),
        Box<dyn std::error::Error>,
    > {
        let node_count = self.size() - 1;
        let edge_count = self.width();

        let threads = self.thread_count.max(1) as usize;
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = Arc::new(SharedSlice::<usize>::new(
            self.index.as_ptr() as *const usize,
            node_count + 1,
        ));
        let graph_ptr = Arc::new(SharedSlice::<Edge>::new(
            self.graph.as_ptr() as *const Edge,
            edge_count,
        ));

        let (er, eo) = self.init_procedural_memory_build_reciprocal()?;

        let synchronize = Arc::new(Barrier::new(threads));

        thread::scope(|scope| {
            for tid in 0..threads {
                let graph_ptr = Arc::clone(&graph_ptr);
                let index_ptr = Arc::clone(&index_ptr);

                let mut er = er.shared_slice();
                let mut eo = eo.shared_slice();

                let synchronize = Arc::clone(&synchronize);

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);
                scope.spawn(move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let mut edges_start = *index_ptr.get(begin);
                    for u in begin..end {
                        let mut eo_at_end = true;
                        let edges_stop = *index_ptr.get(u + 1);
                        for edge_offset in edges_start..edges_stop {
                            let v = graph_ptr.get(edge_offset).dest();
                            if v > u {
                                eo_at_end = false;
                                *eo.get_mut(u) = edge_offset;
                                break;
                            }
                            // FIXME: add section u == v to update edge_reciprocal array?
                        }
                        if eo_at_end {
                            *eo.get_mut(u) = edges_stop;
                        }
                        edges_start = edges_stop;
                    }

                    synchronize.wait();

                    for u in begin..end {
                        for edge_offset in *eo.get(u)..*index_ptr.get(u + 1) {
                            let v = graph_ptr.get(edge_offset).dest();
                            let mut floor = *index_ptr.get(v);
                            let mut ceil = *eo.get(v);
                            // binary search on neighbours w, where w < v
                            let reciprocal = loop {
                                if floor > ceil {
                                    return Err(format!("error couldn't find reciprocal for edge {edge_offset}, u: ({u}) -> v: ({v})").into());
                                }
                                let m = floor + (ceil - floor).div_floor(2);
                                let dest = graph_ptr.get(m).dest();
                                match dest.cmp(&u) {
                                    std::cmp::Ordering::Greater => ceil = m - 1,
                                    std::cmp::Ordering::Less => floor = m + 1,
                                    _ => break m,
                                }
                            };
                            *er.get_mut(edge_offset) = reciprocal;
                            *er.get_mut(reciprocal) = edge_offset;
                        }
                    }

                    Ok(())
                });
            }
        })
        .unwrap();
        er.flush()?;
        eo.flush()?;
        Ok((er, eo))
    }

    pub(crate) fn get_edge_reciprocal(
        &self,
    ) -> Result<AbstractedProceduralMemory<usize>, Box<dyn std::error::Error>> {
        let fn_template = self.graph_cache.graph_filename.clone();
        let er_fn = cache_file_name(fn_template.clone(), FileType::EdgeReciprocal, None)?;
        let dud = Vec::new();
        let er = match OpenOptions::new().read(true).open(er_fn.as_str()) {
            Ok(i) => SharedSlice::<usize>::abstract_mem(
                er_fn,
                dud,
                i.metadata().unwrap().len() as usize / std::mem::size_of::<usize>(),
                true,
            ),
            Err(_) => {
                self.clone().build_reciprocal_edge_index()?;
                match OpenOptions::new().read(true).open(er_fn.as_str()) {
                    Ok(i) => SharedSlice::<usize>::abstract_mem(
                        er_fn,
                        dud,
                        i.metadata().unwrap().len() as usize / std::mem::size_of::<usize>(),
                        true,
                    ),
                    Err(e) => {
                        return Err(format!(
                            "error creating abstractmemory for edge_reciprocal {e}"
                        )
                        .into());
                    }
                }
            }
        };
        Ok(er?)
    }

    pub(crate) fn get_edge_dest_id_over_source(
        &self,
    ) -> Result<AbstractedProceduralMemory<usize>, Box<dyn std::error::Error>> {
        let fn_template = self.graph_cache.graph_filename.clone();
        let eo_fn = cache_file_name(fn_template.clone(), FileType::EdgeOver, None)?;
        let dud = Vec::new();
        let eo = match OpenOptions::new().read(true).open(eo_fn.as_str()) {
            Ok(i) => SharedSlice::<usize>::abstract_mem(
                eo_fn,
                dud,
                i.metadata().unwrap().len() as usize / std::mem::size_of::<usize>(),
                true,
            ),
            Err(_) => {
                self.clone().build_reciprocal_edge_index()?;
                match OpenOptions::new().read(true).open(eo_fn.as_str()) {
                    Ok(i) => SharedSlice::<usize>::abstract_mem(
                        eo_fn,
                        dud,
                        i.metadata().unwrap().len() as usize / std::mem::size_of::<usize>(),
                        true,
                    ),
                    Err(e) => {
                        return Err(
                            format!("error creating abstractmemory for edge_over {e}").into()
                        );
                    }
                }
            }
        };
        Ok(eo?)
    }

    pub fn cleanup_cache(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.graph_cache.cleanup_cache()
    }
}

#[derive(Debug, Clone)]
pub struct NeighbourIter<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    edge_ptr: *const Edge,
    _orig_edge_ptr: *const Edge,
    _orig_id_ptr: *const usize,
    id: usize,
    count: usize,
    offset: usize,
    _phantom: std::marker::PhantomData<EdgeType>,
}

#[derive(Debug)]
pub struct EdgeIter<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    edge_ptr: *const Edge,
    id_ptr: *const usize,
    id: usize,
    end: usize,
    count: usize,
    _phantom: std::marker::PhantomData<EdgeType>,
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> NeighbourIter<EdgeType, Edge> {
    fn new(edge_mmap: *const Edge, id_mmap: *const usize, node_id: usize) -> Self {
        let _orig_edge_ptr = edge_mmap;
        let _orig_id_ptr = id_mmap;
        let id_ptr = unsafe { id_mmap.add(node_id) };
        let offset = unsafe { id_ptr.read_unaligned() };

        NeighbourIter {
            edge_ptr: unsafe { edge_mmap.add(offset) },
            _orig_edge_ptr,
            _orig_id_ptr,
            id: node_id,
            count: unsafe { id_ptr.add(1).read_unaligned() - offset },
            offset,
            _phantom: std::marker::PhantomData::<EdgeType>,
        }
    }

    #[inline(always)]
    fn _into_neighbour(&self) -> Self {
        NeighbourIter::new(self._orig_edge_ptr, self._orig_id_ptr, unsafe {
            self.edge_ptr.read_unaligned().dest()
        })
    }

    fn _next_back_with_offset(&mut self) -> Option<(usize, Edge)> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let next: (usize, Edge);
        unsafe {
            next = (self.id, self.edge_ptr.add(self.count).read_unaligned());
        };
        Some(next)
    }

    pub fn remaining_neighbours(&self) -> usize {
        self.count
    }

    fn _next_with_offset(&mut self) -> Option<(usize, Edge)> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let next: (usize, Edge);
        self.edge_ptr = unsafe {
            next = (self.offset, self.edge_ptr.read_unaligned());
            self.edge_ptr.add(1)
        };
        self.offset += 1;
        Some(next)
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> DoubleEndedIterator
    for NeighbourIter<EdgeType, Edge>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<Edge> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let next: Edge;
        unsafe {
            next = self.edge_ptr.add(self.count).read_unaligned();
        };
        Some(next)
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> Iterator
    for NeighbourIter<EdgeType, Edge>
{
    type Item = Edge;

    #[inline(always)]
    fn next(&mut self) -> Option<Edge> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        self.offset += 1;
        let next: Edge;
        self.edge_ptr = unsafe {
            next = self.edge_ptr.read_unaligned();
            self.edge_ptr.add(1)
        };
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.count;
        (remaining, Some(remaining))
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> EdgeIter<EdgeType, Edge> {
    #[inline(always)]
    fn new(edge_mmap: *const Edge, id_mmap: *const usize, start: usize, end: usize) -> Self {
        let id_ptr = unsafe { id_mmap.add(start) };
        let offset = unsafe { id_ptr.read_unaligned() };
        let edge_ptr = unsafe { edge_mmap.add(offset) };

        EdgeIter {
            edge_ptr,
            id_ptr,
            id: start,
            end,
            count: unsafe { id_ptr.add(1).read_unaligned() - offset },
            _phantom: std::marker::PhantomData::<EdgeType>,
        }
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> Iterator for EdgeIter<EdgeType, Edge> {
    type Item = Edge;

    #[inline(always)]
    fn next(&mut self) -> Option<Edge> {
        if self.count == 0 {
            self.id += 1;
            if self.id > self.end {
                return None;
            }
            unsafe {
                self.id_ptr = self.id_ptr.add(1);
                let offset = self.id_ptr.read_unaligned();
                self.count = self.id_ptr.add(1).read_unaligned() - offset;
            };
        }
        self.count -= 1;
        let next: Edge;
        self.edge_ptr = unsafe {
            next = self.edge_ptr.read_unaligned();
            self.edge_ptr.add(1)
        };
        Some(next)
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.end - self.id;
        (remaining, Some(remaining))
    }
}
impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> std::ops::Index<std::ops::RangeFull>
    for GraphMemoryMap<EdgeType, Edge>
{
    type Output = [Edge];
    #[inline]
    fn index(&self, _index: std::ops::RangeFull) -> &[Edge] {
        unsafe {
            slice::from_raw_parts(
                self.graph.as_ptr() as *const Edge,
                self.size() * 8 / self.edge_size,
            )
        }
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> std::ops::Index<std::ops::Range<usize>>
    for GraphMemoryMap<EdgeType, Edge>
{
    type Output = [Edge];
    #[inline]
    fn index(&self, index: std::ops::Range<usize>) -> &[Edge] {
        unsafe {
            slice::from_raw_parts(
                self.graph.as_ptr().add(index.start * self.edge_size) as *const Edge,
                index.end - index.start,
            )
        }
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> std::ops::Index<std::ops::Range<u64>>
    for GraphMemoryMap<EdgeType, Edge>
{
    type Output = [Edge];
    #[inline]
    fn index(&self, index: std::ops::Range<u64>) -> &[Edge] {
        let start = index.start as usize;
        let end = index.end as usize;

        unsafe {
            slice::from_raw_parts(
                self.graph.as_ptr().add(start * self.edge_size) as *const Edge,
                end - start,
            )
        }
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> Debug
    for GraphMemoryMap<EdgeType, Edge>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryMappedData {{\n\t
            filename: {},\n\t
            index_filename: {},\n\t
            size: {},\n\t
            width: {},\n\t
            }}",
            self.graph_cache.graph_filename,
            self.graph_cache.index_filename,
            self.size(),
            self.width(),
        )
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> Debug for GraphCache<EdgeType, Edge> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n\tgraph filename: {}\n\tindex filename: {}\n\tkmer filename: {}\n}}",
            self.graph_filename, self.index_filename, self.kmer_filename
        )
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> Clone for GraphCache<EdgeType, Edge> {
    fn clone(&self) -> Self {
        if !self.readonly {
            panic!(
                "can't clone GraphCache before it is readonly, fst needs file ownership to map kmers to node_ids"
            );
        }
        Self {
            graph_file: self.graph_file.clone(),
            index_file: self.index_file.clone(),
            kmer_file: self.kmer_file.clone(),
            graph_filename: self.graph_filename.clone(),
            index_filename: self.index_filename.clone(),
            kmer_filename: self.kmer_filename.clone(),
            graph_bytes: self.graph_bytes,
            index_bytes: self.index_bytes,
            readonly: true,
            batch: self.batch,
            _marker1: self._marker1,
            _marker2: self._marker2,
        }
    }
}

#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Community<NodeId: Copy + Debug> {
    /// (s, h(s)), ∀ s ϵ S, where h(s) is the heat kernel diffusion for a given node s
    pub nodes: Vec<(NodeId, f64)>,
    /// |S|
    pub size: usize,
    /// vol(S)
    pub width: usize,
    /// ϕ(S)
    pub conductance: f64,
}

type ProcessGGCATEntry = Option<(Vec<u8>, u64)>;
type MTXHeader = (usize, usize, usize);
