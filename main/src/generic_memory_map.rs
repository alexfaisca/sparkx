#[expect(unused_imports)]
use crate::fst_merge_sort;
use crate::generic_edge::{GenericEdge, GenericEdgeType};
use crate::shared_slice::{
    AbstractedProceduralMemory, AbstractedProceduralMemoryMut, SharedQueueMut, SharedSlice,
    SharedSliceMut,
};
use core::{f64, fmt, panic};
use crossbeam::thread;
use fst::{Map, MapBuilder, Streamer};
use glob::glob;
use memmap2::{Mmap, MmapMut, MmapOptions};
use ordered_float::{Float, OrderedFloat};
use rand::Rng;
use rand_distr::{Distribution, Poisson};
use regex::Regex;
use rustworkx_core::petgraph::graph::{DiGraph, NodeIndex};
use static_assertions::const_assert;
use std::{
    any::type_name,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Debug,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Error, Write},
    marker::PhantomData,
    path::Path,
    process::Command,
    slice,
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
    },
};
use zerocopy::*; // Using crossbeam for scoped threads

const_assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u64>());

static CACHE_DIR: &str = "./cache/";
static BYTES_U64: u64 = std::mem::size_of::<u64>() as u64;
// FIXME: Make this a member of struct graph_cache

fn _type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

fn external_sort_by_content(temp: &str, sorted: &str) -> std::io::Result<()> {
    // sort based on 2nd column (content), not line number
    Command::new("sort")
        .args(["-k2", temp, "-o", sorted])
        .status()?;
    Ok(())
}

#[derive(Debug)]
enum FileType {
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
    /// doesn't yield a filename --- yields id to be used in filename
    ExportGraphMask,
}

fn cache_file_name(
    original_filename: String,
    target_type: FileType,
    sequence_number: Option<usize>,
) -> Result<String, Error> {
    let path = Path::new(original_filename.as_str());

    let file_name = match path.file_name() {
        Some(i) => match i.to_str() {
            Some(i) => i,
            None => panic!("error invalid path string"),
        },
        None => panic!("error invalid path"),
    };
    let parent_dir = path.parent().unwrap_or_else(|| Path::new(""));

    // extract id from filename
    let re = match Regex::new(r#"^(?:[a-zA-Z0-9_]+_)(\w+)(\.[a-zA-Z0-9]+$)"#).ok() {
        Some(i) => i,
        None => panic!("error analyzing file name"),
    };
    let caps = match re.captures(file_name) {
        Some(i) => i,
        None => panic!("error capturing file name"),
    };
    let id = match caps.get(1) {
        Some(i) => i.as_str(),
        None => panic!("error capturing file name"),
    };

    // construct filename: e.g., index_12345.mmap
    let new_filename = match target_type {
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
        FileType::KmerSortedTmp => format!("{}_{}.{}", "kmersortedtmpfile", id, "tmp"),
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
        // this isn't a filename --- it's an id to be used in filenames
        FileType::ExportGraphMask => match sequence_number {
            Some(i) => format!("{}{}{}", "masked_export", i, id),
            None => format!("{}{}", "masked_export", id),
        },
    };

    Ok(parent_dir.join(new_filename).to_string_lossy().into_owned())
}

fn cleanup_cache() -> Result<(), Error> {
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
        Err(e) => panic!("{}", e),
    }
}

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
    _marker1: PhantomData<Edge>,
    _marker2: PhantomData<EdgeType>,
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> GraphCache<EdgeType, Edge> {
    fn init_cache_file_from_id_or_random(
        graph_id: Option<String>,
        target_type: FileType,
    ) -> Result<(String, String), Error> {
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
                _ => panic!(
                    "error unsupported file type for GraphCache: {}",
                    target_type
                ),
            },
            id,
        ))
    }

    pub fn init() -> Result<GraphCache<EdgeType, Edge>, Error> {
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }

        let (graph_filename, id) = Self::init_cache_file_from_id_or_random(None, FileType::Edges)?;
        let (index_filename, id) =
            Self::init_cache_file_from_id_or_random(Some(id), FileType::Index)?;
        let (kmer_filename, _) =
            Self::init_cache_file_from_id_or_random(Some(id), FileType::KmerTmp)?;

        let graph_file: File = match OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .create(true)
            .open(graph_filename.as_str())
        {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", graph_filename, e),
        };

        let index_file: File = match OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .create(true)
            .open(index_filename.as_str())
        {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", index_filename, e),
        };

        let kmer_file: File = match OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .create(true)
            .open(kmer_filename.as_str())
        {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", index_filename, e),
        };

        Ok(GraphCache::<EdgeType, Edge> {
            graph_file: Arc::new(graph_file),
            index_file: Arc::new(index_file),
            kmer_file: Arc::new(kmer_file),
            graph_filename,
            index_filename,
            kmer_filename,
            graph_bytes: 0,
            index_bytes: 0,
            readonly: false,
            _marker1: PhantomData::<Edge>,
            _marker2: PhantomData::<EdgeType>,
        })
    }

    pub fn init_with_id(id: String) -> Result<GraphCache<EdgeType, Edge>, Error> {
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }
        if id.is_empty() {
            panic!("error invalid cache id");
        }

        let (graph_filename, id) =
            Self::init_cache_file_from_id_or_random(Some(id), FileType::Edges)?;
        let (index_filename, id) =
            Self::init_cache_file_from_id_or_random(Some(id), FileType::Index)?;
        let (kmer_filename, _) =
            Self::init_cache_file_from_id_or_random(Some(id), FileType::KmerTmp)?;

        let graph_file: File = match OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .create(true)
            .open(graph_filename.as_str())
        {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", graph_filename, e),
        };

        let index_file: File = match OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .create(true)
            .open(index_filename.as_str())
        {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", index_filename, e),
        };

        let kmer_file: File = match OpenOptions::new()
            .read(true)
            .write(true)
            .truncate(true)
            .create(true)
            .open(kmer_filename.as_str())
        {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", index_filename, e),
        };

        Ok(GraphCache::<EdgeType, Edge> {
            graph_file: Arc::new(graph_file),
            index_file: Arc::new(index_file),
            kmer_file: Arc::new(kmer_file),
            graph_filename,
            index_filename,
            kmer_filename,
            graph_bytes: 0,
            index_bytes: 0,
            readonly: false,
            _marker1: PhantomData::<Edge>,
            _marker2: PhantomData::<EdgeType>,
        })
    }

    pub fn open(filename: String) -> Result<GraphCache<EdgeType, Edge>, Error> {
        let graph_filename = cache_file_name(filename.clone(), FileType::Edges, None)?;
        let index_filename = cache_file_name(filename.clone(), FileType::Index, None)?;
        let kmer_filename = cache_file_name(filename.clone(), FileType::KmerTmp, None)?;

        let graph_file: File = match OpenOptions::new().read(true).open(graph_filename.as_str()) {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", graph_filename, e),
        };
        let index_file: File = match OpenOptions::new().read(true).open(index_filename.as_str()) {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", index_filename, e),
        };
        let kmer_file: File = match OpenOptions::new().read(true).open(kmer_filename.as_str()) {
            Ok(file) => file,
            Err(e) => panic!("error couldnt open file {}: {}", kmer_filename, e),
        };

        let graph_len = graph_file.metadata().unwrap().len() as usize;
        let index_len = index_file.metadata().unwrap().len() as usize;

        Ok(GraphCache::<EdgeType, Edge> {
            graph_file: Arc::new(graph_file),
            index_file: Arc::new(index_file),
            kmer_file: Arc::new(kmer_file),
            graph_filename,
            index_filename,
            kmer_filename,
            graph_bytes: graph_len,
            index_bytes: index_len,
            readonly: true,
            _marker1: PhantomData::<Edge>,
            _marker2: PhantomData::<EdgeType>,
        })
    }

    pub fn write_node(&mut self, node_id: usize, data: &[Edge], label: &str) -> Result<(), Error> {
        match node_id == self.index_bytes {
            true => {
                writeln!(self.kmer_file, "{}\t{}", node_id, label)?;

                match self.index_file.write_all(self.graph_bytes.as_bytes()) {
                    Ok(_) => self.index_bytes += 1,
                    Err(e) => panic!("error writing index for {}: {}", node_id, e),
                };

                match self.graph_file.write_all(bytemuck::cast_slice(data)) {
                    Ok(_) => {
                        self.graph_bytes += data.len();
                        Ok(())
                    }
                    Err(e) => panic!("error writing edges for {}: {}", node_id, e),
                }
            }
            false => panic!(
                "error nodes must be mem mapped in ascending order, (id: {}, expected_id: {})",
                node_id, self.index_bytes
            ),
        }
    }

    fn build_fst_from_sorted_file(&self, sorted_file: File) -> Result<(), Error> {
        // As reandonly is false, no clones exist -> safe to take file ownership from Arc
        let mut build = match MapBuilder::new(&*self.kmer_file) {
            Ok(i) => i,
            Err(e) => panic!("error couldn't initialize builder: {}", e),
        };

        let mut reader = BufReader::new(sorted_file);
        let mut line = Vec::new();

        loop {
            let () = match reader.read_until(b'\n', &mut line) {
                Ok(i) => {
                    if !i > 0 {
                        break;
                    }
                }
                Err(e) => panic!("error reading file {}", e),
            };
            if let Ok(text) = std::str::from_utf8(&line) {
                let mut parts = text.trim_end().splitn(2, '\t');
                if let (Some(id_value), Some(kmer)) = (parts.next(), parts.next()) {
                    let id = match id_value.parse::<u64>() {
                        Ok(i) => i,
                        Err(_) => panic!("error couldn't convert {} to u64", id_value),
                    };
                    match build.insert(kmer, id) {
                        Ok(i) => i,
                        Err(e) => panic!(
                            "error couldn't insert kmer for node (id {}): {}",
                            id_value, e
                        ),
                    };
                }
            }
        }

        match build.finish() {
            Ok(_) => Ok(()),
            Err(e) => panic!("error couldn't finish fst build: {}", e),
        }
    }

    pub fn make_readonly(&mut self) -> Result<(), Error> {
        if self.readonly {
            return Ok(());
        }

        // Complete index file
        match self.index_file.write_all(self.graph_bytes.as_bytes()) {
            Ok(_) => self.index_bytes += 1,
            Err(e) => panic!("error couldn't finish index: {}", e),
        };

        // Build finite state tranducer for k-mer to
        let fst_filename = cache_file_name(self.kmer_filename.clone(), FileType::Fst, None)?;
        let sorted_file =
            cache_file_name(self.kmer_filename.clone(), FileType::KmerSortedTmp, None)?;

        //FIXME: Build fst in batches.
        // let mmap = unsafe {
        //     MmapOptions::new().map(
        //         &OpenOptions::new()
        //             .read(true)
        //             .open(self.kmer_filename.clone())?,
        //     )
        // };
        //
        // let merge_sort = fst_merge_sort::Merger::new(mmap.iter(), Path::new(sorted_file.clone()));

        external_sort_by_content(self.kmer_filename.as_str(), sorted_file.as_str())?;

        self.kmer_file = match File::create(fst_filename.clone()) {
            Ok(i) => {
                self.kmer_filename = fst_filename;
                Arc::new(i)
            }
            Err(e) => panic!("error couldn't create mer .fst file: {}", e),
        };

        self.build_fst_from_sorted_file(File::open(sorted_file)?)?;

        // Make all files read-only and cleanup
        for file in [&self.index_file, &self.graph_file, &self.kmer_file] {
            let mut permissions = file.metadata()?.permissions();
            permissions.set_readonly(true);
            file.set_permissions(permissions)?;
        }
        self.readonly = true;
        cleanup_cache()
    }

    pub fn cleanup_cache(&self) -> Result<(), Error> {
        cleanup_cache()
    }
}

#[derive(Clone)]
struct FindDisjointSetsEulerTrails {
    trails: Vec<((usize, usize, usize), usize)>,
    cycle_check: bool,
}

impl FindDisjointSetsEulerTrails {
    pub fn new(trails: &mut [(usize, usize, usize)]) -> Self {
        trails.sort_unstable_by_key(|(trail_id, _, _)| *trail_id);
        FindDisjointSetsEulerTrails {
            trails: trails
                .iter()
                .map(|(trail, parent_trail, pos)| ((*trail, *parent_trail, *pos), *parent_trail))
                .collect::<Vec<((usize, usize, usize), usize)>>(),
            cycle_check: false,
        }
    }

    fn cycle_b(t: &mut [((usize, usize, usize), usize)], visited: &mut [bool], i: usize) -> usize {
        if t[i].0.0 == t[i].1 {
            return t[i].1;
        }

        if visited[t[i].0.0] {
            t[i].1 = i;
            t[i].0.1 = i;
            t[i].0.2 = 0;
            i
        } else {
            visited[t[i].0.0] = true;
            t[i].1 = Self::cycle_b(t, visited, t[i].1);
            t[i].1
        }
    }

    fn cycle_break(&mut self, i: usize) -> usize {
        let mut visited = vec![false; self.trails.len()];
        let t = &mut self.trails;
        Self::cycle_b(t.as_mut_slice(), visited.as_mut_slice(), i)
    }

    pub fn cycle_check(&mut self) {
        if self.cycle_check {
            return;
        }

        // FIXME: Ugly!

        // Break cycles in graph
        for (id, _i) in self.clone().trails.iter().enumerate() {
            self.cycle_break(id);
        }
        // Works as an union find in tree
        for (id, _i) in self.clone().trails.iter().enumerate() {
            self.cycle_break(id);
        }

        self.cycle_check = true;
    }
}

pub struct EulerTrail<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: GraphMemoryMap<EdgeType, Edge>,
}

impl<EdgeType, Edge> EulerTrail<EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    pub fn new(graph: GraphMemoryMap<EdgeType, Edge>) -> Result<EulerTrail<EdgeType, Edge>, Error> {
        Ok(EulerTrail { graph })
    }

    fn create_memmapped_mut_slice_from_tmp_file<V>(
        filename: String,
        len: usize,
    ) -> Result<(SharedSliceMut<V>, MmapMut), Error> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&filename)?;
        file.set_len((len * std::mem::size_of::<V>()) as u64)?;
        SharedSliceMut::<V>::from_file(&file)
    }

    fn create_memmapped_slice_from_tmp_file<V>(
        filename: String,
    ) -> Result<(SharedSlice<V>, Mmap), Error> {
        let file = OpenOptions::new().read(true).open(filename)?;
        SharedSlice::<V>::from_file(&file)
    }

    fn merge_euler_trails(
        &self,
        cycle_offsets: Vec<(usize, usize, usize)>,
    ) -> Result<Vec<(u64, u64)>, Error> {
        let mut trail_heads: HashMap<u64, Vec<(usize, usize, usize)>> = HashMap::new();
        let mmap_fn = cache_file_name(
            self.graph.graph_cache.graph_filename.clone(),
            FileType::EulerTmp,
            None,
        )?;
        let (cycles, _mmap) = Self::create_memmapped_slice_from_tmp_file::<u64>(mmap_fn)?;

        cycle_offsets.iter().for_each(|(idx, begin, _)| {
            trail_heads
                .entry(*cycles.get(*begin))
                .or_default()
                .push((*idx, *idx, 0));
        });

        // generate writing sets
        let chunk = 4096; // read 8 pages of 4KB at a time
        for (t_idx, t_begin, t_end) in cycle_offsets.iter() {
            let trail_ptr = match cycles.slice(*t_begin, *t_end) {
                Some(i) => i.as_ptr(),
                None => panic!("error getting memmapped slice of trail {}", t_idx),
            };
            let trail_slice = SharedSlice::<u64>::new(trail_ptr, *t_end - *t_begin);
            let mut pos = 0; // pos in u64 terms
            while let Some(next_slice) = trail_slice.slice(pos, pos + chunk) {
                for (pos_idx, node) in next_slice.iter().enumerate() {
                    if let Some(head_v) = trail_heads.get_mut(node) {
                        let p_idx = pos_idx + pos + 1;
                        for (vec_idx, (in_cyc, _, _)) in head_v.clone().iter().enumerate() {
                            if *in_cyc == *t_idx {
                                continue;
                            }
                            head_v[vec_idx] = (*in_cyc, *t_idx, p_idx);
                        }
                    }
                }
                pos += next_slice.len();
            }
        }

        // break cycles
        let mut v: Vec<_> = trail_heads
            .values_mut()
            .flat_map(|vec| vec.drain(..))
            .collect();

        let mut euler_trail_sets = FindDisjointSetsEulerTrails::new(v.as_mut());
        euler_trail_sets.cycle_check();

        // Union find and write
        let mut trail_sets: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();

        for ((trail, parent_trail, pos), grand_parent) in euler_trail_sets.trails {
            trail_sets
                .entry(grand_parent)
                .or_default()
                .push((trail, parent_trail, pos));
        }

        let mut output_len: HashMap<usize, (usize, usize)> = HashMap::new();
        // in nodes usize
        let cycles_length: Vec<usize> = cycle_offsets.iter().map(|(_, b, e)| e - b).collect();
        for head_trail in trail_sets.keys() {
            output_len.insert(*head_trail, (*head_trail, 1));
            for (trail, _, _) in trail_sets.get(head_trail).unwrap() {
                output_len.get_mut(head_trail).unwrap().1 += cycles_length[*trail] - 1;
            }
        }

        let mut keys_by_trail_size: Vec<(usize, usize)> =
            output_len.values().map(|&(a, b)| (a, b)).collect();
        keys_by_trail_size.sort_unstable_by_key(|(_, s)| std::cmp::Reverse(*s));

        for (idx, (head_trail, output_len)) in keys_by_trail_size.iter().enumerate() {
            // Initialize writing guide
            let trail_guide = trail_sets.get(head_trail).unwrap();
            let mut pos_map: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for (trail, parent_trail, pos) in trail_guide {
                pos_map
                    .entry(*parent_trail)
                    .or_default()
                    .push((*pos, *trail));
            }

            // sort positions in ascending order for each trail and push head trail to stack
            pos_map
                .values_mut()
                .for_each(|v| v.sort_unstable_by_key(|(pos, _)| std::cmp::Reverse(*pos)));
            let mut stack: Vec<(usize, usize, usize)> = vec![];
            let mut remaining: Vec<(usize, usize)> = vec![];
            let mut expand = Some((*head_trail, 0));
            while let Some((current_trail, pos)) = expand {
                if let Some(nested_trails) = pos_map.get_mut(&current_trail) {
                    if let Some((insert_pos, trail)) = nested_trails.pop() {
                        if trail == current_trail || cycles_length[trail] == 0 || insert_pos < pos {
                            continue;
                        }
                        stack.push((current_trail, pos, insert_pos));
                        remaining.push((current_trail, insert_pos));
                        remaining.push((trail, 1)); // elipse repeated node
                    } else {
                        stack.push((current_trail, pos, cycles_length[current_trail]));
                    }
                } else {
                    stack.push((current_trail, pos, cycles_length[current_trail]));
                }
                expand = remaining.pop()
            }

            let output_len_bytes = *output_len as u64 * BYTES_U64;
            let output_filename = cache_file_name(
                self.graph.graph_cache.graph_filename.clone(),
                FileType::EulerPath,
                Some(idx),
            )?;
            let output_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .read(true)
                .open(&output_filename)?;
            output_file.set_len(output_len_bytes)?;

            let mut output_mmap = unsafe { MmapOptions::new().map_mut(&output_file)? };
            // in bytes
            let mut write_offset_bytes = 0;

            for (cycle, from, to) in stack.iter() {
                if *to <= *from {
                    continue;
                }
                let (_, t_begin, _) = cycle_offsets[*cycle];
                let begin = t_begin + *from;
                let end = t_begin + *to;

                let write_ptr = match cycles.slice(begin, end) {
                    Some(i) => i.as_ptr(),
                    None => panic!(
                        "error getting write slice for trail {} (from {}, to {})",
                        cycle, begin, end
                    ),
                };
                let write = SharedSlice::<u64>::new(write_ptr, end - begin);
                // in u64 nodes
                let mut pos = 0;
                while let Some(next_slice) = write.slice(pos, pos + 4096) {
                    let byte_len = next_slice.as_bytes().len();
                    unsafe {
                        let dest_ptr = output_mmap.as_mut_ptr().add(write_offset_bytes);
                        let src_ptr = next_slice.as_ptr() as *const u8;
                        std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, byte_len);
                    }
                    write_offset_bytes += byte_len;
                    // Prepare to read next slice
                    pos += next_slice.len();
                }

                output_mmap.flush()?;
            }
        }
        Ok(keys_by_trail_size
            .iter()
            .enumerate()
            .map(|(idx, (_, size))| (idx as u64, *size as u64))
            .collect())
    }

    fn _merge_euler_trails_no_mmap(
        &self,
        cycles: Mutex<Vec<Vec<usize>>>,
    ) -> Result<Vec<(usize, usize)>, Error> {
        let mut trail_heads: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
        let cycles = cycles.into_inner().unwrap();
        cycles.iter().enumerate().for_each(|(idx, trail)| {
            if let Some(first) = trail.first() {
                trail_heads.entry(*first).or_default().push((idx, idx, 0));
            }
        });

        // generate writing sets
        for (cycle_idx, cycle) in cycles.iter().enumerate() {
            for (trail_idx, node) in cycle.iter().enumerate().skip(1) {
                if let Some(head_v) = trail_heads.get_mut(node) {
                    for (vec_idx, (in_cyc, _, _)) in head_v.clone().iter().enumerate() {
                        if *in_cyc == cycle_idx {
                            continue;
                        }
                        head_v[vec_idx] = (*in_cyc, cycle_idx, trail_idx + 1);
                    }
                }
            }
        }

        // break cycles
        let mut v: Vec<_> = trail_heads
            .values_mut()
            .flat_map(|vec| vec.drain(..))
            .collect();
        let mut euler_trail_sets = FindDisjointSetsEulerTrails::new(v.as_mut());
        euler_trail_sets.cycle_check();

        // Union find and write
        let mut trail_sets: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();

        for ((trail, parent_trail, pos), grand_parent) in euler_trail_sets.trails {
            trail_sets
                .entry(grand_parent)
                .or_default()
                .push((trail, parent_trail, pos));
        }

        // Get trail sizes for each soon to be merged trails to establish order
        let mut output_len: HashMap<usize, (usize, usize)> = HashMap::new();
        let cycles_length: Vec<usize> = cycles.iter().map(|x| x.len()).collect();
        for head_trail in trail_sets.keys() {
            output_len.insert(*head_trail, (*head_trail, 1));
            for (trail, _, _) in trail_sets.get(head_trail).unwrap() {
                output_len.get_mut(head_trail).unwrap().1 += cycles_length[*trail] - 1;
            }
            // Adjust output length to size in bytes
            output_len.get_mut(head_trail).unwrap().1 *= std::mem::size_of::<u64>();
        }

        let mut keys_by_trail_size: Vec<(usize, usize)> =
            output_len.values().map(|&(a, b)| (a, b)).collect();
        keys_by_trail_size.sort_unstable_by_key(|(_, s)| std::cmp::Reverse(*s));

        // Write
        for (idx, (head_trail, output_len)) in keys_by_trail_size.iter().enumerate() {
            let trail_guide = trail_sets.get(head_trail).unwrap();
            let mut pos_map: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            let cycles_length: Vec<usize> = cycles.iter().map(|x| x.len()).collect();
            for (trail, parent_trail, pos) in trail_guide {
                pos_map
                    .entry(*parent_trail)
                    .or_default()
                    .push((*pos, *trail));
            }
            pos_map
                .values_mut()
                .for_each(|v| v.sort_unstable_by_key(|(pos, _)| std::cmp::Reverse(*pos)));

            let mut stack: Vec<(usize, usize, usize)> = vec![];
            let mut remaining: Vec<(usize, usize)> = vec![];
            let mut expand = Some((*head_trail, 0));
            while let Some((current_trail, pos)) = expand {
                if let Some(nested_trails) = pos_map.get_mut(&current_trail) {
                    if let Some((insert_pos, trail)) = nested_trails.pop() {
                        if trail == current_trail {
                            continue;
                        }
                        if cycles_length[trail] == 0 {
                            continue;
                        }
                        if insert_pos < pos {
                            continue;
                        }
                        stack.push((current_trail, pos, insert_pos));
                        remaining.push((current_trail, insert_pos));
                        remaining.push((trail, 1)); // elipse repeated node
                    } else {
                        stack.push((current_trail, pos, cycles_length[current_trail]));
                    }
                } else {
                    stack.push((current_trail, pos, cycles_length[current_trail]));
                }
                expand = remaining.pop()
            }

            let output_filename = cache_file_name(
                self.graph.graph_cache.graph_filename.clone(),
                FileType::EulerPath,
                Some(idx),
            )?;
            let output_file = OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .read(true)
                .open(&output_filename)?;
            output_file.set_len(*output_len as u64)?;

            let mut output_mmap = unsafe { MmapOptions::new().map_mut(&output_file)? };
            let mut write_offset = 0;

            for (cycle, from, to) in stack.iter() {
                if *to <= *from {
                    continue;
                }
                let slice = &cycles[*cycle][*from..*to];
                let byte_len = slice.as_bytes().len();
                unsafe {
                    let dest_ptr = output_mmap.as_mut_ptr().add(write_offset);
                    let src_ptr = slice.as_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, byte_len);
                }
                write_offset += byte_len;
            }
            output_mmap.flush()?;
        }

        Ok(keys_by_trail_size
            .iter()
            .enumerate()
            .map(|(idx, (_, size))| (idx, *size))
            .collect())
    }

    fn initialize_hierholzers_procedural_memory(
        &self,
        mmap: u8,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<AtomicUsize>,
            AbstractedProceduralMemoryMut<AtomicU8>,
        ),
        Error,
    > {
        let node_count = self.graph.size() - 1;
        let index_ptr = Arc::new(SharedSlice::<usize>::new(
            self.graph.index.as_ptr() as *const usize,
            self.graph.size(),
        ));
        let template_fn = self.graph.graph_cache.graph_filename.clone();
        let e_fn = cache_file_name(template_fn.clone(), FileType::EulerTmp, Some(1))?;
        let c_fn = cache_file_name(template_fn.clone(), FileType::EulerTmp, Some(2))?;

        let edges = SharedSliceMut::<AtomicUsize>::abst_mem_mut(e_fn, node_count, mmap > 0)?;
        let count = SharedSliceMut::<AtomicU8>::abst_mem_mut(c_fn, node_count, mmap > 1)?;

        thread::scope(|scope| {
            let threads = self.graph.thread_count as usize;
            for i in 0..threads {
                let index = Arc::clone(&index_ptr);
                let edges = &edges;
                let count = &count;
                let begin = node_count / threads * i;
                let end = if i == threads - 1 {
                    node_count
                } else {
                    node_count / threads * (i + 1)
                };
                scope.spawn(move |_| {
                    for k in begin..end {
                        edges.get(k).store(*index.get(k), Ordering::Relaxed);
                        count
                            .get(k)
                            .store((*index.get(k + 1) - *index.get(k)) as u8, Ordering::Relaxed);
                    }
                });
            }
        })
        .unwrap();
        edges.flush()?;
        count.flush()?;
        Ok((edges, count))
    }

    /// find Eulerian cycle and write sequence of node ids to memory-mapped file.
    /// num_threads controls parallelism level (defaults to 1, single-threaded).
    /// returns vec of (euler path file sequence number, file size(vytes)).
    pub fn find_eulerian_cycle(&self, mmap: u8) -> Result<Vec<(u64, u64)>, Error> {
        let node_count = match self.graph.size() - 1 {
            0 => panic!("Graph has no vertices"),
            i => i,
        };
        let graph_ptr = Arc::new(SharedSlice::<Edge>::new(
            self.graph.graph.as_ptr() as *const Edge,
            self.graph.width(),
        ));

        // The Vec<_> and MmapMut refs need to be in scope for the structures not to be deallocated
        let (edges, edge_count) = self.initialize_hierholzers_procedural_memory(mmap)?;

        // Atomic counter to pick next starting vertex for a new cycle
        let start_vertex_counter = Arc::new(AtomicUsize::new(0));
        // mmap to store disjoined trails for subsequent merging
        let filename = cache_file_name(
            self.graph.graph_cache.graph_filename.clone(),
            FileType::EulerTmp,
            None,
        )?;
        let (mmap_slice, mmap) = Self::create_memmapped_mut_slice_from_tmp_file::<usize>(
            filename,
            self.graph.width() * 2,
        )?;
        let mmap_mutex = Mutex::new(mmap_slice);

        let cycle_offsets: Mutex<Vec<(usize, usize, usize)>> = std::sync::Mutex::new(vec![]);
        let mmap_offset: Mutex<usize> = Mutex::new(0);

        thread::scope(|scope| {
            for _ in 0..self.graph.thread_count {
                let graph = Arc::clone(&graph_ptr);
                let next_edge = &edges;
                let edge_count = &edge_count;
                let start_vertex_counter = Arc::clone(&start_vertex_counter);
                let cycle_offsets = &cycle_offsets;
                let mmap_offset = &mmap_offset;
                let mmap = &mmap_mutex;

                // Spawn a thread
                scope.spawn(move |_| {
                    // find cycles until no unused edges remain
                    loop {
                        let start_v = loop {
                            let idx = start_vertex_counter
                                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                                    if x >= node_count {
                                        Some(0)
                                    } else {
                                        Some(x + 1)
                                    }
                                })
                                .unwrap_or(0);
                            if idx >= node_count {
                                break None;
                            }
                            if edge_count.get(idx).load(Ordering::Relaxed) > 0 {
                                break Some(idx);
                            }
                        };
                        let start_v = match start_v {
                            Some(v) => v,
                            None => {
                                break;
                            }
                        };

                        // Hierholzer's DFS
                        let mut stack: Vec<usize> = Vec::new();
                        let mut cycle: Vec<usize> = Vec::new();
                        stack.push(start_v);
                        while let Some(&v) = stack.last() {
                            if edge_count
                                .get(v)
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                                    if x > 0 {
                                        Some(x - 1)
                                    } else {
                                        None // don't update
                                    }
                                })
                                .is_ok()
                            {
                                let edge_idx = next_edge.get(v).fetch_add(1, Ordering::Relaxed);
                                stack.push(graph.get(edge_idx).dest());
                            } else {
                                stack.pop();
                                if cycle.is_empty() {
                                    // check if error of atomic
                                    if stack.is_empty() {
                                        continue;
                                    }
                                }
                                cycle.push(v);
                            }
                        }
                        if !cycle.is_empty() {
                            cycle.reverse();
                            let cycle = cycle.as_slice();
                            let mut cycle_stack = cycle_offsets.lock().unwrap();
                            let mut offset = mmap_offset.lock().unwrap();
                            let begin = *offset;
                            *offset += cycle.len();
                            let end = *offset;
                            let cycle_id = cycle_stack.len();
                            cycle_stack.push((cycle_id, begin, end));

                            // Write to mmap, this may not done concurrently
                            let mut mmap_guard = match mmap.lock() {
                                Ok(i) => i,
                                Err(e) => panic!("error mutex 1: {:?}", e),
                            };
                            let () = match mmap_guard.mut_slice(begin, end) {
                                Some(i) => i.copy_from_slice(cycle),
                                None => panic!("error couldn't slice mmap to write cycle"),
                            };
                        }
                    }
                });
            }
        })
        .unwrap();
        mmap.flush()?;
        let disjoint_cycles = cycle_offsets.into_inner().unwrap();
        // Euler trails are in the mem_mapped file
        self.merge_euler_trails(disjoint_cycles)
    }
    /// find Eulerian cycle and write sequence of node ids to memory-mapped file.
    /// num_threads controls parallelism level (defaults to 1, single-threaded).
    /// returns vec of (euler path file sequence number, file size(vytes)).
    pub fn _find_eulerian_cycle_no_mmap(&self) -> Result<Vec<(usize, usize)>, Error> {
        let node_count = match self.graph.size() - 1 {
            0 => panic!("Graph has no vertices"),
            i => i,
        };

        let index_ptr =
            SharedSlice::<usize>::new(self.graph.index.as_ptr() as *const usize, self.graph.size());
        let graph_ptr = Arc::new(SharedSlice::<Edge>::new(
            self.graph.graph.as_ptr() as *const Edge,
            self.graph.width(),
        ));

        // atomic array for next unused edge index for each node and unread edge count
        let mut next_edge_vec: Vec<AtomicUsize> = Vec::with_capacity(node_count);
        next_edge_vec.resize_with(node_count, || AtomicUsize::new(0));

        let mut edge_count: Vec<AtomicU8> = Vec::with_capacity(node_count);
        edge_count.resize_with(node_count, || AtomicU8::new(0));

        for i in 0..node_count {
            next_edge_vec[i].store(*index_ptr.get(i), Ordering::Relaxed);
            edge_count[i].store(
                (*index_ptr.get(i + 1) - *index_ptr.get(i)) as u8,
                Ordering::Relaxed,
            );
        }
        let next_edge = Arc::new(next_edge_vec);
        // mutex is needed to make sure a read isn't made between a successful check and a decrease
        let edge_count = Arc::new(edge_count);

        // Atomic counter to pick next starting vertex for a new cycle
        let start_vertex_counter = Arc::new(AtomicUsize::new(0));
        let cycles_mutex = std::sync::Mutex::new(Vec::new());

        thread::scope(|scope| {
            for _ in 0..self.graph.thread_count {
                let graph_ptr = Arc::clone(&graph_ptr);
                let next_edge = Arc::clone(&next_edge);

                let edge_count = Arc::clone(&edge_count);
                let start_vertex_counter = Arc::clone(&start_vertex_counter);
                let cycles_mutex = &cycles_mutex;

                // Spawn a thread
                scope.spawn(move |_| {
                    let mut local_cycles: Vec<Vec<usize>> = Vec::new();
                    // find cycles until no unused edges remain
                    loop {
                        let start_v = loop {
                            let idx = start_vertex_counter
                                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                                    if x >= node_count {
                                        Some(0)
                                    } else {
                                        Some(x + 1)
                                    }
                                })
                                .unwrap_or(0);
                            if idx >= node_count {
                                break None;
                            }
                            if edge_count[idx].load(Ordering::Relaxed) > 0 {
                                break Some(idx);
                            }
                        };
                        let start_v = match start_v {
                            Some(v) => v,
                            None => {
                                // all edges used
                                break;
                            }
                        };

                        // Hierholzer's DFS
                        let mut stack: Vec<usize> = Vec::new();
                        let mut cycle: Vec<usize> = Vec::new();
                        stack.push(start_v);
                        while let Some(&v) = stack.last() {
                            // get next unused edge from v
                            if edge_count[v]
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                                    if x > 0 {
                                        Some(x - 1)
                                    } else {
                                        None // don't update
                                    }
                                })
                                .is_ok()
                            {
                                let edge_idx = next_edge[v].fetch_add(1, Ordering::Relaxed);
                                stack.push(graph_ptr.get(edge_idx).dest());
                            } else {
                                stack.pop();
                                if cycle.is_empty() {
                                    // check if error of atomic
                                    if stack.is_empty() {
                                        continue;
                                    }
                                }
                                cycle.push(v);
                            }
                        }
                        if !cycle.is_empty() {
                            cycle.reverse();
                            local_cycles.push(cycle.clone());
                        }
                    }
                    let mut global_cycles = cycles_mutex.lock().unwrap();
                    global_cycles.extend(local_cycles);
                });
            }
        })
        .unwrap();
        self._merge_euler_trails_no_mmap(cycles_mutex)
    }
}

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

#[derive(Clone)]
pub struct HKRelax<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: GraphMemoryMap<EdgeType, Edge>,
    pub n: usize,
    pub t: f64,
    pub eps: f64,
    pub seed: Vec<usize>,
    pub psis: Vec<f64>,
}

impl<EdgeType, Edge> HKRelax<EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    #[inline(always)]
    fn f64_is_nomal(val: f64, op_description: &str) -> Result<f64, Error> {
        if !val.is_normal() {
            panic!(
                "error hk-relax abnormal value at {} = {}",
                op_description, val
            );
        }
        Ok(val)
    }

    fn f64_to_usize_safe(x: f64) -> Option<usize> {
        if x.is_normal() && x > 0f64 && x <= usize::MAX as f64 {
            Some(x as usize) // truncates toward zero
        } else {
            None
        }
    }

    fn evaluate_params(
        graph: GraphMemoryMap<EdgeType, Edge>,
        t: f64,
        eps: f64,
        seed: Vec<usize>,
    ) -> Result<(), Error> {
        let node_count = match graph.size().overflowing_sub(1) {
            (_, true) => {
                panic!("error hk-relax invalid parameters: |V| == 0, the graph is empty");
            }
            (i, _) => {
                if i == 0 {
                    panic!(
                        "error hk-relax invalid parameters: actual |V| == 0, the graph is empty"
                    );
                }
                i
            }
        };
        if !t.is_normal() || t <= 0f64 {
            panic!(
                "error hk-relax invalid parameters: t == {} doesn't satisfy t > 0.0",
                t
            );
        }
        if !eps.is_normal() || eps <= 0f64 || eps >= 1f64 {
            panic!(
                "error hk-relax invalid parameters: ε == {} doesn't satisfy 0.0 < ε 1.0",
                eps
            );
        }
        if seed.is_empty() {
            panic!(
                "error hk-relax invalid parameters: seed_nodes.len() == 0, please provide at least one seed node"
            );
        }
        for (idx, seed_node) in seed.iter().enumerate() {
            if *seed_node > node_count - 1 {
                panic!(
                    "error hk-relax invalid parameters: id(seed_nodes[{}]) == {} but max_id(v in V) == {}",
                    idx,
                    seed_node,
                    node_count - 1
                );
            }
        }
        Ok(())
    }

    fn compute_psis(n: usize, t: f64) -> Result<Vec<f64>, Error> {
        let mut psis = vec![0f64; n + 1];
        psis[n] = 1f64;
        for i in (0..n).rev() {
            psis[i] = Self::f64_is_nomal(
                psis[i + 1] * t / (i as f64 + 1f64) + 1f64,
                format!(
                    "{{at round: {}}} (psis[{} + 1] * t / (i + 1) + 1)",
                    n - i + 1,
                    i
                )
                .as_str(),
            )?;
        }
        Ok(psis)
    }

    fn compute_n(t: f64, eps: f64) -> Result<usize, Error> {
        let bound = Self::f64_is_nomal(eps / 2f64, "ε/2")?;

        let n_plus_two_i32 = match i32::try_from(t.floor() as i64) {
            Ok(n) => match n.overflowing_add(1) {
                (r, false) => r,
                (_, true) => panic!(
                    "error computing n + 2 for hk-relax (n + 1 = {}) + 1 overflowed",
                    n
                ),
            },
            Err(e) => panic!(
                "error computing n for hk-relax couldn't cast t: {} to i32: {}",
                t, e
            ),
        };

        let mut t_power_n_plus_one = Self::f64_is_nomal(t.powi(n_plus_two_i32 - 1), "t^(n + 1)")?;

        let mut n_plus_one_fac = match (2..n_plus_two_i32).try_fold(1.0_f64, |acc, x| {
            let res = acc * (x as f64);
            if res.is_finite() { Some(res) } else { None }
        }) {
            Some(fac) => Self::f64_is_nomal(fac, "(n + 1)!")?,
            None => panic!("error computing n for hk-relax overflowed trying to compute (n + 1)!"),
        };

        let mut n_plus_two = n_plus_two_i32 as f64;

        while (t_power_n_plus_one * n_plus_two / n_plus_one_fac / (n_plus_two - t)) >= bound {
            t_power_n_plus_one = Self::f64_is_nomal(t_power_n_plus_one * t, "t^(n + 1)")?;
            n_plus_one_fac = Self::f64_is_nomal(n_plus_one_fac * n_plus_two, "(n + 1)!")?;
            n_plus_two += 1f64;
        }

        // convert the f64 value to usize subtract 2 and output n
        // FIXME: are there other methods to validate n?
        match Self::f64_to_usize_safe(n_plus_two) {
            Some(n_plus_two_usize) => match n_plus_two_usize.overflowing_sub(2) {
                (n, false) => Ok(n),
                (_, true) => panic!(
                    "error computing n for hk-relax overflowed trying to compute {} - 2",
                    n_plus_two_usize
                ),
            },
            None => panic!(
                "error computing n for hk-relax can't convert {} to usize",
                n_plus_two
            ),
        }
    }

    pub fn new(
        graph: GraphMemoryMap<EdgeType, Edge>,
        t: f64,
        eps: f64,
        seed: Vec<usize>,
    ) -> Result<Self, Error> {
        let () = match Self::evaluate_params(graph.clone(), t, eps, seed.clone()) {
            Ok(_) => {}
            Err(e) => panic!("error creating HKRelax instance: {}", e),
        };
        let n = Self::compute_n(t, eps)?;
        println!("n computed to be {}", n);
        Ok(HKRelax {
            graph,
            n,
            t,
            eps,
            seed,
            psis: Self::compute_psis(n, t)?,
        })
    }

    /// receives an instance of HKRelax for a given graph and parameters for t, epsilon and a
    /// vector of seed nodes.
    /// returns an instance of HKRelax for the same graph with parameters adjusted accordingly
    pub fn _adjust_parameters(&self, t: f64, eps: f64, seed: Vec<usize>) -> Result<Self, Error> {
        let () = match Self::evaluate_params(self.graph.clone(), t, eps, seed.clone()) {
            Ok(_) => {}
            Err(e) => panic!("error creating HKRelax instance: {}", e),
        };
        let n = Self::compute_n(t, eps)?;
        Ok(HKRelax {
            graph: self.graph.clone(),
            n,
            t,
            eps,
            seed,
            psis: Self::compute_psis(n, t)?,
        })
    }

    pub fn compute(&self) -> Result<Community<usize>, Error> {
        let n = self.n as f64;
        let threshold_pre_u_pre_j = Self::f64_is_nomal(self.t.exp() * self.eps / n, "e^t * ε / n")?;
        let mut x: HashMap<usize, f64> = HashMap::new();
        let mut r: HashMap<(usize, usize), f64> = HashMap::new();
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

        let seed_len_f64 = self.seed.clone().len() as f64;
        for seed_node in self.seed.iter() {
            if r.contains_key(&(*seed_node, 0usize)) {
                panic!(
                    "error hk-relax {} seed node is present multiple times in seed array",
                    seed_node
                );
            }
            r.insert((*seed_node, 0usize), 1f64 / seed_len_f64);
            queue.push_back((*seed_node, 0));
        }

        while let Some((v, j)) = queue.pop_front() {
            let rvj = match r.get(&(v, j)) {
                Some(i) => *i,
                None => panic!(
                    "error hk-relax ({}, {}) present in queue but not in residual",
                    v, j
                ),
            };

            // x[v] += rvj && check if r[(v, j)] is normal
            match x.get_mut(&v) {
                Some(x_v) => {
                    *x_v = Self::f64_is_nomal(*x_v + rvj, "x[v] + r([v, j])")?;
                }
                None => {
                    x.insert(v, Self::f64_is_nomal(rvj, "r([v, j])")?);
                }
            };

            // r[(v, j)] = 0
            match r.get_mut(&(v, j)) {
                Some(rvj) => *rvj = 0f64,
                None => panic!("error hk-relax r({}, {}) Some and None", v, j),
            };

            //  mass = (t*rvj/(float(j)+1.))/len(G[v])
            let (deg_v, v_n) = match self.graph.neighbours(v) {
                Ok(v_n) => (v_n.remaining_neighbours() as f64, v_n),
                Err(e) => panic!("error hk-relax getting neighbours of {}: {}", v, e),
            };

            // result checked when node is popped from queue
            let mass = self.t * rvj / (j as f64 + 1f64) / deg_v;

            for u in v_n {
                let u = u.dest();
                let next = (u, j + 1);
                if j + 1 == self.n {
                    match x.get_mut(&u) {
                        Some(x_u) => {
                            *x_u = Self::f64_is_nomal(
                                *x_u + rvj / deg_v,
                                "x[u] + (r[(v, j)] / deg_v)",
                            )?
                        }
                        None => {
                            x.insert(u, Self::f64_is_nomal(rvj / deg_v, "r[(v, j)] / deg_v")?);
                        }
                    };
                    continue;
                }
                let r_next = match r.get_mut(&next) {
                    Some(r_next) => r_next,
                    None => {
                        r.insert(next, 0f64);
                        match r.get_mut(&next) {
                            Some(r_next) => r_next,
                            None => panic!(
                                "error hk-relax just inserted ({}, {}) into residual and got None",
                                u,
                                j + 1
                            ),
                        }
                    }
                };
                let deg_u = self.graph.node_degree(u) as f64;
                let threshold = Self::f64_is_nomal(
                    threshold_pre_u_pre_j * deg_u / self.psis[j + 1],
                    "e^t * ε * deg_u / (n * psis[j + 1])",
                )?;

                if *r_next < threshold && *r_next + mass >= threshold {
                    queue.push_back(next);
                }
                *r_next += mass;
            }
        }

        let mut h: Vec<(usize, f64)> = x
            .keys()
            .map(|v| (*v, x.get(v).unwrap() / self.graph.node_degree(*v) as f64))
            .collect::<Vec<(usize, f64)>>();

        match self
            .graph
            .sweep_cut_over_diffusion_vector_by_conductance(h.as_mut())
        {
            Ok(c) => {
                println!(
                    "best community {{\n\tsize: {}\n\tvolume/width: {}\n\tconductance: {}\n}}",
                    c.size, c.width, c.conductance
                );
                Ok(c)
            }
            Err(e) => panic!("error: {}", e),
        }
    }
}

#[expect(dead_code)]
#[derive(Clone)]
pub struct ApproxDirHKPR<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: GraphMemoryMap<EdgeType, Edge>,
    pub t: f64,
    pub eps: f64,
    pub seed: usize,
    pub target_size: usize,
    pub target_vol: usize,
    pub target_conductance: f64,
}

// #[expect(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum ApproxDirichletHeatKernelK {
    None,
    Mean,
    Unlim,
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> ApproxDirHKPR<EdgeType, Edge> {
    #[inline(always)]
    fn f64_is_nomal(val: f64, op_description: &str) -> Result<f64, Error> {
        if !val.is_normal() {
            panic!(
                "error hk-relax abnormal value at {} = {}",
                op_description, val
            );
        }
        Ok(val)
    }

    fn f64_to_usize_safe(x: f64) -> Option<usize> {
        if x.is_normal() && x > 0f64 && x <= usize::MAX as f64 {
            Some(x as usize) // truncates toward zero
        } else {
            None
        }
    }

    fn evaluate_params(
        graph: GraphMemoryMap<EdgeType, Edge>,
        seed_node: usize,
        eps: f64,
        _target_size: usize,
        _target_vol: usize,
        target_conductance: f64,
    ) -> Result<(), Error> {
        let node_count = match graph.size().overflowing_sub(1) {
            (_, true) => {
                panic!("error hk-relax invalid parameters: |V| == 0, the graph is empty");
            }
            (i, _) => {
                if i == 0 {
                    panic!(
                        "error hk-relax invalid parameters: actual |V| == 0, the graph is empty"
                    );
                }
                i
            }
        };
        if !eps.is_normal() || eps <= 0f64 || eps >= 1f64 {
            panic!(
                "error hk-relax invalid parameters: ε == {} doesn't satisfy 0.0 < ε 1.0",
                eps
            );
        }
        if !target_conductance.is_normal()
            || target_conductance <= 0f64
            || target_conductance >= 1f64
        {
            panic!(
                "error hk-relax invalid parameters: target_conductance == {} doesn't satisfy 0.0 < target_conductance < 1.0",
                target_conductance
            );
        }
        if seed_node > node_count - 1 {
            panic!(
                "error hk-relax invalid parameters: id(seed_nodes) == {} but max_id(v in V) == {}",
                seed_node,
                node_count - 1
            );
        }
        Ok(())
    }

    pub fn new(
        graph: GraphMemoryMap<EdgeType, Edge>,
        eps: f64,
        seed: usize,
        target_size: usize,
        target_vol: usize,
        target_conductance: f64,
    ) -> Result<Self, Error> {
        let () = Self::evaluate_params(
            graph.clone(),
            seed,
            eps,
            target_size,
            target_vol,
            target_conductance,
        )?;
        let t_formula = "(1. / target_conductance) * ln((2. * sqrt(target_vol)) / (1. - ε) + 2. * ε * target_size)";
        let t = Self::f64_is_nomal(
            (1. / target_conductance)
                * f64::ln(
                    (2. * f64::sqrt(target_vol as f64)) / (1. - eps)
                        + 2. * eps * target_size as f64,
                ),
            t_formula,
        )?;
        Ok(ApproxDirHKPR {
            graph,
            t,
            eps,
            seed,
            target_size,
            target_vol,
            target_conductance,
        })
    }

    fn random_sample_poisson(lambda: f64, n: usize) -> Result<Vec<OrderedFloat<f64>>, Error> {
        let mut rng = rand::rng();
        let poisson = match Poisson::new(lambda) {
            Ok(dist) => dist,
            Err(e) => panic!(
                "error approx-dirchlet-hk couldn't sample poission distribution: {}",
                e
            ),
        };
        Ok(poisson
            .sample_iter(&mut rng)
            .take(n)
            .map(OrderedFloat)
            .collect())
    }

    #[inline(always)]
    fn random_neighbour(&self, deg_u: usize, u_n: NeighbourIter<EdgeType, Edge>) -> usize {
        let inv_deg_u = 1. / deg_u as f64;
        let random: f64 = rand::rng().random();
        let mut idx_plus_one = 1f64;
        for v in u_n {
            if idx_plus_one * inv_deg_u > random {
                return v.dest();
            }
            idx_plus_one += idx_plus_one;
        }
        panic!("error approx-dirchlet-hk didn't find random neighbour");
    }

    fn random_walk_seed(&self, k: usize, seed_node: usize) -> usize {
        let mut curr_node = seed_node;
        for _ in 0..k {
            let (deg_u, u_n) = match self.graph.neighbours(curr_node) {
                Ok(u_n) => (u_n.remaining_neighbours(), u_n),
                Err(e) => panic!(
                    "error approx-dirchlet-hk couldn't get neighbours for {}: {}",
                    curr_node, e
                ),
            };
            if deg_u > 0 {
                curr_node = self.random_neighbour(deg_u, u_n);
            } else {
                return curr_node;
            }
        }
        curr_node
    }

    #[allow(clippy::unreachable)]
    pub fn compute(&self, big_k: ApproxDirichletHeatKernelK) -> Result<Community<usize>, Error> {
        let node_count = match self.graph.size().overflowing_sub(1) {
            (_, true) => panic!("error approx-dirchlet-hk |V| + 1 == 0"),
            (0, _) => panic!("error approx-dirchlet-hk |V| == 0"),
            (i, _) => i as f64,
        };

        let r = Self::f64_is_nomal(
            (16. / self.eps.powi(2)) * f64::ln(node_count),
            "(16.0 / ε²) * ln(|V|)",
        )?;
        let one_over_r = Self::f64_is_nomal(1. / r, "1. / r")?;

        let k = match big_k {
            ApproxDirichletHeatKernelK::None => Self::f64_is_nomal(
                2. * f64::ln(1. / self.eps) / (f64::ln(f64::ln(1. / self.eps))),
                "2. * ln(1. / ε) / ln(ln(1. / ε))",
            )?,
            ApproxDirichletHeatKernelK::Mean => Self::f64_is_nomal(2. * self.t, "2. * t")?,
            ApproxDirichletHeatKernelK::Unlim => f64::infinity(),
            #[expect(unreachable_patterns)]
            _ => panic!("error approx-dirchlet-hk unknown K {}", big_k),
        };
        let k = OrderedFloat(k);
        println!(
            "k (ceil on sample value) computed to be {}\nr (number of samples) computed to be {}",
            k, r
        );

        let num_samples: usize = match Self::f64_to_usize_safe(r) {
            Some(s) => s,
            None => panic!("error approx-dirchlet-hk couldn't cast {} to usize", r),
        };
        let steps: Vec<OrderedFloat<f64>> = Self::random_sample_poisson(self.t, num_samples)?;
        let mut aprox_hkpr_samples: HashMap<usize, f64> = HashMap::new();

        for little_k in steps {
            let OrderedFloat(little_k) = std::cmp::min(little_k, k);
            let little_k_usize = match Self::f64_to_usize_safe(little_k) {
                Some(val) => val,
                None => panic!(
                    "error approx-dirchlet-hk couldn't cast {} to usize",
                    little_k
                ),
            };
            let v = self.random_walk_seed(little_k_usize, self.seed);
            match aprox_hkpr_samples.get_mut(&v) {
                Some(v) => *v += one_over_r,
                None => {
                    aprox_hkpr_samples.insert(v, one_over_r);
                }
            };
        }

        // FIXME: Never normalized
        let mut p: Vec<(usize, f64)> = aprox_hkpr_samples
            .keys()
            .map(|u| {
                (
                    *u,
                    *aprox_hkpr_samples.get(u).unwrap() / self.graph.node_degree(*u) as f64,
                )
            })
            .collect::<Vec<(usize, f64)>>();

        match self
            .graph
            .sweep_cut_over_diffusion_vector_by_conductance(p.as_mut())
        {
            Ok(c) => {
                println!(
                    "best community {{\n\tsize: {}\n\tvolume/width: {}\n\tconductance: {}\n}}",
                    c.size, c.width, c.conductance
                );
                Ok(c)
            }
            Err(e) => panic!("error: {}", e),
        }
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

impl<EdgeType, Edge> GraphMemoryMap<EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    pub fn init(
        cache: GraphCache<EdgeType, Edge>,
        thread_count: u8,
    ) -> Result<GraphMemoryMap<EdgeType, Edge>, Error> {
        if cache.readonly {
            let mmap = unsafe {
                match MmapOptions::new().map(&File::open(&cache.kmer_filename)?) {
                    Ok(i) => i,
                    Err(e) => panic!("error couldn't mmap k-mer fst: {}", e),
                }
            };
            let thread_count = if thread_count == 0 { 1 } else { thread_count };
            return Ok(GraphMemoryMap {
                graph: unsafe { Arc::new(Mmap::map(&cache.graph_file)?) },
                index: unsafe { Arc::new(Mmap::map(&cache.index_file)?) },
                kmers: match Map::new(mmap) {
                    Ok(i) => Arc::new(i),
                    Err(e) => panic!("error couldn't map k-mer fst mmap: {}", e),
                },
                graph_cache: cache,
                edge_size: std::mem::size_of::<Edge>(),
                thread_count,
                exports: 0,
            });
        }

        panic!("error graph cache must be readonly to be memmapped");
    }

    #[inline(always)]
    pub fn node_degree(&self, node_id: usize) -> usize {
        unsafe {
            let ptr = (self.index.as_ptr() as *const usize).add(node_id);
            let begin = ptr.read_unaligned();
            ptr.add(1).read_unaligned() - begin
        }
    }

    #[inline(always)]
    pub fn node_id_from_kmer(&self, kmer: &str) -> Result<u64, Error> {
        if let Some(val) = self.kmers.get(kmer) {
            Ok(val)
        } else {
            Err(Error::new(
                std::io::ErrorKind::Other,
                format!("error k-mer {kmer} not found"),
            ))
        }
    }

    #[inline(always)]
    pub fn index_node(&self, node_id: usize) -> std::ops::Range<usize> {
        unsafe {
            let ptr = (self.index.as_ptr() as *const usize).add(node_id);
            ptr.read_unaligned()..ptr.add(1).read_unaligned()
        }
    }

    pub fn neighbours(&self, node_id: usize) -> Result<NeighbourIter<EdgeType, Edge>, Error> {
        if node_id >= self.size() {
            panic!("error invalid range");
        }

        Ok(NeighbourIter::<EdgeType, Edge>::new(
            self.graph.as_ptr() as *const Edge,
            self.index.as_ptr() as *const usize,
            node_id,
        ))
    }

    pub fn edges(&self) -> Result<EdgeIter<EdgeType, Edge>, Error> {
        Ok(EdgeIter::<EdgeType, Edge>::new(
            self.graph.as_ptr() as *const Edge,
            self.index.as_ptr() as *const usize,
            0,
            self.size(),
        ))
    }

    pub fn _edges_in_range(
        &self,
        start_node: usize,
        end_node: usize,
    ) -> Result<EdgeIter<EdgeType, Edge>, Error> {
        if start_node > end_node {
            panic!("error invalid range, beginning after end");
        }
        if start_node > self.size() || end_node > self.size() {
            panic!("error invalid range");
        }

        Ok(EdgeIter::<EdgeType, Edge>::new(
            self.graph.as_ptr() as *const Edge,
            self.index.as_ptr() as *const usize,
            start_node,
            end_node,
        ))
    }

    fn sweep_cut_over_diffusion_vector_by_conductance(
        &self,
        diffusion: &mut [(usize, f64)],
    ) -> Result<Community<usize>, Error> {
        diffusion.sort_unstable_by_key(|(_, mass)| std::cmp::Reverse(OrderedFloat(*mass)));
        // debug
        // println!("descending-ordered mass vector {:?}", s);

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
                        (_, true) => panic!(
                            "error hk-relax overflow_add in vol_s sweep cut at node {}",
                            u
                        ),
                    };
                    vol_v_minus_s = match vol_v_minus_s.overflowing_sub(u_n.remaining_neighbours())
                    {
                        (r, false) => r,
                        (_, true) => panic!(
                            "error hk-relax overflow_add in vol_v_minus_s sweep cut at node {}",
                            u
                        ),
                    };
                    u_n
                }
                Err(e) => panic!(
                    "error hk-relax sweep cut couldn't get {} neighbours: {}",
                    u, e
                ),
            };
            match community.get(u) {
                Some(_) => panic!(
                    "error building best community from diffusion vector: {} is present multiple times",
                    u
                ),
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
                        (_, true) => panic!(
                            "error hk-relax overflow_sub in cut_s sweep cut at node {} in neighbour {}",
                            u,
                            v.dest()
                        ),
                    };
                } else {
                    cut_s = match cut_s.overflowing_add(1) {
                        (r, false) => r,
                        (_, true) => panic!(
                            "error hk-relax overflow_add in sweep cut at node {} in neighbour {}",
                            u,
                            v.dest()
                        ),
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
        }

        Ok(Community {
            nodes: best_community,
            size: best_size,
            width: best_width,
            conductance: best_conductance,
        })
    }

    #[expect(dead_code)]
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

    /// returns number of entries in node index file (== |V| + 1)
    pub fn size(&self) -> usize {
        self.graph_cache.index_bytes // num nodes
    }

    /// returns number of entries in edge file (== |E|)
    pub fn width(&self) -> usize {
        self.graph_cache.graph_bytes // num edges
    }

    fn get_exports_add_one(&mut self) -> Result<u8, Error> {
        self.exports = match self.exports.overflowing_add(1) {
            (r, false) => r,
            (_, true) => {
                self.exports = u8::MAX;
                panic!(
                    "error overflowed export count var in graph struct, please provide an identifier for your export"
                )
            }
        };
        Ok(self.exports - 1)
    }

    pub fn apply_mask_to_nodes(
        &self,
        mask: fn(usize) -> bool,
        identifier: Option<String>,
    ) -> Result<GraphMemoryMap<EdgeType, Edge>, Error> {
        let node_count = self.size();
        if node_count < 2 {
            panic!("error can't mask empty graph");
        }
        let c_fn = CACHE_DIR.to_string() + "edge_count.tmp";
        let i_fn = CACHE_DIR.to_string() + "node_index.tmp";
        let mut edge_count = SharedSliceMut::<usize>::abst_mem_mut(c_fn, node_count, true)?;
        let mut node_index = SharedSliceMut::<usize>::abst_mem_mut(i_fn, node_count - 1, true)?;

        let mut curr_node_index: usize = 0;
        let mut curr_edge_count: usize = 0;
        *edge_count.get_mut(0) = curr_edge_count;
        for u in 0..node_count {
            if mask(u) {
                *node_index.get_mut(u) = curr_node_index;
                curr_node_index += 1;
                let neighbours = self
                    .neighbours(u)?
                    .into_iter()
                    .filter(|x| mask(x.dest()))
                    .count();
                curr_edge_count += neighbours;
                *edge_count.get_mut(u + 1) = curr_edge_count;
            } else {
                *node_index.get_mut(u) = usize::MAX;
                *edge_count.get_mut(u + 1) = curr_edge_count;
            }
        }

        let mut kmer_stream = self.kmers.stream();
        let graph_id = match identifier {
            Some(id) => id,
            None => cache_file_name(
                self.graph_cache.graph_filename.clone(),
                FileType::ExportGraphMask,
                Some(self.clone().get_exports_add_one()? as usize),
            )?,
        };

        let edges_fn = cache_file_name(graph_id.clone(), FileType::Edges, None)?;
        let index_fn = cache_file_name(graph_id.clone(), FileType::Index, None)?;
        let kmers_fn = cache_file_name(graph_id.clone(), FileType::Fst, None)?;
        let mut edges = SharedSliceMut::<Edge>::abst_mem_mut(edges_fn, curr_edge_count, true)?;
        let mut index = SharedSliceMut::<usize>::abst_mem_mut(index_fn, curr_node_index + 1, true)?;
        let kmer_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(kmers_fn.clone())?;
        let mut build = match MapBuilder::new(&kmer_file) {
            Ok(i) => i,
            Err(e) => panic!("error couldn't initialize builder: {}", e),
        };

        *index.get_mut(0) = 0;
        while let Some((kmer, node_id)) = kmer_stream.next() {
            let id = node_id as usize;
            if mask(id) {
                // write index file for next node (id + 1)
                let new_id = *node_index.get(id);
                *index.get_mut(new_id + 1) = *edge_count.get(id + 1);
                // write edge file node
                let edge_begin = *edge_count.get(id);
                let node_edges: Vec<Edge> =
                    self.neighbours(id)?.filter(|x| mask(x.dest())).collect();
                edges
                    .write_slice(edge_begin, node_edges.as_slice())
                    .ok_or_else(|| {
                        Error::new(
                            std::io::ErrorKind::Other,
                            "error writing edges for node {id}",
                        )
                    })?;
                // write fst for node
                match build.insert(kmer, new_id as u64) {
                    Ok(i) => i,
                    Err(e) => panic!("error couldn't insert kmer for node (id {new_id}): {e}"),
                };
            }
        }
        let cache: GraphCache<EdgeType, Edge> = GraphCache::open(kmers_fn)?;
        GraphMemoryMap::init(cache, self.thread_count)
    }

    #[expect(dead_code)]
    pub fn export_petgraph(&self) -> Result<DiGraph<NodeIndex<usize>, EdgeType>, Error> {
        let mut graph = DiGraph::<NodeIndex<usize>, EdgeType>::new();
        let node_count = self.size() - 1;

        (0..node_count).for_each(|u| {
            graph.add_node(NodeIndex::new(u));
        });
        (0..node_count)
            .map(|u| match self.neighbours(u) {
                Ok(neighbours_of_u) => (u, neighbours_of_u),
                Err(e) => panic!(
                    "while exporting petgraph `DiGraphMap`, error getting neighbours of {u}: {e}",
                ),
            })
            .for_each(|(u, u_n)| {
                u_n.for_each(|v| {
                    graph.add_edge(NodeIndex::new(u), NodeIndex::new(v.dest()), v.e_type());
                });
            });

        println!("{:?}", graph);

        Ok(graph)
    }

    fn init_procedural_memory_bz(&self, mmap: u8) -> Result<ProceduralMemoryBZ, Error> {
        let node_count = self.size() - 1;

        let template_fn = self.graph_cache.graph_filename.clone();
        let d_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(0))?;
        let n_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(1))?;
        let c_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(2))?;
        let p_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(3))?;

        let degree = SharedSliceMut::<u8>::abst_mem_mut(d_fn, node_count, mmap > 0)?;
        let node = SharedSliceMut::<usize>::abst_mem_mut(n_fn, node_count, mmap > 1)?;
        let core = SharedSliceMut::<u8>::abst_mem_mut(c_fn, node_count, mmap > 2)?;
        let pos = SharedSliceMut::<usize>::abst_mem_mut(p_fn, node_count, mmap > 3)?;

        Ok((degree, node, core, pos))
    }

    pub fn compute_k_core_bz(&self, mmap: u8) -> Result<String, Error> {
        let node_count = self.size() - 1;
        let edge_count = self.width();

        if node_count == 0 {
            return Err(Error::new(
                std::io::ErrorKind::Other,
                "Graph has no vertices",
            ));
        }

        let threads = self.thread_count.max(1) as usize;
        let thread_load = node_count.div_ceil(threads);

        let (degree, mut node, mut core, mut pos) = self.init_procedural_memory_bz(mmap)?;
        // compute out-degrees in parallel
        let index_ptr = Arc::new(SharedSlice::<usize>::new(
            self.index.as_ptr() as *const usize,
            node_count + 1,
        ));
        let graph_ptr = Arc::new(SharedSlice::<Edge>::new(
            self.graph.as_ptr() as *const Edge,
            edge_count,
        ));

        // initialize degree and bins count vecs
        let mut bins: Vec<usize> = match thread::scope(|scope| {
            let mut bins = vec![0usize; 20];
            let mut max_vecs = vec![];

            for tid in 0..threads {
                let index_ptr = Arc::clone(&index_ptr);

                let mut deg_arr = degree.slice;

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                max_vecs.push(scope.spawn(move |_| {
                    let mut bins: Vec<usize> = vec![0; 20];
                    for v in start..end {
                        let deg = index_ptr.get(v + 1) - index_ptr.get(v);
                        if deg > u8::MAX as usize {
                            panic!("error degree({}) == {} but theoretical max is 16", v, deg);
                        }
                        bins[deg] += 1;
                        *deg_arr.get_mut(v) = deg as u8;
                    }
                    bins
                }));
            }
            // join results
            max_vecs.into_iter().for_each(|v| {
                let bin = match v.join() {
                    Ok(i) => i,
                    Err(e) => panic!("error getting thread bin count {:?}", e),
                };
                for (degree, count) in bin.iter().enumerate() {
                    bins[degree] += *count;
                }
            });
            bins
        }) {
            Ok(i) => {
                degree.flush()?;
                i
            }
            _ => panic!("error calculating max degree"),
        };

        let max_degree = match bins
            .iter()
            .enumerate()
            .max_by_key(|(deg, c)| *deg * (if **c != 0 { 1 } else { 0 }))
        {
            Some((deg, _)) => {
                bins.resize(deg + 1, 0);
                deg
            }
            None => panic!("error couldn't get max degree"),
        };

        // prefix sum to get starting indices for each degree
        let mut start_index = 0usize;
        for i in bins.iter_mut() {
            let count = *i;
            *i = start_index;
            start_index += count;
        }
        // `bins[d]` now holds the starting index in `vert` for vertices of degree d.
        // fill node array with vertices ordered by degree
        for v in 0..node_count {
            let d = *degree.get(v) as usize;
            let idx = bins[d] as usize;
            *node.get_mut(idx) = v;
            *pos.get_mut(v) = idx;
            bins[d] += 1; // increment the bin index for the next vertex of same degree
        }
        node.flush()?;
        pos.flush()?;

        // restore bin starting positions
        for d in (1..=max_degree).rev() {
            bins[d] = bins[d - 1];
        }
        bins[0] = 0;

        // peel vertices in order of increasing current degree
        let mut degree = degree.slice;
        for i in 0..node_count {
            let v = *node.get(i);
            let deg_v = *degree.get(v);
            *core.get_mut(v) = deg_v; // coreness of v

            // iterate outgoing neighbors of v
            for e in *index_ptr.get(v)..*index_ptr.get(v + 1) {
                let u = (*graph_ptr.get(e)).dest();
                let deg_u = *degree.get(u);
                if deg_u > deg_v {
                    // swap u's position node array to maintain order
                    let u_pos = *pos.get(u);
                    let new_pos = bins[deg_u as usize];
                    // bins[deg_u] points to start of nodes with degree >= old_deg
                    // swap the node at new_pos with u to move u into the bucket of (u_new_degree)
                    let w = *node.get(new_pos);
                    if u != w {
                        *node.get_mut(u_pos) = w;
                        *pos.get_mut(w) = u_pos;
                        *node.get_mut(new_pos) = u;
                        *pos.get_mut(u) = new_pos;
                    }
                    bins[deg_u as usize] += 1;
                    *degree.get_mut(u) = deg_u - 1;
                }
            }
        }
        core.flush()?;

        let output_filename = cache_file_name(
            self.graph_cache.graph_filename.clone(),
            FileType::KCore,
            None,
        )?;
        let outfile = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&output_filename)?;
        outfile.set_len(edge_count as u64 * std::mem::size_of::<u8>() as u64)?;
        let (out, _mmap): (SharedSliceMut<u8>, MmapMut) = SharedSliceMut::from_file(&outfile)?;
        let out_slice = out;

        thread::scope(|scope| {
            let mut res = vec![];
            for tid in 0..threads {
                let index_ptr = Arc::clone(&index_ptr);
                let graph_ptr = Arc::clone(&graph_ptr);

                let mut out_ptr = out_slice;
                let core = &core.slice;

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                res.push(scope.spawn(move |_| -> Vec<u64> {
                    let mut res = vec![0u64; 20];
                    for u in start..end {
                        let core_u = *core.get(u);
                        for e in *index_ptr.get(u)..*index_ptr.get(u + 1) {
                            let v = (*graph_ptr.get(e)).dest();
                            // edge core = min(core[u], core[v])
                            let core_val = if *core.get(u) < *core.get(v) {
                                core_u
                            } else {
                                *core.get(v)
                            };
                            *out_ptr.get_mut(e) = core_val;
                            res[core_val as usize] += 1;
                        }
                    }
                    res
                }));
            }
            let joined_res: Vec<Vec<u64>> = res
                .into_iter()
                .map(|v| v.join().expect("error thread panicked"))
                .collect();
            let mut r = vec![0u64; 16];
            for i in 0..16 {
                for v in joined_res.clone() {
                    r[i] += v[i];
                }
            }
            println!("k-cores {:?}", r);
        })
        .unwrap();

        // flush output to ensure all data is written to disk
        _mmap.flush()?;

        Ok(output_filename)
    }

    fn init_procedural_memory_liu_et_al(&self, mmap: u8) -> Result<ProceduralMemoryLiuEtAL, Error> {
        let node_count = self.size() - 1;
        let edge_count = self.width();

        let template_fn = self.graph_cache.graph_filename.clone();
        let d_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(0))?;
        let ni_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(5))?;
        let a_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(1))?;
        let c_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(2))?;
        let f_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(3))?;
        let fs_fn = cache_file_name(template_fn.clone(), FileType::KCore, Some(4))?;

        let degree = SharedSliceMut::<AtomicU8>::abst_mem_mut(d_fn, node_count, mmap > 0)?;
        let node_index = SharedSliceMut::<usize>::abst_mem_mut(ni_fn, node_count, mmap > 3)?;
        let alive = SharedSliceMut::<AtomicBool>::abst_mem_mut(a_fn, node_count, mmap > 1)?;
        let coreness = SharedSliceMut::<u8>::abst_mem_mut(c_fn, node_count, mmap > 2)?;
        let frontier = SharedSliceMut::<usize>::abst_mem_mut(f_fn, edge_count, mmap > 3)?;
        let frontier_swap = SharedSliceMut::<usize>::abst_mem_mut(fs_fn, edge_count, mmap > 3)?;

        Ok((degree, node_index, alive, coreness, frontier, frontier_swap))
    }

    pub fn compute_k_core_liu_et_al(&self, mmap: u8) -> Result<String, Error> {
        let node_count = self.size() - 1;
        let edge_count = self.width();

        if node_count == 0 {
            return Err(Error::new(
                std::io::ErrorKind::Other,
                "Graph has no vertices",
            ));
        }

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

        let (degree, _node_index, alive, coreness, frontier, swap) =
            self.init_procedural_memory_liu_et_al(mmap)?;

        // Initialize
        let total_dead_nodes = thread::scope(|s| -> usize {
            let mut dead_nodes = vec![];
            for tid in 0..threads {
                let index_ptr = &index_ptr;

                let mut degree = degree.slice.clone();
                let mut alive = alive.slice.clone();

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                dead_nodes.push(s.spawn(move |_| -> usize {
                    let mut dead_nodes = 0;
                    for u in start..end {
                        let deg_u = index_ptr.get(u + 1) - index_ptr.get(u);
                        *degree.get_mut(u) = AtomicU8::new(deg_u as u8);
                        if deg_u == 0 {
                            *alive.get_mut(u) = AtomicBool::new(false);
                            dead_nodes += 1;
                        } else {
                            *alive.get_mut(u) = AtomicBool::new(true);
                        }
                    }
                    dead_nodes
                }));
            }
            let mut total_dead_nodes = 0;
            dead_nodes
                .into_iter()
                .map(|e| e.join().expect("error"))
                .for_each(|e| total_dead_nodes += e);
            let _ = degree._flush_async();
            let _ = alive._flush_async();
            total_dead_nodes
        })
        .unwrap();

        // ditch node sampling as graphs are inherintely sparse
        // use veertical granularity control:
        // When peeling a low-degree vertex 𝑣, we
        // place all its active neighbors in a FIFO queue, referred to as the local
        // queue of 𝑣, and process all vertices in the local queue sequentially.
        // When we decrementing the induced degree of a neighbor 𝑢, if 𝑑˜ [𝑢]
        // drops to 𝑘 (line 6 in Alg. 3), instead of adding 𝑢 to Fnext , we add 𝑢 to
        // the local queue. This allows 𝑢 to be processed in the same subround
        // as 𝑣, rather than waiting for the next subround. We refer to this
        // process as a local search at 𝑣.

        // --- Core-peeling loop (Liu et al. algorithm) ---
        // for nodes with degree <= 16 no bucketing is used
        let mut k = 1u8;
        let mut remaining = node_count - total_dead_nodes; // number of vertices not yet peeled
        let frontier = SharedQueueMut::<usize>::from_shared_slice(frontier.slice);
        let swap = SharedQueueMut::<usize>::from_shared_slice(swap.slice);
        let synchronize = Arc::new(Barrier::new(threads));

        thread::scope(|s| {
            for tid in 0..threads {
                let index_ptr = &index_ptr;
                let graph_ptr = &graph_ptr;

                let degree = &degree;
                let alive = &alive.slice;
                let mut coreness = coreness.slice;
                let mut frontier = frontier.clone();
                let mut swap = swap.clone();

                let synchronize = Arc::clone(&synchronize);

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                s.spawn(move |_| {
                    let _local_queue: Vec<usize> = vec![];
                    while remaining > 0 {
                        synchronize.wait();
                        // Build initial frontier = all vertices with degree <= k that are still active.
                        // FIXME:: Abstract frontier to RAM with fallback mmap
                        for u in start..end {
                            if alive.get(u).load(Ordering::Relaxed)
                                && degree.get(u).load(Ordering::Relaxed) <= k
                            {
                                alive.get(u).store(false, Ordering::Relaxed);
                                let _ = frontier.push(u);
                            }
                        }

                        synchronize.wait();

                        if frontier.is_empty() {
                            k = match k.overflowing_add(1) {
                                (r, false) => r,
                                _ => panic!("error overflow when adding to k ({} + {})", k, 1),
                            };
                            continue;
                        }

                        // Process subrounds for current k: peel all vertices with degree k.
                        // FIXME: stack struct isn't being shared by threads, only memory location
                        while !frontier.is_empty() {
                            remaining = match remaining.overflowing_sub(frontier.len()) {
                                (r, false) => r,
                                _ => panic!(
                                    "error overflow when decreasing remaining ({} - {})",
                                    remaining,
                                    frontier.len()
                                ),
                            };

                            let chunk_size = frontier.len().div_ceil(threads);
                            let start = std::cmp::min(tid * chunk_size, frontier.len());
                            let end = std::cmp::min(start + chunk_size, frontier.len());

                            if let Some(chunk) = frontier.slice(start, end) {
                                for i in 0..end - start {
                                    // Set coreness and decrement neighbour degrees
                                    let u = *chunk.get(i);
                                    *coreness.get_mut(u) = k;
                                    // For each neighbor v of u:

                                    for idx in *index_ptr.get(u)..*index_ptr.get(u + 1) {
                                        let v = graph_ptr.get(idx).dest();
                                        if let Ok(old) = degree.get(v).fetch_update(
                                            Ordering::Relaxed,
                                            Ordering::Relaxed,
                                            |x| {
                                                if x > k {
                                                    match x.overflowing_sub(1) {
                                                        (r, false) => Some(r),
                                                        _ => None,
                                                    }
                                                } else {
                                                    None
                                                }
                                            },
                                        ) {
                                            if old == k + 1 {
                                                let life =
                                                    alive.get(v).swap(false, Ordering::Relaxed);
                                                if life {
                                                    let _ = swap.push(v);
                                                }
                                            }
                                        }
                                    }
                                }
                            }

                            synchronize.wait();

                            swap = std::mem::replace(&mut frontier, swap).clear();

                            synchronize.wait();
                        }
                        k = match k.overflowing_add(1) {
                            (r, false) => r,
                            _ => panic!("error overflow when adding to k ({} + {})", k, 1),
                        };
                    }
                });
            }
        })
        .unwrap();
        coreness.flush()?;

        // --- Compute per-edge core labels and write output ---
        // Create an output memory-mapped buffer for edge labels (u32 per directed edge).
        let output_filename = cache_file_name(
            self.graph_cache.graph_filename.clone(),
            FileType::KCore,
            None,
        )?;
        let outfile = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&output_filename)?;
        outfile.set_len(edge_count as u64 * std::mem::size_of::<u8>() as u64)?;
        let (out, _mmap): (SharedSliceMut<u8>, MmapMut) = SharedSliceMut::from_file(&outfile)?;

        // parallel edge labeling: partition vertices among threads and write edge core values
        thread::scope(|scope| {
            let mut res = vec![];
            let threads = self.thread_count.max(1) as usize;
            for tid in 0..threads {
                let index_ptr = &index_ptr;
                let graph_ptr = &graph_ptr;
                let coreness = &coreness.slice;
                let start = thread_load * tid;
                let end = std::cmp::min(start + thread_load, node_count);
                let mut edge_coreness = out;
                res.push(scope.spawn(move |_| -> Vec<u64> {
                    let mut res = vec![0u64; 20];
                    for u in start..end {
                        let core_u = *coreness.get(u);
                        for e in *index_ptr.get(u)..*index_ptr.get(u + 1) {
                            let v = graph_ptr.get(e).dest();
                            // edge_coreness = min(core[u], core[v])
                            let core_val = if *coreness.get(u) < *coreness.get(v) {
                                core_u
                            } else {
                                *coreness.get(v)
                            };
                            *edge_coreness.get_mut(e) = core_val;
                            res[core_val as usize] += 1;
                        }
                    }
                    res
                }));
            }
            let joined_res: Vec<Vec<u64>> = res
                .into_iter()
                .map(|v| v.join().expect("error thread panicked"))
                .collect();
            let mut r = vec![0u64; 16];
            for i in 0..16 {
                for v in joined_res.clone() {
                    r[i] += v[i];
                }
            }
            println!("k-cores {:?}", r);
        })
        .unwrap();

        // flush output to ensure all data is written to disk
        _mmap.flush()?;

        Ok(output_filename)
    }
    fn init_procedural_memory_k_truss_decomposition(
        &self,
        mmap: u8,
    ) -> Result<ProceduralMemoryKTDecomposition, Error> {
        let edge_count = self.width();

        let template_fn = self.graph_cache.graph_filename.clone();
        let t_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(0))?;
        let el_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(1))?;
        let ei_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(2))?;
        let s_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(3))?;
        let e_fn = cache_file_name(template_fn.clone(), FileType::KTruss, None)?;

        let tri_count = SharedSliceMut::<AtomicU8>::abst_mem_mut(t_fn, edge_count, mmap > 0)?;
        let edge_list = SharedSliceMut::<usize>::abst_mem_mut(el_fn, edge_count, mmap > 1)?;
        let edge_index = SharedSliceMut::<usize>::abst_mem_mut(ei_fn, edge_count, mmap > 1)?;
        let stack = SharedSliceMut::<(usize, usize)>::abst_mem_mut(s_fn, edge_count * 2, mmap > 2)?;
        let edge_trussness = SharedSliceMut::<u8>::abst_mem_mut(e_fn, edge_count, true)?;

        Ok((tri_count, edge_list, edge_index, stack, edge_trussness))
    }

    pub fn k_truss_decomposition(&self, mmap: u8) -> Result<String, Error> {
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

        // Shared atomic & simple arrays for counts and trussness
        let (triangle_count, edges, edge_index, edge_stack, edge_trussness) =
            self.init_procedural_memory_k_truss_decomposition(mmap)?;

        let edge_reciprocal = self.clone().get_edge_reciprocal()?;
        let edge_out = self.clone().get_edge_dest_id_over_source()?;

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        // Algorithm 1 - adjusted for directed scheme
        thread::scope(|scope| {
            for tid in 0..threads {
                let index_ptr = Arc::clone(&index_ptr);
                let graph_ptr = Arc::clone(&graph_ptr);

                let eo = edge_out.slice;
                let er = edge_reciprocal.slice;

                let mut trussness = edge_trussness.slice;
                let mut tris = triangle_count.slice.clone();
                let mut edges = edges.slice;
                let mut edge_index = edge_index.slice;

                let synchronize = Arc::clone(&synchronize);

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                scope.spawn(move |_| {
                    // initialize triangle_count with zeroes
                    let edge_begin = *index_ptr.get(start);
                    let edge_end = *index_ptr.get(end);
                    for idx in edge_begin..edge_end {
                        *trussness.get_mut(idx) = 0;
                        *tris.get_mut(idx) = AtomicU8::new(0);
                    }

                    synchronize.wait();

                    let mut neighbours = HashMap::<usize, usize>::new();
                    for u in start..end {
                        for j in *eo.get(u)..*index_ptr.get(u + 1) {
                            let w = *graph_ptr.get(j);
                            *edges.get_mut(j) = j;
                            *edge_index.get_mut(j) = j;
                            neighbours.insert(w.dest(), j);
                        }
                        for u_v in *index_ptr.get(u)..*eo.get(u) {
                            *edges.get_mut(u_v) = u_v;
                            *edge_index.get_mut(u_v) = u_v;
                            let v = *graph_ptr.get(u_v);
                            let v = v.dest();
                            if u == v {
                                continue;
                            }
                            for v_w in (*eo.get(v)..*index_ptr.get(v + 1)).rev() {
                                let w = graph_ptr.get(v_w).dest();
                                if w <= u {
                                    break;
                                }
                                let w_u = match neighbours.get(&w) {
                                    Some(i) => *i,
                                    None => continue,
                                };

                                tris.get(u_v).fetch_add(1, Ordering::Relaxed);
                                tris.get(v_w).fetch_add(1, Ordering::Relaxed);
                                tris.get(w_u).fetch_add(1, Ordering::Relaxed);
                                tris.get(*er.get(u_v)).fetch_add(1, Ordering::Relaxed);
                                tris.get(*er.get(v_w)).fetch_add(1, Ordering::Relaxed);
                                tris.get(*er.get(w_u)).fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        neighbours.clear();
                    }
                });
            }
        })
        .unwrap();

        let stack = SharedQueueMut::<(usize, usize)>::from_shared_slice(edge_stack.slice);

        // Algorithm 2 - sentinel value is 0
        // blank (u64, usize) value
        let mut edges = edges.slice;
        let mut edge_index = edge_index.slice;
        let mut edge_count = edge_count;
        let er = edge_reciprocal.slice;
        let mut trussness = edge_trussness.slice;
        let tris = triangle_count.slice.clone();
        let mut stack = stack.clone();
        let mut test = vec![0u64; 16];

        // max node degree is 16
        for k in 1..16 {
            if edge_count == 0 {
                break;
            }
            let mut idx = 0;
            while idx < edge_count {
                let edge_offset = *edge_index.get(idx);
                let t_count = tris.get(edge_offset).load(Ordering::Relaxed);
                if t_count == k {
                    let u = graph_ptr.get(*er.get(edge_offset)).dest();
                    stack.push((u, edge_offset));
                    idx = match idx.overflowing_add(1) {
                        (r, false) => r,
                        _ => panic!("error overflow adding to idx ({} + {})", idx, 1),
                    };
                } else if t_count == 0 {
                    edge_count = match edge_count.overflowing_sub(1) {
                        (r, false) => r,
                        _ => panic!(
                            "error overflow subtracting to edge_count ({} - {})",
                            edge_count, 1
                        ),
                    };
                    let e_index = *edge_index.get(edge_count);
                    let r_index = *edges.get(edge_offset);
                    *edge_index.get_mut(edge_count) = *edge_index.get(r_index);
                    *edge_index.get_mut(r_index) = e_index;
                    *edges.get_mut(edge_offset) = edge_count;
                    *edges.get_mut(e_index) = r_index;
                    // store edge trussness
                    *trussness.get_mut(edge_offset) = k - 1;
                    test[k as usize - 1] += 1;
                    continue;
                } else {
                    idx = match idx.overflowing_add(1) {
                        (r, false) => r,
                        _ => panic!("error overflow adding to idx ({} + {})", idx, 1),
                    };
                }
            }
            let mut neighbours = HashMap::<usize, usize>::new();
            while let Some((u, offset)) = stack.pop() {
                tris.get(offset).store(0, Ordering::Relaxed);
                let v = graph_ptr.get(offset).dest();
                for u_w in *index_ptr.get(u)..*index_ptr.get(u + 1) {
                    let w = graph_ptr.get(u_w).dest();
                    if w != u && w != v {
                        neighbours.insert(w, u_w);
                    }
                }
                for v_w in *index_ptr.get(v)..*index_ptr.get(v + 1) {
                    let w = graph_ptr.get(v_w).dest();
                    if w == v {
                        continue;
                    }
                    let u_w = match neighbours.get(&w) {
                        Some(i) => *i,
                        None => continue,
                    };

                    let w_u = *er.get(u_w);
                    if tris.get(v_w).load(Ordering::Relaxed) != 0
                        && tris.get(w_u).load(Ordering::Relaxed) != 0
                    {
                        let prev_w_u = tris.get(w_u).fetch_sub(1, Ordering::Relaxed);
                        let prev_v_w = tris.get(v_w).fetch_sub(1, Ordering::Relaxed);
                        if prev_w_u == k + 1 {
                            stack.push((w, w_u));
                        }
                        if prev_v_w == k + 1 {
                            stack.push((v, v_w));
                        }
                    }
                }
                neighbours.clear();
            }
        }
        println!("k-trussness {:?}", test);
        edge_trussness.flush()?;

        cache_file_name(
            self.graph_cache.graph_filename.clone(),
            FileType::KTruss,
            None,
        )
    }

    fn init_procedural_memory_build_reciprocal(
        &self,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<usize>,
            AbstractedProceduralMemoryMut<usize>,
        ),
        Error,
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
        Error,
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

                let mut er = er.slice;
                let mut eo = eo.slice;

                let synchronize = Arc::clone(&synchronize);

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);
                scope.spawn(move |_| -> Result<(), Error> {
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
                                    panic!("error couldn't find reciprocal for edge {}, u: ({}) -> v: ({})", edge_offset, u, v);
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

    fn get_edge_reciprocal(&self) -> Result<AbstractedProceduralMemory<usize>, Error> {
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
                    Err(e) => panic!("error creating abstractmemory for edge_reciprocal {}", e),
                }
            }
        };
        er
    }

    fn get_edge_dest_id_over_source(&self) -> Result<AbstractedProceduralMemory<usize>, Error> {
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
                    Err(e) => panic!("error creating abstractmemory for edge_over {}", e),
                }
            }
        };
        eo
    }

    fn init_procedural_memory_pkt(&self, mmap: u8) -> Result<ProceduralMemoryPKT, Error> {
        let edge_count = self.width();

        let template_fn = self.graph_cache.graph_filename.clone();
        let c_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(1))?;
        let n_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(2))?;
        let p_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(3))?;
        let ic_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(4))?;
        let in_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(5))?;
        let s_fn = cache_file_name(template_fn.clone(), FileType::KTruss, None)?;

        let curr = SharedSliceMut::<usize>::abst_mem_mut(c_fn, edge_count, mmap > 2)?;
        let next = SharedSliceMut::<usize>::abst_mem_mut(n_fn, edge_count, mmap > 2)?;
        let processed = SharedSliceMut::<bool>::abst_mem_mut(p_fn, edge_count, mmap > 3)?;
        let in_curr = SharedSliceMut::<bool>::abst_mem_mut(ic_fn, edge_count, mmap > 3)?;
        let in_next = SharedSliceMut::<bool>::abst_mem_mut(in_fn, edge_count, mmap > 3)?;
        let s = SharedSliceMut::<AtomicU8>::abst_mem_mut(s_fn, edge_count, true)?;

        Ok((curr, next, processed, in_curr, in_next, s))
    }

    pub fn pkt(&self, mmap: u8) -> Result<String, Error> {
        let node_count = self.size() - 1;
        let edge_count = self.width();

        let threads = self.thread_count.max(1) as usize;
        let edge_load = edge_count.div_ceil(threads);
        let node_load = node_count.div_ceil(threads);

        let index_ptr = Arc::new(SharedSlice::<usize>::new(
            self.index.as_ptr() as *const usize,
            node_count + 1,
        ));
        let graph_ptr = Arc::new(SharedSlice::<Edge>::new(
            self.graph.as_ptr() as *const Edge,
            edge_count,
        ));

        // Shared arrays
        let (curr, next, processed, in_curr, in_next, s) = self.init_procedural_memory_pkt(mmap)?;
        let edge_reciprocal = self.clone().get_edge_reciprocal()?;
        let edge_out = self.clone().get_edge_dest_id_over_source()?;

        // Allocate memory for thread local arrays
        let template_fn = self.graph_cache.graph_filename.clone();
        let mut x: Vec<AbstractedProceduralMemoryMut<usize>> = Vec::new();
        for i in 0..threads {
            let x_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(8 + i))?;
            x.push(SharedSliceMut::<usize>::abst_mem_mut(
                x_fn,
                node_count,
                mmap > 0,
            )?)
        }
        let x = Arc::new(x);

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        // ParTriangle-AM4
        thread::scope(|scope| {
            for tid in 0..threads {
                // eid is unnecessary as graph + index alwready do the job
                let graph_ptr = Arc::clone(&graph_ptr);
                let index_ptr = Arc::clone(&index_ptr);
                let mut x = x[tid].slice;
                let eo = edge_out.slice;
                let mut s = s.slice.clone();
                let er = edge_reciprocal.slice;
                let mut curr = curr.slice;
                let mut next = next.slice;
                let mut in_curr = in_curr.slice;
                let mut in_next = in_next.slice;
                let mut processed = processed.slice;
                let synchronize = Arc::clone(&synchronize);

                let init_begin = std::cmp::min(tid * edge_load, edge_count);
                let init_end = std::cmp::min(init_begin + edge_load, edge_count);
                let begin = std::cmp::min(tid * node_load, node_count - 1); // is node and edge
                let end = std::cmp::min(begin + node_load, node_count - 1); // limit accurate?
                scope.spawn(move |_| {
                    // initialize s, edge_out, x, curr, next, in_curr, in_next & processed
                    for edge_offset in init_begin..init_end {
                        *s.get_mut(edge_offset) = AtomicU8::new(0);
                        *curr.get_mut(edge_offset) = 0;
                        *next.get_mut(edge_offset) = 0;
                        *in_curr.get_mut(edge_offset) = false;
                        *in_next.get_mut(edge_offset) = false;
                        *processed.get_mut(edge_offset) = false;
                        *x.get_mut(graph_ptr.get(edge_offset).dest()) = 0;
                    }

                    synchronize.wait();

                    for u in begin..end {
                        let edges_stop = *index_ptr.get(u + 1);
                        let eo_u = *eo.get(u);
                        for j in eo_u..edges_stop {
                            *x.get_mut(graph_ptr.get(j).dest()) = j + 1;
                        }
                        for u_v in *index_ptr.get(u)..eo_u {
                            let v = graph_ptr.get(u_v).dest();
                            if v == u {
                                break;
                            }
                            for v_w in (*eo.get(v)..*index_ptr.get(v + 1)).rev() {
                                let w = graph_ptr.get(v_w).dest();
                                if w <= u {
                                    break;
                                }

                                let w_u = match x.get(w).cmp(&0) {
                                    std::cmp::Ordering::Equal => continue,
                                    _ => *x.get(w) - 1,
                                };

                                s.get(u_v).fetch_add(1, Ordering::Relaxed);
                                s.get(v_w).fetch_add(1, Ordering::Relaxed);
                                s.get(w_u).fetch_add(1, Ordering::Relaxed);
                                s.get(*er.get(u_v)).fetch_add(1, Ordering::Relaxed);
                                s.get(*er.get(v_w)).fetch_add(1, Ordering::Relaxed);
                                s.get(*er.get(w_u)).fetch_add(1, Ordering::Relaxed);
                            }
                        }

                        for j in eo_u..edges_stop {
                            *x.get_mut(graph_ptr.get(j).dest()) = 0;
                        }
                    }
                });
            }
        })
        .unwrap();

        let mut l: u8 = 1;
        let buff_size = 4096;
        let total_duds = Arc::new(AtomicUsize::new(0));
        let curr = SharedQueueMut::from_shared_slice(curr.slice);
        let next = SharedQueueMut::from_shared_slice(next.slice);

        thread::scope(|scope| {
            let mut res = Vec::new();
            for tid in 0..threads {
                let graph_ptr = Arc::clone(&graph_ptr);
                let index_ptr = Arc::clone(&index_ptr);

                let mut todo = edge_count;
                let mut x = x[tid].slice;

                let s = s.slice.clone();
                let mut curr = curr.clone();
                let mut next = next.clone();
                let er = edge_reciprocal.slice;
                let mut in_curr = in_curr.slice;
                let mut in_next = in_next.slice;
                let mut processed = processed.slice;

                let total_duds = Arc::clone(&total_duds);
                let synchronize = Arc::clone(&synchronize);

                let begin = std::cmp::min(tid * edge_load, edge_count);
                let end = std::cmp::min(begin + edge_load, edge_count);

                res.push(scope.spawn(move |_| -> Result<Vec<u64>, Error> {
                    let mut res = vec![0_u64; 20];
                    let mut buff = vec![0; buff_size];
                    let mut i = 0;

                    // Remove 0-triangle edges
                    for e in begin..end {
                        if s.get(e).load(Ordering::Relaxed) == 0 {
                            *processed.get_mut(e) = true;
                            res[0] += 1;
                            i += 1;
                        }
                    }
                    total_duds.fetch_add(i, Ordering::SeqCst);
                    i = 0;
                    synchronize.wait();

                    todo = match todo.overflowing_sub(total_duds.load(Ordering::Relaxed)) {
                        (r, false) => r,
                        _ => panic!(
                            "error overflow when decrementing todo ({} - {})",
                            todo,
                            total_duds.load(Ordering::Relaxed)
                        ),
                    };

                    // println!("triangles removed");
                    while todo > 0 {
                        for e in begin..end {
                            if s.get(e).load(Ordering::Relaxed) == l {
                                buff[i] = e;
                                *in_curr.get_mut(e) = true;
                                i += 1;
                            }
                            if i == buff_size {
                                curr.push_slice(buff.as_slice());
                                i = 0;
                            }
                        }
                        if i > 0 {
                            curr.push_slice(&buff[0..i]);
                            i = 0;
                        }
                        synchronize.wait();

                        let mut to_process = match curr.slice(0, curr.len()) {
                            Some(i) => i,
                            None => panic!("error reading curr in pkt"),
                        };
                        // println!("new cicle initialized {} {:?}", todo, curr.ptr);
                        while to_process.len() != 0 {
                            todo = match todo.overflowing_sub(to_process.len()) {
                                (r, false) => r,
                                _ => panic!(
                                    "error overflow when decrementing todo ({} - {})",
                                    todo,
                                    to_process.len()
                                ),
                            };
                            synchronize.wait();

                            // ProcessSubLevel
                            let thread_load = curr.len().div_ceil(threads);
                            let begin = tid * thread_load;
                            let end = std::cmp::min(begin + thread_load, curr.len());

                            for e_idx in begin..end {
                                let u_v = *to_process.get(e_idx);

                                let u = graph_ptr.get(*er.get(u_v)).dest();
                                let v = graph_ptr.get(u_v).dest();

                                let edges_start = *index_ptr.get(u);
                                let edges_stop = *index_ptr.get(u + 1);

                                // mark u neighbours
                                for u_w in edges_start..edges_stop {
                                    let w = graph_ptr.get(u_w).dest();
                                    if w != u {
                                        *x.get_mut(w) = *er.get(u_w) + 1;
                                    }
                                }

                                for v_w in *index_ptr.get(v)..*index_ptr.get(v + 1) {
                                    let w = graph_ptr.get(v_w).dest();
                                    if *x.get(w) == 0 {
                                        continue;
                                    }
                                    let w_u = *x.get(w) - 1;
                                    if *processed.get(v_w) || *processed.get(w_u) {
                                        continue;
                                    }

                                    if s.get(v_w).load(Ordering::Relaxed) > l
                                        && s.get(w_u).load(Ordering::Relaxed) > l
                                    {
                                        let prev_l_v_w = s.get(v_w).fetch_sub(1, Ordering::SeqCst);
                                        if prev_l_v_w == l + 1 {
                                            *in_next.get_mut(v_w) = true;
                                            buff[i] = v_w;
                                            i += 1;
                                            if i == buff_size {
                                                next.push_slice(&buff[..]);
                                                i = 0;
                                            }
                                        }
                                        if prev_l_v_w <= l {
                                            s.get(v_w).fetch_add(1, Ordering::SeqCst);
                                        }
                                        let prev_l_w_u = s.get(w_u).fetch_sub(1, Ordering::SeqCst);
                                        if prev_l_w_u == l + 1 {
                                            *in_next.get_mut(w_u) = true;
                                            buff[i] = w_u;
                                            i += 1;
                                            if i == buff_size {
                                                next.push_slice(&buff[..]);
                                                i = 0;
                                            }
                                        }
                                        if prev_l_w_u <= l {
                                            s.get(w_u).fetch_add(1, Ordering::SeqCst);
                                        }
                                    } else if s.get(v_w).load(Ordering::Relaxed) > l
                                        && ((u_v < w_u && *in_curr.get(w_u)) || !*in_curr.get(w_u))
                                    {
                                        let prev_l_v_w = s.get(v_w).fetch_sub(1, Ordering::SeqCst);
                                        if prev_l_v_w == l + 1 {
                                            *in_next.get_mut(v_w) = true;
                                            buff[i] = v_w;
                                            i += 1;
                                            if i == buff_size {
                                                next.push_slice(&buff[..]);
                                                i = 0;
                                            }
                                        }
                                        if prev_l_v_w <= l {
                                            s.get(v_w).fetch_add(1, Ordering::SeqCst);
                                        }
                                    } else if s.get(w_u).load(Ordering::Relaxed) > l
                                        && ((u_v < v_w && *in_curr.get(v_w)) || !*in_curr.get(v_w))
                                    {
                                        let prev_l_w_u = s.get(w_u).fetch_sub(1, Ordering::SeqCst);
                                        if prev_l_w_u == l + 1 {
                                            *in_next.get_mut(w_u) = true;
                                            buff[i] = w_u;
                                            i += 1;
                                            if i == buff_size {
                                                next.push_slice(&buff[..]);
                                                i = 0;
                                            }
                                        }
                                        if prev_l_w_u <= l {
                                            s.get(w_u).fetch_add(1, Ordering::SeqCst);
                                        }
                                    }
                                }

                                // unmark u neighbours
                                for u_w in edges_start..edges_stop {
                                    *x.get_mut(graph_ptr.get(u_w).dest()) = 0;
                                }
                            }
                            if i > 0 {
                                next.push_slice(&buff[0..i]);
                                i = 0;
                            }
                            for e_idx in begin..end {
                                let edge = *to_process.get(e_idx);
                                *processed.get_mut(edge) = true;
                                *in_curr.get_mut(edge) = false; // FIXME: this can be removed?
                            }
                            // println
                            for _e in begin..end {
                                res[l as usize] += 1;
                            }

                            synchronize.wait();
                            next = std::mem::replace(&mut curr, next).clear();
                            in_next = std::mem::replace(&mut in_curr, in_next);

                            synchronize.wait();
                            to_process = match curr.slice(0, curr.len()) {
                                Some(i) => i,
                                None => panic!("error couldn't get new to_process vec"),
                            };
                            synchronize.wait();
                        }
                        l = match l.overflowing_add(1) {
                            (r, false) => r,
                            _ => panic!("error overflow when adding to l ({} - {})", l, 1),
                        };
                        synchronize.wait();
                    }
                    Ok(res)
                }));
            }
            let joined_res: Vec<Vec<u64>> = res
                .into_iter()
                .map(|v| v.join().expect("error thread panicked").expect("error ??1"))
                .collect();
            let mut r = vec![0u64; 16];
            for i in 0..16 {
                for v in joined_res.clone() {
                    r[i] += v[i];
                }
            }
            println!("k-trussness {:?}", r);
        })
        .unwrap();

        cache_file_name(
            self.graph_cache.graph_filename.clone(),
            FileType::KTruss,
            None,
        )
    }

    pub fn cleanup_cache(&self) -> Result<(), Error> {
        self.graph_cache.cleanup_cache()
    }
}

#[derive(Debug, Clone)]
pub struct NeighbourIter<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    edge_ptr: *const Edge,
    _orig_edge_ptr: *const Edge,
    _orig_id_ptr: *const usize,
    #[expect(dead_code)]
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

    fn remaining_neighbours(&self) -> usize {
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
            _marker1: self._marker1,
            _marker2: self._marker2,
        }
    }
}

impl fmt::Display for FileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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
            FileType::ExportGraphMask => "GraphExport",
        };
        write!(f, "{}", s)
    }
}

impl fmt::Display for ApproxDirichletHeatKernelK {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            ApproxDirichletHeatKernelK::None => "ApproxDirichletHeatKernelK::None",
            ApproxDirichletHeatKernelK::Mean => "ApproxDirichletHeatKernelK::Mean",
            ApproxDirichletHeatKernelK::Unlim => "ApproxDirichletHeatKernelK::Unlim",
        };
        write!(f, "{}", s)
    }
}

type ProceduralMemoryBZ = (
    AbstractedProceduralMemoryMut<u8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<u8>,
    AbstractedProceduralMemoryMut<usize>,
);

type ProceduralMemoryLiuEtAL = (
    AbstractedProceduralMemoryMut<AtomicU8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<AtomicBool>,
    AbstractedProceduralMemoryMut<u8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<usize>,
);

type ProceduralMemoryKTDecomposition = (
    AbstractedProceduralMemoryMut<AtomicU8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<(usize, usize)>,
    AbstractedProceduralMemoryMut<u8>,
);

type ProceduralMemoryPKT = (
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<AtomicU8>,
);
