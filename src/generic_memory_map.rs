use crate::node::EdgeType;
use crate::shared_slice::{SharedSlice, SharedSliceMut};

use bitfield::bitfield;
use bytemuck::{Pod, Zeroable};
use core::{fmt, panic};
use crossbeam::thread;
use fst::{Map, MapBuilder};
use glob::glob;
use memmap::{Mmap, MmapMut, MmapOptions};
use rand::seq::IndexedRandom;
use regex::Regex;
use static_assertions::const_assert;
use std::sync::atomic::{AtomicU32, AtomicUsize};
use std::usize;
use std::{
    any::type_name,
    collections::HashMap,
    fmt::{Debug, Display},
    fs::{self, File, OpenOptions},
    hash::{Hash, Hasher},
    io::{BufRead, BufReader, Error, Write},
    marker::PhantomData,
    path::Path,
    process::Command,
    slice,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicU64, Ordering},
    },
};
use zerocopy::*; // Using crossbeam for scoped threads

const_assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u64>());

static CACHE_DIR: &str = "./cache/";
static BYTES_U64: u64 = std::mem::size_of::<u64>() as u64;

pub trait EdgeOutOf {
    fn dest(&self) -> u64;
    fn edge_type(&self) -> EdgeType;
}

pub trait GraphEdge {
    fn new(orig: u64, out_edge: impl EdgeOutOf) -> Self;
    fn orig(&self) -> u64;
    fn dest(&self) -> u64;
    fn edge_type(&self) -> EdgeType;
}

bitfield! {
    #[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Pod, Zeroable)]
    #[repr(C)]
    pub struct OutEdgeRecord(u64);
    impl BitAnd;
    impl BitOr;
    impl BitXor;
    impl new;
    u64;
    u8, from into EdgeType, edge_type, set_edge_type: 1, 0;
    u64, dest_node, set_dest_node: 63, 2;
}

impl EdgeOutOf for OutEdgeRecord {
    fn dest(&self) -> u64 {
        self.dest_node()
    }
    fn edge_type(&self) -> EdgeType {
        self.edge_type()
    }
}

#[derive(Clone, Copy, FromBytes, IntoBytes, Immutable, KnownLayout, Pod, Zeroable)]
#[repr(C)]
pub struct DirectedEdge {
    origin: u64,
    edge: OutEdgeRecord,
}

impl DirectedEdge {
    #[inline]
    fn origin(&self) -> u64 {
        self.origin
    }

    #[inline]
    fn dest(&self) -> u64 {
        self.edge.dest_node()
    }

    #[inline]
    fn edge_type(&self) -> EdgeType {
        self.edge.edge_type()
    }
}

impl GraphEdge for DirectedEdge {
    #[inline]
    fn new(origin: u64, out_edge: impl EdgeOutOf) -> DirectedEdge {
        DirectedEdge {
            origin,
            edge: OutEdgeRecord::new(out_edge.edge_type(), out_edge.dest()),
        }
    }

    #[inline]
    fn orig(&self) -> u64 {
        self.origin
    }

    #[inline]
    fn dest(&self) -> u64 {
        self.dest()
    }

    #[inline]
    fn edge_type(&self) -> EdgeType {
        self.edge_type()
    }
}

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
}

fn cache_file_name(
    original_filename: String,
    target_type: FileType,
    sequence_number: Option<u64>,
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

pub struct GraphCache<T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf> {
    pub graph_file: Arc<File>,
    pub index_file: Arc<File>,
    pub kmer_file: Arc<File>,
    pub graph_filename: String,
    pub index_filename: String,
    pub kmer_filename: String,
    pub graph_bytes: u64,
    pub index_bytes: u64,
    pub readonly: bool,
    _marker: PhantomData<T>,
}

impl<T> GraphCache<T>
where
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
{
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

    pub fn init() -> Result<GraphCache<T>, Error> {
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

        Ok(GraphCache::<T> {
            graph_file: Arc::new(graph_file),
            index_file: Arc::new(index_file),
            kmer_file: Arc::new(kmer_file),
            graph_filename,
            index_filename,
            kmer_filename,
            graph_bytes: 0,
            index_bytes: 0,
            readonly: false,
            _marker: PhantomData::<T>,
        })
    }

    pub fn init_with_id(id: String) -> Result<GraphCache<T>, Error> {
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

        Ok(GraphCache::<T> {
            graph_file: Arc::new(graph_file),
            index_file: Arc::new(index_file),
            kmer_file: Arc::new(kmer_file),
            graph_filename,
            index_filename,
            kmer_filename,
            graph_bytes: 0,
            index_bytes: 0,
            readonly: false,
            _marker: PhantomData::<T>,
        })
    }

    pub fn open(filename: String) -> Result<GraphCache<T>, Error> {
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

        let graph_len = graph_file.metadata().unwrap().len();
        let index_len = index_file.metadata().unwrap().len();

        Ok(GraphCache::<T> {
            graph_file: Arc::new(graph_file),
            index_file: Arc::new(index_file),
            kmer_file: Arc::new(kmer_file),
            graph_filename,
            index_filename,
            kmer_filename,
            graph_bytes: graph_len,
            index_bytes: index_len,
            readonly: true,
            _marker: PhantomData::<T>,
        })
    }

    pub fn write_node(&mut self, node_id: u64, data: &[T], label: &str) -> Result<(), Error>
    where
        T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    {
        match node_id == self.index_bytes / 8 {
            true => {
                writeln!(self.kmer_file, "{}\t{}", node_id, label)?;

                match self.index_file.write_all(self.graph_bytes.as_bytes()) {
                    Ok(_) => self.index_bytes += BYTES_U64,
                    Err(e) => panic!("error writing index for {}: {}", node_id, e),
                };

                match self.graph_file.write_all(bytemuck::cast_slice(data)) {
                    Ok(_) => {
                        self.graph_bytes += data.len() as u64;
                        Ok(())
                    }
                    Err(e) => panic!("error writing edges for {}: {}", node_id, e),
                }
            }
            false => panic!(
                "error nodes must be mem mapped in ascending order, (id: {}, expected_id: {})",
                node_id,
                self.index_bytes / 8
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

        while let l = reader.read_until(b'\n', &mut line) {
            if !l.unwrap() > 0 {
                break;
            }
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
            Ok(_) => self.index_bytes += BYTES_U64,
            Err(e) => panic!("error couldn't finish index: {}", e),
        };

        // Build finite state tranducer for k-mer to
        let fst_filename = cache_file_name(self.kmer_filename.clone(), FileType::Fst, None)?;
        let sorted_file =
            cache_file_name(self.kmer_filename.clone(), FileType::KmerSortedTmp, None)?;

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
}

#[derive(Clone)]
struct FindDisjointSetsEulerTrails {
    trails: Vec<((usize, usize, usize), usize)>,
    cycle_check: bool,
}

impl FindDisjointSetsEulerTrails {
    pub fn new(trails: &mut [(usize, usize, usize)]) -> Self {
        trails.sort_by_key(|(trail_id, _, _)| *trail_id);
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

        // Break cycles in graph
        // println!("{:?}", self.trails);
        for (id, _i) in self.clone().trails.iter().enumerate() {
            // println!("check {}: {} -> {}", id, i.0.0, i.1);
            self.cycle_break(id);
        }
        // Works as an union find in tree
        for (id, _i) in self.clone().trails.iter().enumerate() {
            // println!("check {}: {} - -> {}", id, i.0.0, i.1);
            self.cycle_break(id);
        }

        self.cycle_check = true;
    }
}

struct TmpMemoryHelperStruct {
    _a: Vec<AtomicU64>,
    _b: Vec<AtomicU8>,
    _c: MmapMut,
    _d: MmapMut,
}

pub struct EulerTrail<
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
> {
    graph: GraphMemoryMap<T, U>,
}

impl<T, U> EulerTrail<T, U>
where
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
{
    pub fn new(graph: GraphMemoryMap<T, U>) -> Result<EulerTrail<T, U>, Error> {
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
        for (t_idx, t_begin, t_end) in cycle_offsets.iter() {
            let trail_ptr = match cycles.slice(*t_begin, *t_end) {
                Some(i) => i.as_ptr(),
                None => panic!("error getting memmapped slice of trail {}", t_idx),
            };
            let trail_slice = SharedSlice::<u64>::new(trail_ptr, *t_end - *t_begin);
            let mut pos = 0; // pos in u64 terms
            // read 8 pages of 4KB at a time
            while let Some(next_slice) = trail_slice.slice(pos, pos + 4096) {
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
        // in nodes u64
        let cycles_length: Vec<usize> = cycle_offsets.iter().map(|(_, b, e)| e - b).collect();
        for head_trail in trail_sets.keys() {
            output_len.insert(*head_trail, (*head_trail, 1));
            for (trail, _, _) in trail_sets.get(head_trail).unwrap() {
                output_len.get_mut(head_trail).unwrap().1 += cycles_length[*trail] - 1;
            }
        }

        let mut keys_by_trail_size: Vec<(usize, usize)> =
            output_len.values().map(|&(a, b)| (a, b)).collect();
        keys_by_trail_size.sort_by_key(|(_, s)| std::cmp::Reverse(*s));

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
                .for_each(|v| v.sort_by_key(|(pos, _)| std::cmp::Reverse(*pos)));
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
                Some(idx as u64),
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

    fn merge_euler_trails_no_mmap(
        &self,
        cycles: Mutex<Vec<Vec<u64>>>,
    ) -> Result<Vec<(u64, u64)>, Error> {
        let mut trail_heads: HashMap<u64, Vec<(usize, usize, usize)>> = HashMap::new();
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
        keys_by_trail_size.sort_by_key(|(_, s)| std::cmp::Reverse(*s));

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
                .for_each(|v| v.sort_by_key(|(pos, _)| std::cmp::Reverse(*pos)));

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
                Some(idx as u64),
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
            .map(|(idx, (_, size))| (idx as u64, *size as u64))
            .collect())
    }

    fn initialize_hierholzers_procedural_memory(
        &self,
        mmap: u8,
    ) -> Result<
        (
            Arc<SharedSliceMut<AtomicU64>>,
            Arc<SharedSliceMut<AtomicU8>>,
            TmpMemoryHelperStruct,
        ),
        Error,
    > {
        let node_count = (self.graph.size() - 1) as usize;
        let index_ptr = Arc::new(SharedSlice::<u64>::new(
            self.graph.index.as_ptr() as *const u64,
            self.graph.size() as usize,
        ));
        let edge_vec_fn = cache_file_name(
            self.graph.graph_cache.graph_filename.clone(),
            FileType::EulerTmp,
            Some(1),
        )?;
        let edge_count_fn = cache_file_name(
            self.graph.graph_cache.graph_filename.clone(),
            FileType::EulerTmp,
            Some(2),
        )?;
        let edge_vec_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&edge_vec_fn)?;
        edge_vec_file.set_len(
            if mmap > 0 { self.graph.size() - 1 } else { 1 }
                * std::mem::size_of::<AtomicU64>() as u64,
        )?;
        let edge_count_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&edge_count_fn)?;
        edge_count_file.set_len(
            if mmap > 1 { self.graph.size() - 1 } else { 1 }
                * std::mem::size_of::<AtomicU8>() as u64,
        )?;

        let mut edge_vec: Vec<AtomicU64> = vec![];
        if mmap < 1 {
            edge_vec.resize_with(node_count, || AtomicU64::new(0))
        }
        let mut edge_count_vec: Vec<AtomicU8> = vec![];
        if mmap < 2 {
            edge_count_vec.resize_with(node_count, || AtomicU8::new(0))
        }

        let (edge, edge_mmap): (Arc<SharedSliceMut<AtomicU64>>, MmapMut) = {
            if mmap < 1 {
                unsafe {
                    (
                        Arc::new(SharedSliceMut::<AtomicU64>::new(
                            edge_vec.as_mut_ptr(),
                            edge_vec.len(),
                        )),
                        MmapOptions::new().map_mut(&edge_vec_file)?,
                    )
                }
            } else {
                let slice = SharedSliceMut::<AtomicU64>::from_file(&edge_vec_file)?;
                (Arc::new(slice.0), slice.1)
            }
        };
        let (count, edge_count_mmap): (Arc<SharedSliceMut<AtomicU8>>, MmapMut) = {
            if mmap < 2 {
                unsafe {
                    (
                        Arc::new(SharedSliceMut::<AtomicU8>::new(
                            edge_count_vec.as_mut_ptr(),
                            edge_count_vec.len(),
                        )),
                        MmapOptions::new().map_mut(&edge_count_file)?,
                    )
                }
            } else {
                let slice = SharedSliceMut::<AtomicU8>::from_file(&edge_count_file)?;
                (Arc::new(slice.0), slice.1)
            }
        };

        thread::scope(|scope| {
            let threads = self.graph.thread_count as usize;
            for i in 0..threads {
                let index = Arc::clone(&index_ptr);
                let edge = Arc::clone(&edge);
                let count = Arc::clone(&count);
                let begin = node_count / threads * i;
                let end = if i == threads - 1 {
                    node_count
                } else {
                    node_count / threads * (i + 1)
                };
                scope.spawn(move |_| {
                    for k in begin..end {
                        edge.get(k).store(*index.get(k), Ordering::Relaxed);
                        count
                            .get(k)
                            .store((*index.get(k + 1) - *index.get(k)) as u8, Ordering::Relaxed);
                    }
                });
            }
        })
        .unwrap();
        if mmap > 0 {
            edge_mmap.flush()?;
        }
        if mmap > 1 {
            edge_count_mmap.flush()?;
        }
        Ok((
            edge,
            count,
            TmpMemoryHelperStruct {
                _a: edge_vec,
                _b: edge_count_vec,
                _c: edge_mmap,
                _d: edge_count_mmap,
            },
        ))
    }

    /// find Eulerian cycle and write sequence of node ids to memory-mapped file.
    /// num_threads controls parallelism level (defaults to 1, single-threaded).
    /// returns vec of (euler path file sequence number, file size(vytes)).
    pub fn find_eulerian_cycle(&self, mmap: u8) -> Result<Vec<(u64, u64)>, Error> {
        let node_count = match (self.graph.size() - 1) as usize {
            0 => panic!("Graph has no vertices"),
            i => i,
        };
        let graph_ptr = Arc::new(SharedSlice::<T>::new(
            self.graph.graph.as_ptr() as *const T,
            self.graph.width() as usize,
        ));

        // The Vec<_> and MmapMut refs need to be in scope for the structures not to be deallocated
        let (edge_vec, edge_count, _procedural_memory_ref) =
            self.initialize_hierholzers_procedural_memory(mmap)?;

        // Atomic counter to pick next starting vertex for a new cycle
        let start_vertex_counter = Arc::new(AtomicU64::new(0));
        // mmap to store disjoined trails for subsequent merging
        let filename = cache_file_name(
            self.graph.graph_cache.graph_filename.clone(),
            FileType::EulerTmp,
            None,
        )?;
        let (mmap_slice, mmap) = Self::create_memmapped_mut_slice_from_tmp_file::<u64>(
            filename,
            (self.graph.width() * 2) as usize,
        )?;
        let mmap_mutex = Mutex::new(mmap_slice);

        let cycle_offsets: Mutex<Vec<(usize, usize, usize)>> = std::sync::Mutex::new(vec![]);
        let mmap_offset: Mutex<usize> = Mutex::new(0);

        thread::scope(|scope| {
            for _ in 0..self.graph.thread_count {
                let graph = Arc::clone(&graph_ptr);
                let next_edge = Arc::clone(&edge_vec);
                let edge_count = Arc::new(&edge_count);
                let start_vertex_counter = Arc::clone(&start_vertex_counter);
                let cycle_offsets = &cycle_offsets;
                let mmap_offset = &mmap_offset;
                let mmap = &mmap_mutex;
                let node_count = node_count as u64;

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
                            if edge_count.get(idx as usize).load(Ordering::Relaxed) > 0 {
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
                        let mut stack: Vec<u64> = Vec::new();
                        let mut cycle: Vec<u64> = Vec::new();
                        stack.push(start_v);
                        while let Some(&v) = stack.last() {
                            if edge_count
                                .get(v as usize)
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                                    if x > 0 {
                                        Some(x - 1)
                                    } else {
                                        None // don't update
                                    }
                                })
                                .is_ok()
                            {
                                let edge_idx =
                                    next_edge.get(v as usize).fetch_add(1, Ordering::Relaxed);
                                stack.push(graph.get(edge_idx as usize).dest());
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
    pub fn find_eulerian_cycle_no_mmap(&self) -> Result<Vec<(u64, u64)>, Error> {
        let node_count = match (self.graph.size() - 1) as usize {
            0 => panic!("Graph has no vertices"),
            i => i,
        };

        let index_ptr = SharedSlice::<u64>::new(
            self.graph.index.as_ptr() as *const u64,
            self.graph.size() as usize,
        );
        let graph_ptr = Arc::new(SharedSlice::<T>::new(
            self.graph.graph.as_ptr() as *const T,
            self.graph.width() as usize,
        ));

        // atomic array for next unused edge index for each node and unread edge count
        let mut next_edge_vec: Vec<AtomicU64> = Vec::with_capacity(node_count);
        next_edge_vec.resize_with(node_count, || AtomicU64::new(0));
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
        let start_vertex_counter = Arc::new(AtomicU64::new(0));
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
                    let mut local_cycles: Vec<Vec<u64>> = Vec::new();
                    // find cycles until no unused edges remain
                    loop {
                        let start_v = loop {
                            let idx = start_vertex_counter
                                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                                    if x >= node_count as u64 {
                                        Some(0)
                                    } else {
                                        Some(x + 1)
                                    }
                                })
                                .unwrap_or(0);
                            if idx >= node_count as u64 {
                                break None;
                            }
                            if edge_count[idx as usize].load(Ordering::Relaxed) > 0 {
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
                        let mut stack: Vec<u64> = Vec::new();
                        let mut cycle: Vec<u64> = Vec::new();
                        stack.push(start_v);
                        while let Some(&v) = stack.last() {
                            // get next unused edge from v
                            if edge_count[v as usize]
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                                    if x > 0 {
                                        Some(x - 1)
                                    } else {
                                        None // don't update
                                    }
                                })
                                .is_ok()
                            {
                                let edge_idx =
                                    next_edge[v as usize].fetch_add(1, Ordering::Relaxed);
                                stack.push(graph_ptr.get(edge_idx as usize).dest());
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
        self.merge_euler_trails_no_mmap(cycles_mutex)
    }
}

#[derive(Clone)]
pub struct GraphMemoryMap<
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
> {
    graph: Arc<Mmap>,
    index: Arc<Mmap>,
    kmers: Arc<Map<Mmap>>,
    graph_cache: GraphCache<T>,
    edge_size: usize,
    thread_count: u8,
    _marker: PhantomData<U>,
}

impl<T, U> GraphMemoryMap<T, U>
where
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
{
    pub fn init(cache: GraphCache<T>, thread_count: u8) -> Result<GraphMemoryMap<T, U>, Error> {
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
                edge_size: std::mem::size_of::<T>(),
                thread_count,
                _marker: PhantomData,
            });
        }

        panic!("error graph cache must be readonly to be memmapped");
    }

    #[inline(always)]
    pub fn _node_degree(&self, node_id: u64) -> u64 {
        unsafe {
            let ptr = (self.index.as_ptr() as *const u64).add(node_id as usize);
            let begin = ptr.read_unaligned();
            ptr.add(1).read_unaligned() - begin
        }
    }

    #[inline(always)]
    pub fn node_id_from_kmer(&self, kmer: &str) -> Result<u64, Error> {
        if let Some(val) = self.kmers.get(kmer) {
            Ok(val)
        } else {
            panic!("error k-mer {} not found", kmer);
        }
    }

    #[inline(always)]
    pub fn _index_node(&self, node_id: u64) -> std::ops::Range<u64> {
        unsafe {
            let ptr = (self.index.as_ptr() as *const u64).add(node_id as usize);
            ptr.read_unaligned()..ptr.add(1).read_unaligned()
        }
    }

    pub fn neighbours(&self, node_id: u64) -> Result<NeighbourIter<T, U>, Error> {
        if node_id >= self.size() {
            panic!("error invalid range");
        }

        Ok(NeighbourIter::<T, U>::new(
            self.graph.as_ptr() as *const T,
            self.index.as_ptr() as *const u64,
            node_id,
        ))
    }

    pub fn edges(&self) -> Result<EdgeIter<T, U>, Error> {
        Ok(EdgeIter::<T, U>::new(
            self.graph.as_ptr() as *const T,
            self.index.as_ptr() as *const u64,
            0,
            self.size(),
        ))
    }

    pub fn _edges_in_range(&self, start_node: u64, end_node: u64) -> Result<EdgeIter<T, U>, Error> {
        if start_node > end_node {
            panic!("error invalid range, beginning after end");
        }
        if start_node > self.size() || end_node > self.size() {
            panic!("error invalid range");
        }

        Ok(EdgeIter::<T, U>::new(
            self.graph.as_ptr() as *const T,
            self.index.as_ptr() as *const u64,
            start_node,
            end_node,
        ))
    }

    pub fn size(&self) -> u64 {
        self.graph_cache.index_bytes / BYTES_U64 // index size stored as bits
    }

    pub fn width(&self) -> u64 {
        self.graph_cache.graph_bytes // graph size stored as edges
    }

    fn initialize_k_core_procedural_memory(
        &self,
        mmap: u8,
    ) -> Result<
        (
            SharedSliceMut<AtomicU8>,
            Vec<AtomicU8>,
            MmapMut,
            SharedSliceMut<usize>,
            Vec<usize>,
            MmapMut,
            SharedSliceMut<u8>,
            Vec<u8>,
            MmapMut,
            SharedSliceMut<usize>,
            Vec<usize>,
            MmapMut,
        ),
        Error,
    > {
        let node_count = (self.size() - 1) as usize;
        // Memory-map a file for degrees (each entry AtomicU32)
        let deg_filename = cache_file_name(
            self.graph_cache.graph_filename.clone(),
            FileType::KCore, // assume FileType::CoreTmp is defined for temp files
            Some(0),
        )?;
        let deg_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&deg_filename)?;
        deg_file.set_len(
            (if mmap > 0 { node_count } else { 1 } * std::mem::size_of::<AtomicU8>()) as u64,
        )?;
        // Use an in-memory vector for degrees
        let mut deg_vec: Vec<AtomicU8> = Vec::new();
        // Initialize with 0s
        if mmap < 1 {
            deg_vec.resize_with(node_count, || AtomicU8::new(0));
        }
        let node_filename = cache_file_name(
            self.graph_cache.graph_filename.clone(),
            FileType::KCore, // assume FileType::CoreTmp is defined for temp files
            Some(1),
        )?;
        let node_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&node_filename)?;
        node_file.set_len(
            (if mmap > 1 { node_count } else { 1 } * std::mem::size_of::<usize>()) as u64,
        )?;
        // Use an in-memory vector for degrees
        let mut node_vec: Vec<usize> = Vec::new();
        // Initialize with 0s
        if mmap < 2 {
            node_vec.resize_with(node_count, || 0);
        }
        let core_filename = cache_file_name(
            self.graph_cache.graph_filename.clone(),
            FileType::KCore, // assume FileType::CoreTmp is defined for temp files
            Some(2),
        )?;
        let core_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&core_filename)?;
        core_file
            .set_len((if mmap > 2 { node_count } else { 1 } * std::mem::size_of::<u8>()) as u64)?;
        // Use an in-memory vector for degrees
        let mut core_vec: Vec<u8> = Vec::new();
        // Initialize with 0s
        if mmap < 3 {
            core_vec.resize_with(node_count, || 0);
        }
        let pos_filename = cache_file_name(
            self.graph_cache.graph_filename.clone(),
            FileType::KCore, // assume FileType::CoreTmp is defined for temp files
            Some(3),
        )?;
        let pos_file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&pos_filename)?;
        pos_file.set_len(
            (if mmap > 3 { node_count } else { 1 } * std::mem::size_of::<usize>()) as u64,
        )?;
        // Use an in-memory vector for degrees
        let mut pos_vec: Vec<usize> = Vec::new();
        // Initialize with 0s
        if mmap < 4 {
            pos_vec.resize_with(node_count, || 0);
        }

        let (degree, deg_mmap): (SharedSliceMut<AtomicU8>, MmapMut) = {
            if mmap < 1 {
                unsafe {
                    (
                        SharedSliceMut::<AtomicU8>::new(deg_vec.as_mut_ptr(), deg_vec.len()),
                        MmapOptions::new().map_mut(&deg_file)?,
                    )
                }
            } else {
                let slice = SharedSliceMut::<AtomicU8>::from_file(&deg_file)?;
                (slice.0, slice.1)
            }
        };
        let (node, node_mmap): (SharedSliceMut<usize>, MmapMut) = {
            if mmap < 2 {
                unsafe {
                    (
                        SharedSliceMut::<usize>::new(node_vec.as_mut_ptr(), node_vec.len()),
                        MmapOptions::new().map_mut(&node_file)?,
                    )
                }
            } else {
                let slice = SharedSliceMut::<usize>::from_file(&node_file)?;
                (slice.0, slice.1)
            }
        };
        let (core, core_mmap): (SharedSliceMut<u8>, MmapMut) = {
            if mmap < 3 {
                unsafe {
                    (
                        SharedSliceMut::<u8>::new(core_vec.as_mut_ptr(), core_vec.len()),
                        MmapOptions::new().map_mut(&core_file)?,
                    )
                }
            } else {
                let slice = SharedSliceMut::<u8>::from_file(&core_file)?;
                (slice.0, slice.1)
            }
        };
        let (pos, pos_mmap): (SharedSliceMut<usize>, MmapMut) = {
            if mmap < 4 {
                unsafe {
                    (
                        SharedSliceMut::<usize>::new(pos_vec.as_mut_ptr(), pos_vec.len()),
                        MmapOptions::new().map_mut(&pos_file)?,
                    )
                }
            } else {
                let slice = SharedSliceMut::<usize>::from_file(&pos_file)?;
                (slice.0, slice.1)
            }
        };
        Ok((
            degree, deg_vec, deg_mmap, node, node_vec, node_mmap, core, core_vec, core_mmap, pos,
            pos_vec, pos_mmap,
        ))
    }

    pub fn compute_k_core(&self, mmap: u8) -> Result<String, Error> {
        let node_count = (self.size() - 1) as usize;
        let edge_count = self.width() as usize;
        if node_count == 0 {
            return Err(Error::new(
                std::io::ErrorKind::Other,
                "Graph has no vertices",
            ));
        }

        let (
            mut degree,
            _deg_vec,
            _deg_mmap,
            mut node,
            _node_vec,
            _node_mmap,
            mut core,
            _core_vec,
            _core_mmap,
            mut pos,
            _pos_vec,
            _pos_mmap,
        ) = self.initialize_k_core_procedural_memory(mmap)?;

        // compute out-degrees in parallel
        let index_ptr = Arc::new(SharedSlice::<u64>::new(
            self.index.as_ptr() as *const u64,
            node_count + 1,
        ));
        let graph_ptr = Arc::new(SharedSlice::<T>::new(
            self.graph.as_ptr() as *const T,
            edge_count,
        ));

        // initialize degree and bins count vecs
        let mut bins: Vec<usize> = match thread::scope(|scope| {
            let threads = self.thread_count.max(1) as usize;
            let mut bins = Vec::new();
            let mut max_vecs = Vec::new();
            for t in 0..threads {
                let index_ptr = Arc::clone(&index_ptr);
                let deg_arr = &degree;
                let start = node_count * t / threads;
                let end = if t == threads - 1 {
                    node_count
                } else {
                    node_count * (t + 1) / threads
                };
                max_vecs.push(scope.spawn(move |_| {
                    let mut bins: Vec<u64> = vec![0; 1];
                    for v in start..end {
                        let degree = (index_ptr.get(v + 1) - index_ptr.get(v)) as u8;
                        if degree >= bins.len() as u8 {
                            println!("bins {:?}", bins);
                            bins.resize_with((degree + 1) as usize, || 0u64);
                            println!("bins {:?}", bins);
                            bins[degree as usize] += 1;
                        } else {
                            bins[degree as usize] += 1;
                        }
                        deg_arr.get(v).store(degree, Ordering::Relaxed);
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
                    if !bins.len() > degree {
                        bins.push(*count as usize);
                    } else {
                        bins[degree] += *count as usize;
                    }
                }
            });
            bins
        }) {
            Ok(i) => {
                if mmap > 0 {
                    _deg_mmap.flush()?;
                }
                i
            }
            _ => panic!("error calculating max degree"),
        };
        let max_degree = bins.len() - 1;

        // prefix sum to get starting indices for each degree
        let mut start_index = 0;
        for d in 0..bins.len() {
            let count = bins[d];
            bins[d] = start_index;
            start_index += count;
        }
        // `bins[d]` now holds the starting index in `vert` for vertices of degree d.
        // fill node array with vertices ordered by degree
        for v in 0..node_count {
            let d = degree.get(v).load(Ordering::Relaxed) as usize;
            let idx = bins[d] as usize;
            *node.get_mut(idx) = v;
            *pos.get_mut(v) = idx;
            bins[d] += 1; // increment the bin index for the next vertex of same degree
        }
        if mmap > 1 {
            _node_mmap.flush()?;
        }
        if mmap > 3 {
            _pos_mmap.flush()?;
        }

        // restore bin starting positions
        for d in (1..=max_degree).rev() {
            bins[d] = bins[d - 1];
        }
        bins[0] = 0;

        // peel vertices in order of increasing current degree
        for i in 0..node_count {
            let v = *node.get(i);
            let deg_v = degree.get(v).load(Ordering::Relaxed);
            *core.get_mut(v) = deg_v; // coreness of v

            // iterate outgoing neighbors of v
            let out_start = *(index_ptr.get(v)) as usize;
            let out_end = *(index_ptr.get(v + 1)) as usize;
            for e in out_start..out_end {
                let u = (*graph_ptr.get(e)).dest() as usize;
                degree
                    .get(u)
                    .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                        if x > deg_v {
                            // swap u's position node array to maintain order
                            let u_pos = *pos.get(u);
                            let new_pos = bins[x as usize];
                            // bins[old_deg] points to start of nodes with degree >= old_deg
                            // swap the node at new_pos with u to move u into the bucket of (u_new_degree)
                            let w = *node.get(new_pos);
                            if u != w {
                                *node.get_mut(u_pos) = w;
                                *pos.get_mut(w) = u_pos;
                                *node.get_mut(new_pos) = u;
                                *pos.get_mut(u) = new_pos;
                            }
                            bins[x as usize] += 1;
                            Some(x - 1)
                        } else {
                            None // don't update
                        }
                    });
            }
        }

        if mmap > 2 {
            _core_mmap.flush()?;
        }

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
        let core = Arc::new(core);

        // parallel edge labeling: partition vertices among threads and write edge core values
        thread::scope(|scope| {
            let threads = self.thread_count.max(1) as usize;
            for t in 0..threads {
                let index_ptr = Arc::clone(&index_ptr);
                let graph_ptr = Arc::clone(&graph_ptr);
                let core = Arc::new(&core);
                let core = Arc::clone(&core);
                let start = node_count * t / threads;
                let end = if t == threads - 1 {
                    node_count
                } else {
                    node_count * (t + 1) / threads
                };
                let mut out_ptr = out_slice;
                scope.spawn(move |_| {
                    for u in start..end {
                        let core_u = *core.get(u);
                        let out_begin = *(index_ptr.get(u)) as usize;
                        let out_end = *(index_ptr.get(u + 1)) as usize;
                        for e in out_begin..out_end {
                            let v = (*graph_ptr.get(e)).dest() as usize;
                            // determine edge's core = min(core_u, core[v])
                            let core_val = if *core.get(u) < *core.get(v) {
                                core_u
                            } else {
                                *core.get(v)
                            };
                            *out_ptr.get_mut(e) = core_val;
                        }
                    }
                });
            }
        })
        .unwrap();

        // flush output to ensure all data is written to disk
        _mmap.flush()?;

        Ok(output_filename)
    }
}

#[derive(Debug)]
pub struct GraphIterator<'a, T: Copy + Debug + Display + Pod + Zeroable> {
    inner: &'a [T],
    pos: usize,
}

#[derive(Debug)]
pub struct NeighbourIter<
    'a,
    T: Copy + Pod + Zeroable + EdgeOutOf,
    U: Copy + Pod + Zeroable + GraphEdge,
> {
    edge_ptr: *const T,
    orig_edge_ptr: *const T,
    orig_id_ptr: *const u64,
    id: u64,
    count: u64,
    _phantom: std::marker::PhantomData<&'a U>,
}

#[derive(Debug)]
pub struct EdgeIter<'a, T: Copy + Pod + Zeroable + EdgeOutOf, U: Copy + Pod + Zeroable + GraphEdge>
{
    edge_ptr: *const T,
    id_ptr: *const u64,
    id: u64,
    end: u64,
    count: u64,
    _phantom: std::marker::PhantomData<&'a U>,
}

impl<'a, T: Copy + Pod + Zeroable + EdgeOutOf, U: Copy + Pod + Zeroable + GraphEdge>
    NeighbourIter<'a, T, U>
{
    fn new(edge_mmap: *const T, id_mmap: *const u64, node_id: u64) -> Self {
        let orig_edge_ptr = edge_mmap;
        let orig_id_ptr = id_mmap;
        let id_ptr = unsafe { id_mmap.add(node_id as usize) };
        let offset = unsafe { id_ptr.read_unaligned() };

        NeighbourIter {
            edge_ptr: unsafe { edge_mmap.add(offset as usize) },
            orig_edge_ptr,
            orig_id_ptr,
            id: node_id,
            count: unsafe { id_ptr.add(1).read_unaligned() - offset },
            _phantom: std::marker::PhantomData::<&'a U>,
        }
    }

    #[inline(always)]
    fn _into_neighbour(&self) -> Self {
        NeighbourIter::new(self.orig_edge_ptr, self.orig_id_ptr, unsafe {
            self.edge_ptr.read_unaligned().dest()
        })
    }
}

impl<'a, T: Copy + Pod + Zeroable + EdgeOutOf, U: Copy + Pod + Zeroable + GraphEdge> Iterator
    for NeighbourIter<'a, T, U>
{
    type Item = U;

    #[inline(always)]
    fn next(&mut self) -> Option<U> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        unsafe { Some(U::new(self.id, self.edge_ptr.add(1).read_unaligned())) }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.count) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy + Pod + Zeroable + EdgeOutOf, U: Copy + Pod + Zeroable + GraphEdge>
    EdgeIter<'a, T, U>
{
    #[inline(always)]
    fn new(edge_mmap: *const T, id_mmap: *const u64, start: u64, end: u64) -> Self {
        let id_ptr = unsafe { id_mmap.add(start as usize) };
        let offset = unsafe { id_ptr.read_unaligned() };
        let edge_ptr = unsafe { edge_mmap.add(offset as usize) };

        EdgeIter {
            edge_ptr,
            id_ptr,
            id: start,
            end,
            count: unsafe { id_ptr.add(1).read_unaligned() - offset },
            _phantom: std::marker::PhantomData::<&'a U>,
        }
    }
}

impl<'a, T: Copy + Pod + Zeroable + EdgeOutOf, U: Copy + Pod + Zeroable + GraphEdge> Iterator
    for EdgeIter<'a, T, U>
{
    type Item = U;

    #[inline(always)]
    fn next(&mut self) -> Option<U> {
        if self.count == 0 {
            self.id += 1;
            if self.id > self.end {
                return None;
            }

            let offset = unsafe { self.id_ptr.read_unaligned() };
            self.count = unsafe { self.id_ptr.add(1).read_unaligned() - offset };
        }
        unsafe { Some(U::new(self.id, self.edge_ptr.add(1).read_unaligned())) }
    }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = (self.end - self.id) as usize;
        (remaining, Some(remaining))
    }
}

impl<'a, T: Copy + Debug + Display + Pod + Zeroable> Iterator for GraphIterator<'a, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.inner.len() {
            None
        } else {
            self.pos += 1;
            Some(self.inner[self.pos - 1])
        }
    }
}

impl<
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
> std::ops::Index<std::ops::RangeFull> for GraphMemoryMap<T, U>
{
    type Output = [T];
    #[inline]
    fn index(&self, _index: std::ops::RangeFull) -> &[T] {
        unsafe {
            slice::from_raw_parts(
                self.graph.as_ptr() as *const T,
                (self.size() * 8) as usize / self.edge_size,
            )
        }
    }
}

impl<
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
> std::ops::Index<std::ops::Range<usize>> for GraphMemoryMap<T, U>
{
    type Output = [T];
    #[inline]
    fn index(&self, index: std::ops::Range<usize>) -> &[T] {
        unsafe {
            slice::from_raw_parts(
                self.graph.as_ptr().add(index.start * self.edge_size) as *const T,
                index.end - index.start,
            )
        }
    }
}

impl<
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
> std::ops::Index<std::ops::Range<u64>> for GraphMemoryMap<T, U>
{
    type Output = [T];
    #[inline]
    fn index(&self, index: std::ops::Range<u64>) -> &[T] {
        let start = index.start as usize;
        let end = index.end as usize;

        unsafe {
            slice::from_raw_parts(
                self.graph.as_ptr().add(start * self.edge_size) as *const T,
                end - start,
            )
        }
    }
}

impl<
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
> Debug for GraphMemoryMap<T, U>
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

impl PartialEq for OutEdgeRecord {
    fn eq(&self, other: &Self) -> bool {
        self.dest_node() == other.dest_node() && self.edge_type() == other.edge_type()
    }
}

impl Eq for OutEdgeRecord {}

impl Hash for OutEdgeRecord {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dest_node().hash(state);
    }
}

impl Debug for OutEdgeRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Edge(type: {:?}, dest: {:?})",
            self.edge_type(),
            self.dest_node()
        )
    }
}

impl Display for OutEdgeRecord {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{}, {}}}", self.edge_type(), self.dest_node())
    }
}

impl PartialEq for DirectedEdge {
    fn eq(&self, other: &Self) -> bool {
        self.origin() == other.origin()
            && self.dest() == other.dest()
            && self.edge_type() == other.edge_type()
    }
}

impl Eq for DirectedEdge {}

impl Hash for DirectedEdge {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.origin().hash(state);
        self.dest().hash(state);
    }
}

impl Debug for DirectedEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Edge(type: {:?}, origin: {:?}, dest: {:?})",
            self.edge_type(),
            self.origin(),
            self.dest()
        )
    }
}

impl Display for DirectedEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{{}, {}, {}}}",
            self.edge_type(),
            self.origin(),
            self.dest()
        )
    }
}

impl<T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf> Debug for GraphCache<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n\tgraph filename: {}\n\tindex filename: {}\n\tkmer filename: {}\n}}",
            self.graph_filename, self.index_filename, self.kmer_filename
        )
    }
}

impl<T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf> Clone for GraphCache<T> {
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
            _marker: self._marker,
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
        };
        write!(f, "{}", s)
    }
}
