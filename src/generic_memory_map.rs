use crate::node::EdgeType;

use bitfield::bitfield;
use bytemuck::{Pod, Zeroable};
use core::panic;
use crossbeam::thread;
use fst::{Map, MapBuilder};
use glob::glob;
use memmap::{Mmap, MmapMut, MmapOptions};
use rand::seq::IndexedRandom;
use regex::Regex;
use static_assertions::const_assert;
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
    usize,
};
use zerocopy::*; // Using crossbeam for scoped threads

const_assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u64>());

static CACHE_DIR: &str = "./cache/";
static BYTES_U64: u64 = std::mem::size_of::<u64>() as u64;

#[derive(Copy, Clone)]
struct SharedPtr(*const u8);

// SAFETY: The underlying data is in a memory-mapped file and not mutated across threads.
unsafe impl Send for SharedPtr {}
unsafe impl Sync for SharedPtr {}

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

enum FileType {
    Edges,
    Index,
    Fst,
    EulerPath,
    EulerTmp,
    KmerTmp,
    KmerSortedTmp,
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
    let re = match Regex::new(r#"^(?:[a-zA-Z0-9_]+_)(\d+)(\.[a-zA-Z0-9]+$)"#).ok() {
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
    fn init_cache_file_from_random(target_type: FileType) -> Result<String, Error> {
        let id = rand::random::<u64>().to_string();
        Ok(match target_type {
            FileType::Edges => format!("{}{}_{}.{}", CACHE_DIR, "edges", id, "mmap"),
            FileType::Index => format!("{}{}_{}.{}", CACHE_DIR, "index", id, "mmap"),
            FileType::Fst => format!("{}{}_{}.{}", CACHE_DIR, "fst", id, "fst"),
            FileType::EulerPath => format!("{}{}_{}.{}", CACHE_DIR, "eulerpath", id, "mmap"),
            FileType::EulerTmp => format!("{}{}_{}.{}", CACHE_DIR, "eulertmp", id, "tmp"),
            FileType::KmerTmp => format!("{}{}_{}.{}", CACHE_DIR, "kmertmpfile", id, "tmp"),
            FileType::KmerSortedTmp => {
                format!("{}{}_{}.{}", CACHE_DIR, "kmersortedtmpfile", id, "tmp")
            }
        })
    }
    pub fn init() -> Result<GraphCache<T>, Error> {
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR)?;
        }

        let graph_filename = Self::init_cache_file_from_random(FileType::Edges)?;
        let index_filename = cache_file_name(graph_filename.clone(), FileType::Index, None)?;
        let kmer_filename = cache_file_name(graph_filename.clone(), FileType::KmerTmp, None)?;

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

        let reader = BufReader::new(sorted_file);

        for line in reader.lines() {
            let line = line?;
            let mut parts = line.splitn(2, '\t');
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
    trails: Vec<((u64, u64, u64), u64)>,
    cycle_check: bool,
}

impl FindDisjointSetsEulerTrails {
    pub fn new(trails: &mut [(u64, u64, u64)]) -> Self {
        trails.sort_by_key(|(trail_id, _, _)| *trail_id);
        FindDisjointSetsEulerTrails {
            trails: trails
                .iter()
                .map(|(trail, parent_trail, pos)| ((*trail, *parent_trail, *pos), *parent_trail))
                .collect::<Vec<((u64, u64, u64), u64)>>(),
            cycle_check: false,
        }
    }

    fn cycle_b(t: &mut [((u64, u64, u64), u64)], visited: &mut [bool], i: u64) -> u64 {
        let i_usize = i as usize;
        if t[i_usize].0.0 == t[i_usize].1 {
            return t[i_usize].1;
        }

        if visited[t[i_usize].0.0 as usize] {
            t[i_usize].1 = i;
            t[i_usize].0.1 = i;
            t[i_usize].0.2 = 0;
            i
        } else {
            visited[t[i_usize].0.0 as usize] = true;
            t[i_usize].1 = Self::cycle_b(t, visited, t[i_usize].1);
            t[i_usize].1
        }
    }

    fn cycle_break(&mut self, i: u64) -> u64 {
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
            self.cycle_break(id as u64);
        }
        // Works as an union find in tree
        for (id, _i) in self.clone().trails.iter().enumerate() {
            // println!("check {}: {} - -> {}", id, i.0.0, i.1);
            self.cycle_break(id as u64);
        }

        self.cycle_check = true;
    }
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

    fn next_mmapped_chunck_into_slice<'a>(
        mmap_pointer: *const u8,
        mmap_remaining: usize,
    ) -> Option<&'a [u64]> {
        if mmap_remaining == 0 {
            return None;
        }
        if mmap_remaining > 4096 {
            // in bytes it's equivalent to for pages of 4Kb
            unsafe { Some(std::slice::from_raw_parts(mmap_pointer as *const u64, 4096)) }
        } else {
            unsafe {
                Some(std::slice::from_raw_parts(
                    mmap_pointer as *const u64,
                    mmap_remaining,
                ))
            }
        }
    }

    fn merge_euler_trails(
        &self,
        cycle_offsets: Vec<(u64, u64, u64)>,
    ) -> Result<Vec<(u64, u64)>, Error> {
        let mut trail_heads: HashMap<u64, Vec<(u64, u64, u64)>> = HashMap::new();
        let mmap_filename = cache_file_name(
            self.graph.graph_cache.graph_filename.clone(),
            FileType::EulerTmp,
            None,
        )?;
        let cycles = unsafe {
            MmapOptions::new().map(
                &OpenOptions::new()
                    .read(true)
                    .open(mmap_filename)
                    .map_err(|e| e)?,
            )?
        };

        cycle_offsets.iter().for_each(|(idx, begin, end)| {
            let first = unsafe {
                std::ptr::read_unaligned(cycles.as_ptr().clone().add(*begin as usize) as *const u64)
            };
            trail_heads.entry(first).or_default().push((*idx, *idx, 0));
        });

        // generate writing sets
        for (t_idx, t_begin, t_end) in cycle_offsets.iter() {
            let mmap_ptr = unsafe { cycles.as_ptr().clone().add(*t_begin as usize) };
            let mmap_remaining_size = (*t_end - *t_begin) as usize / std::mem::size_of::<u64>();
            let mut pos = 0; // pos in u64 terms
            while let Some(next_slice) =
                Self::next_mmapped_chunck_into_slice(mmap_ptr, mmap_remaining_size - pos)
            {
                for (pos_idx, node) in next_slice.iter().enumerate() {
                    if let Some(head_v) = trail_heads.get_mut(node) {
                        let c_idx = *t_idx;
                        let p_idx = (pos_idx + pos + 1) as u64;
                        for (vec_idx, (in_cyc, _, _)) in head_v.clone().iter().enumerate() {
                            if *in_cyc == *t_idx {
                                continue;
                            }
                            head_v[vec_idx] = (*in_cyc, c_idx, p_idx);
                        }
                    }
                }
                // Prepare to read next slice
                unsafe { mmap_ptr.add(next_slice.len() * std::mem::size_of::<u64>()) };
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
        let mut trail_sets: HashMap<u64, Vec<(u64, u64, u64)>> = HashMap::new();

        for ((trail, parent_trail, pos), grand_parent) in euler_trail_sets.trails {
            trail_sets
                .entry(grand_parent)
                .or_default()
                .push((trail, parent_trail, pos));
        }

        let mut output_len: HashMap<u64, (u64, u64)> = HashMap::new();
        let cycles_length: Vec<u64> = cycle_offsets.iter().map(|(_, b, e)| e - b).collect();
        for head_trail in trail_sets.keys() {
            output_len.insert(
                *head_trail,
                (*head_trail, std::mem::size_of::<u64>() as u64),
            );
            for (trail, _, _) in trail_sets.get(head_trail).unwrap() {
                output_len.get_mut(head_trail).unwrap().1 +=
                    cycles_length[*trail as usize] - std::mem::size_of::<u64>() as u64;
            }
        }

        let mut keys_by_trail_size: Vec<(u64, u64)> =
            output_len.values().map(|&(a, b)| (a, b)).collect();
        keys_by_trail_size.sort_by_key(|(_, s)| std::cmp::Reverse(*s));

        for (idx, (head_trail, output_len)) in keys_by_trail_size.iter().enumerate() {
            // Initialize writing guide
            let trail_guide = trail_sets.get(head_trail).unwrap();
            let mut pos_map: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
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
            let mut stack: Vec<(u64, u64, u64)> = vec![];
            let mut remaining: Vec<(u64, u64)> = vec![];
            let mut expand = Some((*head_trail, 0));
            while let Some((current_trail, pos)) = expand {
                if let Some(nested_trails) = pos_map.get_mut(&current_trail) {
                    if let Some((insert_pos, trail)) = nested_trails.pop() {
                        if trail == current_trail
                            || cycles_length[trail as usize] == 0
                            || insert_pos < pos
                        {
                            continue;
                        }
                        stack.push((current_trail, pos, insert_pos));
                        remaining.push((current_trail, insert_pos));
                        remaining.push((trail, 1)); // elipse repeated node
                    } else {
                        stack.push((current_trail, pos, cycles_length[current_trail as usize]));
                    }
                } else {
                    stack.push((current_trail, pos, cycles_length[current_trail as usize]));
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
            output_file.set_len(*output_len)?;

            let mut output_mmap = unsafe { MmapOptions::new().map_mut(&output_file)? };
            let mut write_offset = 0;

            for (cycle, from, to) in stack.iter() {
                if *to <= *from {
                    continue;
                }
                let (_, t_begin, _) = cycle_offsets[*cycle as usize];
                let begin = (t_begin + *from) as usize;
                let end = (t_begin + *to) as usize;

                let slice_ptr = unsafe { cycles.as_ptr().clone().add(begin) };
                // in u64 nodes
                let slice_remaining_size = (end - begin) / std::mem::size_of::<u64>();
                // in u64 nodes
                let mut pos = 0;
                while let Some(next_slice) =
                    Self::next_mmapped_chunck_into_slice(slice_ptr, slice_remaining_size - pos)
                {
                    let byte_len = next_slice.as_bytes().len();
                    unsafe {
                        let dest_ptr = output_mmap.as_mut_ptr().add(write_offset);
                        let src_ptr = next_slice.as_ptr() as *const u8;
                        std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, byte_len);
                    }
                    write_offset += byte_len;
                    // Prepare to read next slice
                    unsafe { slice_ptr.add(byte_len) };
                    pos += next_slice.len();
                }

                output_mmap.flush()?;
            }
        }
        Ok(keys_by_trail_size
            .iter()
            .enumerate()
            .map(|(idx, (_, size))| (idx as u64, *size))
            .collect())
    }

    fn merge_euler_trails_no_mmap(
        &self,
        cycles: Mutex<Vec<Vec<u64>>>,
    ) -> Result<Vec<(u64, u64)>, Error> {
        let mut trail_heads: HashMap<u64, Vec<(u64, u64, u64)>> = HashMap::new();
        let cycles = cycles.into_inner().unwrap();
        cycles.iter().enumerate().for_each(|(idx, trail)| {
            if let Some(first) = trail.first() {
                trail_heads
                    .entry(*first)
                    .or_default()
                    .push((idx as u64, idx as u64, 0));
            }
        });

        // generate writing sets
        for (cycle_idx, cycle) in cycles.iter().enumerate() {
            for (trail_idx, node) in cycle.iter().enumerate().skip(1) {
                if let Some(head_v) = trail_heads.get_mut(node) {
                    let c_idx = cycle_idx as u64;
                    let p_idx = (trail_idx + 1) as u64;
                    for (vec_idx, (in_cyc, _, _)) in head_v.clone().iter().enumerate() {
                        if *in_cyc == cycle_idx as u64 {
                            continue;
                        }
                        head_v[vec_idx] = (*in_cyc, c_idx, p_idx);
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
        let mut trail_sets: HashMap<u64, Vec<(u64, u64, u64)>> = HashMap::new();

        for ((trail, parent_trail, pos), grand_parent) in euler_trail_sets.trails {
            trail_sets
                .entry(grand_parent)
                .or_default()
                .push((trail, parent_trail, pos));
        }

        // Get trail sizes for each soon to be merged trails to establish order
        let mut output_len: HashMap<u64, (u64, u64)> = HashMap::new();
        let cycles_length: Vec<u64> = cycles.iter().map(|x| x.len() as u64).collect();
        for head_trail in trail_sets.keys() {
            output_len.insert(*head_trail, (*head_trail, 1));
            for (trail, _, _) in trail_sets.get(head_trail).unwrap() {
                output_len.get_mut(head_trail).unwrap().1 += cycles_length[*trail as usize] - 1;
            }
            // Adjust output length to size in bytes
            output_len.get_mut(head_trail).unwrap().1 *= std::mem::size_of::<u64>() as u64;
        }

        let mut keys_by_trail_size: Vec<(u64, u64)> =
            output_len.values().map(|&(a, b)| (a, b)).collect();
        keys_by_trail_size.sort_by_key(|(_, s)| std::cmp::Reverse(*s));

        // Write
        for (idx, (head_trail, output_len)) in keys_by_trail_size.iter().enumerate() {
            let trail_guide = trail_sets.get(head_trail).unwrap();
            let mut pos_map: HashMap<u64, Vec<(u64, u64)>> = HashMap::new();
            let cycles_length: Vec<u64> = cycles.iter().map(|x| x.len() as u64).collect();
            for (trail, parent_trail, pos) in trail_guide {
                pos_map
                    .entry(*parent_trail)
                    .or_default()
                    .push((*pos, *trail));
            }
            pos_map
                .values_mut()
                .for_each(|v| v.sort_by_key(|(pos, _)| std::cmp::Reverse(*pos)));

            let mut stack: Vec<(u64, u64, u64)> = vec![];
            let mut remaining: Vec<(u64, u64)> = vec![];
            let mut expand = Some((*head_trail, 0));
            while let Some((current_trail, pos)) = expand {
                if let Some(nested_trails) = pos_map.get_mut(&current_trail) {
                    if let Some((insert_pos, trail)) = nested_trails.pop() {
                        if trail == current_trail {
                            continue;
                        }
                        if cycles_length[trail as usize] == 0 {
                            continue;
                        }
                        if insert_pos < pos {
                            continue;
                        }
                        stack.push((current_trail, pos, insert_pos));
                        remaining.push((current_trail, insert_pos));
                        remaining.push((trail, 1)); // elipse repeated node
                    } else {
                        stack.push((current_trail, pos, cycles_length[current_trail as usize]));
                    }
                } else {
                    stack.push((current_trail, pos, cycles_length[current_trail as usize]));
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
            output_file.set_len(*output_len)?;

            let mut output_mmap = unsafe { MmapOptions::new().map_mut(&output_file)? };
            let mut write_offset = 0;

            for (cycle, from, to) in stack.iter() {
                if *to <= *from {
                    continue;
                }
                let slice = &cycles[*cycle as usize][*from as usize..*to as usize];
                let byte_len = slice.as_bytes().len();
                unsafe {
                    let dest_ptr = output_mmap.as_mut_ptr().add(write_offset);
                    let src_ptr = slice.as_ptr() as *const u8;
                    std::ptr::copy_nonoverlapping(src_ptr, dest_ptr, byte_len);
                }
                write_offset += byte_len;
            }

            output_mmap.flush()?;
            // println!("map has {} bytes", *output_len);
        }

        Ok(keys_by_trail_size
            .iter()
            .enumerate()
            .map(|(idx, (_, size))| (idx as u64, *size))
            .collect())
    }

    fn create_mmap_for_concurrent_data_write(
        &self,
        size: u64,
        id: Option<u64>,
    ) -> Result<Mutex<MmapMut>, Error> {
        let filename = cache_file_name(
            self.graph.graph_cache.graph_filename.clone(),
            FileType::EulerTmp,
            id,
        )?;

        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(filename)?;
        file.set_len(size)?;
        Ok(Mutex::new(unsafe { MmapMut::map_mut(&file)? }))
    }

    /// find Eulerian cycle and write sequence of node ids to memory-mapped file.
    /// num_threads controls parallelism level (defaults to 1, single-threaded).
    /// returns vec of (euler path file sequence number, file size(vytes)).
    pub fn find_eulerian_cycle(&self, mmap: u8) -> Result<Vec<(u64, u64)>, Error> {
        let node_count = match (self.graph.size() - 1) as usize {
            0 => panic!("Graph has no vertices"),
            i => i,
        };

        let index_ptr = self.graph.index.as_ptr() as *const u64;
        let graph_ptr = Arc::new(SharedPtr(self.graph.graph.as_ptr()));
        let edge_size = self.graph.edge_size;

        // atomic array for next unused edge index for each node and unread edge count
        let mut next_edge_vec: Vec<AtomicU64> = Vec::with_capacity(node_count);
        next_edge_vec.resize_with(node_count, || AtomicU64::new(0));
        let mut edge_count: Vec<AtomicU8> = Vec::with_capacity(node_count);
        edge_count.resize_with(node_count, || AtomicU8::new(0));

        unsafe {
            for i in 0..node_count {
                next_edge_vec[i].store(*index_ptr.add(i), Ordering::Relaxed);
                edge_count[i].store(
                    (*index_ptr.add(i + 1) - *index_ptr.add(i)) as u8,
                    Ordering::Relaxed,
                );
            }
        }
        let next_edge = Arc::new(next_edge_vec);
        // mutex is needed to make sure a read isn't made between a successful check and a decrease
        let edge_count = Arc::new(edge_count);

        // Atomic counter to pick next starting vertex for a new cycle
        let start_vertex_counter = Arc::new(AtomicU64::new(0));
        // mmap to store disjoined trails for subsequent merging
        let mmap_write = self.create_mmap_for_concurrent_data_write(
            (self.graph.width() * 2) * std::mem::size_of::<u64>() as u64,
            None,
        )?;

        let cycle_offsets: Mutex<Vec<(u64, u64, u64)>> = std::sync::Mutex::new(vec![]);
        let mmap_offset: Mutex<u64> = Mutex::new(0);

        thread::scope(|scope| {
            for _ in 0..self.graph.thread_count {
                let graph_ptr = Arc::clone(&graph_ptr);
                let next_edge = Arc::clone(&next_edge);
                let edge_count = Arc::new(&edge_count);
                let start_vertex_counter = Arc::clone(&start_vertex_counter);
                let cycle_offsets = &cycle_offsets;
                let mmap_offset = &mmap_offset;
                let mmap = &mmap_write;

                // Spawn a thread
                scope.spawn(move |_| {
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
                                break;
                            }
                        };

                        // Hierholzer's DFS
                        let mut stack: Vec<u64> = Vec::new();
                        let mut cycle: Vec<u64> = Vec::new();
                        stack.push(start_v);
                        while let Some(&v) = stack.last() {
                            // Get the next unused edge from v
                            // let mut end_offsets = end_offsets.lock().unwrap();
                            // if end_offsets[v as usize] != 0 {
                            //     end_offsets[v as usize] -= 1;
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
                                let neighbor: u64;
                                unsafe {
                                    neighbor = std::ptr::read_unaligned(
                                        graph_ptr.0.add((edge_idx * (edge_size as u64)) as usize)
                                            as *const T,
                                    )
                                    .dest();
                                }
                                stack.push(neighbor);
                            } else {
                                // No more unused edges from v -> backtrack
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
                            let cycle = cycle.as_bytes();
                            let mut cycle_stack = cycle_offsets.lock().unwrap();
                            let mut offset = mmap_offset.lock().unwrap();
                            let begin = *offset;
                            *offset += cycle.len() as u64;
                            let end = *offset;
                            let cycle_id = cycle_stack.len() as u64;
                            cycle_stack.push((cycle_id, begin, end));

                            // Write to mmap, this may be done concurrently
                            let mut mmap_guard = match mmap.lock() {
                                Ok(i) => i,
                                Err(e) => panic!("error mutex 1: {:?}", e),
                            };
                            mmap_guard[begin as usize..end as usize].copy_from_slice(cycle);
                        }
                    }
                });
            }
        })
        .unwrap();
        match mmap_write.into_inner() {
            Ok(i) => i.flush()?,
            Err(e) => panic!("can't flush {}", e),
        };
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

        let index_ptr = self.graph.index.as_ptr() as *const u64;
        let graph_ptr = Arc::new(SharedPtr(self.graph.graph.as_ptr()));
        let edge_size = self.graph.edge_size;

        // atomic array for next unused edge index for each node and unread edge count
        let mut next_edge_vec: Vec<AtomicU64> = Vec::with_capacity(node_count);
        next_edge_vec.resize_with(node_count, || AtomicU64::new(0));
        let mut edge_count: Vec<AtomicU8> = Vec::with_capacity(node_count);
        edge_count.resize_with(node_count, || AtomicU8::new(0));

        unsafe {
            for i in 0..node_count {
                next_edge_vec[i].store(*index_ptr.add(i), Ordering::Relaxed);
                edge_count[i].store(
                    (*index_ptr.add(i + 1) - *index_ptr.add(i)) as u8,
                    Ordering::Relaxed,
                );
            }
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
                            // Get the next unused edge from v
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
                                // let mut end_offsets = end_offsets.lock().unwrap();
                                // if end_offsets[v as usize] != 0 {
                                //     end_offsets[v as usize] -= 1;
                                let edge_idx =
                                    next_edge[v as usize].fetch_add(1, Ordering::Relaxed);
                                let neighbor: u64;
                                unsafe {
                                    neighbor = std::ptr::read_unaligned(
                                        graph_ptr.0.add((edge_idx * (edge_size as u64)) as usize)
                                            as *const T,
                                    )
                                    .dest();
                                }
                                stack.push(neighbor);
                            } else {
                                // No more unused edges from v -> backtrack
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
