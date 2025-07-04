use crate::node::EdgeType;

use bitfield::bitfield;
use bytemuck::{Pod, Zeroable};
use core::panic;
use fst::{Map, MapBuilder};
use memmap::{Mmap, MmapOptions};
use static_assertions::const_assert;
use std::any::type_name;
use std::hash::{Hash, Hasher};
use std::io::{BufRead, BufReader};
use std::{
    fmt::{Debug, Display},
    fs::{self, File, OpenOptions},
    io::{Error, Write},
    marker::PhantomData,
    path::Path,
    slice,
};
use zerocopy::*;

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

pub struct GraphCache<T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf> {
    pub graph_file: File,
    pub index_file: File,
    pub kmer_file: File,
    pub graph_filename: String,
    pub index_filename: String,
    pub kmer_filename: String,
    pub graph_bytes: u64,
    pub index_bytes: u64,
    pub readonly: bool,
    _marker: PhantomData<T>,
}

fn _type_of<T>(_: T) -> &'static str {
    type_name::<T>()
}

use std::process::Command;

fn external_sort_by_content(temp: &str, sorted: &str) -> std::io::Result<()> {
    // sort based on 2nd column (content), not line number
    Command::new("sort")
        .args(["-k2", temp, "-o", sorted])
        .status()?;
    Ok(())
}

impl<T> GraphCache<T>
where
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
{
    pub fn init() -> Result<GraphCache<T>, Error> {
        if !Path::new(CACHE_DIR).exists() {
            fs::create_dir_all(CACHE_DIR).expect("error creating cache dir");
        }

        let rand_str = rand::random::<u64>().to_string();
        let graph_filename = CACHE_DIR.to_string() + rand_str.as_str() + ".mmap";
        let index_filename = CACHE_DIR.to_string() + "index_" + rand_str.as_str() + ".mmap";
        let kmer_filename = CACHE_DIR.to_string() + rand_str.as_str() + ".tmp";

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
            graph_file,
            index_file,
            kmer_file,
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
        let graph_filename = CACHE_DIR.to_string() + filename.as_str() + ".mmap";
        let index_filename = CACHE_DIR.to_string() + "index_" + filename.as_str() + ".mmap";
        let kmer_filename = CACHE_DIR.to_string() + "fst_" + filename.as_str() + ".fst";

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
            Err(e) => panic!("error couldnt open file {}: {}", index_filename, e),
        };

        let graph_len = graph_file.metadata().unwrap().len();
        let index_len = index_file.metadata().unwrap().len();

        Ok(GraphCache::<T> {
            graph_file,
            index_file,
            kmer_file,
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
                writeln!(self.kmer_file, "{}\t{}", node_id, label)
                    .expect(format!("error writing kmer for {}", node_id).as_str());
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

    pub fn make_readonly(&mut self) -> Result<(), Error> {
        if self.readonly {
            return Ok(());
        }

        let mut sorted_file = self.kmer_filename.clone();
        sorted_file = sorted_file.replace("./cache/", "./cache/sorted_");
        let mut fst_file = self.kmer_filename.clone();
        fst_file = fst_file.replace("./cache/", "./cache/fst_");
        fst_file = fst_file.replace(".tmp", ".fst");

        external_sort_by_content(self.kmer_filename.as_str(), sorted_file.as_str())
            .expect("error couldn't sort kmers");

        let kmer_file =
            File::create(fst_file.clone()).expect("error couldn't create mer .fst file");
        self.kmer_file = kmer_file.try_clone().expect("error couldn't clone fst fd");
        let mut build = MapBuilder::new(kmer_file).expect("error couldn't initialize builder");

        let reader = BufReader::new(File::open(sorted_file)?);
        for line in reader.lines() {
            let line = line?;
            let mut parts = line.splitn(2, '\t');
            if let (Some(id_value), Some(kmer)) = (parts.next(), parts.next()) {
                build
                    .insert(
                        kmer,
                        id_value
                            .parse::<u64>()
                            .expect("error failed to cast id as u64"),
                    )
                    .expect(
                        format!("error couldn't insert kmer for node (id {})", id_value).as_str(),
                    );
            }
        }
        build.finish().expect("error couldn't finish fst build"); // finalize and write the FST to disk
        self.kmer_filename = fst_file;

        self.index_file
            .write_all(self.graph_bytes.as_bytes())
            .expect("error couldn't finish index");
        self.index_bytes += BYTES_U64;

        for file in [&self.index_file, &self.graph_file, &self.kmer_file] {
            let metadata = file.metadata()?;
            let mut permissions = metadata.permissions();
            permissions.set_readonly(true);
            file.set_permissions(permissions)?;
        }
        self.readonly = true;

        Ok(())
    }
}

pub struct GraphMemoryMap<
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
> {
    graph: Mmap,
    index: Mmap,
    kmers: Map<Mmap>,
    graph_cache: GraphCache<T>,
    edge_size: usize,
    _marker: PhantomData<U>,
}

impl<T, U> GraphMemoryMap<T, U>
where
    T: Copy + Debug + Display + Pod + Zeroable + EdgeOutOf,
    U: Copy + Debug + Display + Pod + Zeroable + GraphEdge,
{
    pub fn init(cache: GraphCache<T>) -> Result<GraphMemoryMap<T, U>, Error> {
        match cache.readonly {
            true => {
                let mmap = unsafe {
                    MmapOptions::new()
                        .map(&File::open(&cache.kmer_filename).expect("Failed to open file"))
                        .expect("error couldn't mmap k-mer fst")
                };
                Ok(GraphMemoryMap {
                    graph: unsafe {
                        Mmap::map(&cache.graph_file).expect("error couldn't create graph mmap")
                    },
                    index: unsafe {
                        Mmap::map(&cache.index_file).expect("error couldn't create index mmap")
                    },
                    kmers: Map::new(mmap).expect("error couldn't map k-mer fst mmap"),
                    graph_cache: cache,
                    edge_size: std::mem::size_of::<T>(),
                    _marker: PhantomData,
                })
            }
            false => panic!("error file must be readonly to be memmapped"),
        }
    }

    #[inline(always)]
    pub fn node_degree(&self, node_id: u64) -> u64 {
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
    pub fn index_node(&self, node_id: u64) -> std::ops::Range<u64> {
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

    pub fn edges_in_range(&self, start_node: u64, end_node: u64) -> Result<EdgeIter<T, U>, Error> {
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
    fn into_neighbour(&self) -> Self {
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
