pub(crate) mod cache;
mod graph_miscelanious;
pub(crate) mod impl_t;
mod induced_subgraph;
mod partition;
mod reciprocal_edges;

pub mod edge;

#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

use crate::graph::cache::GraphCache;
use crate::shared_slice::AbstractedProceduralMemory;
use cache::utils::{H, cache_file_name};
pub use graph_derive::{GenericEdge, GenericEdgeType};
use std::mem::ManuallyDrop;
use std::path::Path;
use std::{fmt::Debug, fs::File, sync::Arc};

/// Describes the behavior edge types must exhibit to be used by the tool.
///
/// Given an struct this trait may be automatically implemented using the provided procedural macro `GenericEdgeType`.
#[allow(dead_code)]
pub trait GenericEdgeType:
    Copy
    + Clone
    + Default
    + Debug
    + std::fmt::Display
    + PartialEq
    + Eq
    + bytemuck::Zeroable
    + From<u64>
    + From<usize>
    + rustworkx_core::petgraph::EdgeType
    + Send
    + Sync
{
    /// Edge label getter.
    fn label(&self) -> usize;
    /// Edge label setter.
    fn set_label(&mut self, label: u64);
}

/// Describes the behavior edges must exhibit to be used by the tool.
///
/// Given an implicitly packed struct this trait may be automatically implemented using the provided procedural macro `GenericEdge`.
#[allow(dead_code)]
pub trait GenericEdge<T: GenericEdgeType>:
    Copy
    + Clone
    + Default
    + Debug
    + std::fmt::Display
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + bytemuck::Pod
    + bytemuck::Zeroable
    + Send
    + Sync
{
    /// Constructor from an <<edge_dest: u64>> and an <<edge_type: u64>>
    fn new(edge_dest: u64, edge_type: u64) -> Self;
    /// Edge destiny node setter from a <<new_edge_dest: u64>>.
    fn set_edge_dest(&mut self, new_edge_dest: u64) -> &mut Self;
    /// Edge type setter from a <<new_edge_type: u64>>.
    fn set_edge_type(&mut self, new_edge_type: u64) -> &mut Self;
    /// Edge destiny node getter.
    fn dest(&self) -> usize;
    /// Edge type getter.
    fn e_type(&self) -> T;
}

#[derive(Clone, Debug, PartialEq)]
#[allow(dead_code)]
pub enum CacheFile {
    /// FIXME: Only member that should be visible to users
    General,
    BFS,
    DFS,
    EulerTrail,
    KCoreBZ,
    KCoreLEA,
    KTrussBEA,
    KTrussPKT,
    ClusteringCoefficient,
    EdgeReciprocal,
    EdgeOver,
    HyperBall,
    HyperBallDistances,
    HyperBallInvDistances,
    HyperBallClosenessCentrality,
    HyperBallHarmonicCentrality,
    HyperBallLinCentrality,
    GVELouvain,
}

enum GraphFile {
    Edges,
    Index,
    Metalabel,
}

#[derive(Clone)]
pub struct GraphMemoryMap<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: Arc<memmap2::Mmap>,
    index: Arc<memmap2::Mmap>,
    metalabels: Arc<fst::Map<memmap2::Mmap>>,
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
    /// * `graph_cache` --- [`GraphCache`] instance to be used.
    /// * `thread_count`--- user suggested number of threads to be used when computing algorithms on the graph.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn init_from_cache(
        graph_cache: GraphCache<EdgeType, Edge>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if graph_cache.readonly {
            let graph = Arc::new(unsafe { memmap2::Mmap::map(&graph_cache.graph_file)? });
            let index = Arc::new(unsafe { memmap2::Mmap::map(&graph_cache.index_file)? });
            let metalabels = Arc::new(fst::Map::new(unsafe {
                memmap2::MmapOptions::new().map(&File::open(&graph_cache.metalabel_filename)?)?
            })?);
            let edge_size = std::mem::size_of::<Edge>();
            let thread_count = thread_count.unwrap_or(1).max(1);
            let exports = 0u8;

            return Ok(GraphMemoryMap {
                graph,
                index,
                metalabels,
                graph_cache,
                edge_size,
                thread_count,
                exports,
            });
        }

        Err("error graph cache must be readonly to be memmapped".into())
    }

    #[inline(always)]
    pub fn from_file<P: AsRef<Path>>(
        p: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_cache = GraphCache::<EdgeType, Edge>::from_file(p, id, batch, in_fst)?;
        // println!(
        //     "{:?}",
        //     cache_file_name(
        //         &graph_cache.graph_filename,
        //         cache::utils::FileType::ExactClosenessCentrality(H::H),
        //         None
        //     )?
        // );
        Self::init_from_cache(graph_cache, thread_count)
    }

    #[inline(always)]
    pub fn open(
        filename: &str,
        batch: Option<usize>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let graph_cache = GraphCache::<EdgeType, Edge>::open(filename, batch)?;
        Self::init_from_cache(graph_cache, thread_count)
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
        assert!(node_id < self.size());
        unsafe {
            let ptr = (self.index.as_ptr() as *const usize).add(node_id);
            let begin = ptr.read_unaligned();
            ptr.add(1).read_unaligned() - begin
        }
    }

    /// Returns the given (by id) node's metalabel if it exists and was stored in the graph's fst.
    #[inline(always)]
    pub fn node_id_from_metalabel(
        &self,
        metalabel: &str,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        if let Some(val) = self.metalabels.get(metalabel) {
            Ok(val)
        } else {
            Err(format!("error metalabel {metalabel} not found").into())
        }
    }

    /// Returns the given (by id) node's edges' offsets.
    #[inline(always)]
    pub fn index_node(&self, node_id: usize) -> std::ops::Range<usize> {
        assert!(node_id < self.size());
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
        if node_id >= self.size() {
            return Err(
                format!("error {node_id} must be smaller than |V| = {}", self.size()).into(),
            );
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
    /// * `diffusion` --- the diffusion vector[^1][^2].
    /// * `target_size` --- the partition's target size[^3].
    /// * `target_volume` --- the partition's target volume[^4].
    ///
    /// [^1]: diffusion vector entries must be of type (node_id: [`usize`], heat: [`f64`]).
    /// [^2]: entries must be descendingly ordered by diffusion.
    /// [^3]: if [`None`] is provided defaults to `|V|`, effectively, the overall best partition by conducatance is returned independent on the number of nodes in it.
    /// [^4]: if [`None`] is provided defaults to `|E|`, effectively, the overall best partition by conducatance is returned independent on the number of edges in it.
    pub fn sweep_cut(
        &self,
        diffusion: &mut [(usize, f64)],
        target_size: Option<usize>,
        target_volume: Option<usize>,
    ) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        self.sweep_cut_over_diffusion_vector_by_conductance(diffusion, target_size, target_volume)
    }

    /// Computes the modularity over a given partition over the [`GraphMemoryMap`] instance's nodes.
    ///
    /// # Arguments
    ///
    /// * `communities` --- a slice containing each node's community, the [`GraphMemoryMap`] instance's partition.
    /// * `comms_cardinality` --- cardinality of distinct communities in the given partition.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn modularity(
        &self,
        communities: &[usize],
        comms_cardinality: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        self.modularity_impl(communities, comms_cardinality)
    }

    /// Returns number of entries in the offset file[^1].
    ///
    /// [^1]: this is equivalent to `|V| + 1`, as there is an extra offset file entry to mark the end of edges' offsets.
    #[inline(always)]
    pub fn offsets_size(&self) -> usize {
        self.graph_cache.index_bytes // num nodes
    }

    /// Returns number of nodes in the offset file[^1].
    ///
    /// Performs a saturating subtraction of 1 to the number of entries in the offset file.
    ///
    /// [^1]: this is equivalent to `|V|`.
    #[inline(always)]
    pub fn size(&self) -> usize {
        self.graph_cache.index_bytes.saturating_sub(1)
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
    /// The resulting subgraph is that wherein only the set of nodes, `S ⊂ V`, of the nodes for whose the output of mask is true, as well as, only the set of edges coming from and going to nodes in `S`[^1].
    ///
    /// [^1]: the node ids of the subgraph may not, and probably will not, correspond to the original node identifiers, efffectively, it will be a whole new graph.
    pub fn induced_subgraph(
        &mut self,
        mask: fn(usize) -> bool,
        identifier: Option<&str>,
    ) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
        self.apply_mask_to_nodes(mask, identifier)
    }

    /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format keeping all edge and node labelings[^1].
    ///
    /// [^1]: if none of the edge or node labeling is wanted consider using [`export_petgraph_stripped`].
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    /// [`export_petgraph_stripped`]: ./struct.GraphMemoryMap.html#method.export_petgraph_stripped
    pub fn export_petgraph(
        &self,
    ) -> Result<
        rustworkx_core::petgraph::graph::DiGraph<
            rustworkx_core::petgraph::graph::NodeIndex<usize>,
            EdgeType,
        >,
        Box<dyn std::error::Error>,
    > {
        let mut graph = rustworkx_core::petgraph::graph::DiGraph::<
            rustworkx_core::petgraph::graph::NodeIndex<usize>,
            EdgeType,
        >::new();
        let node_count = self.size();

        (0..node_count).for_each(|u| {
            graph.add_node(rustworkx_core::petgraph::graph::NodeIndex::new(u));
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
                    graph.add_edge(
                        rustworkx_core::petgraph::graph::NodeIndex::new(u),
                        rustworkx_core::petgraph::graph::NodeIndex::new(v.dest()),
                        v.e_type(),
                    );
                });
            });

        Ok(graph)
    }

    /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format stripping any edge or node labelings whatsoever.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    pub fn export_petgraph_stripped(
        &self,
    ) -> Result<rustworkx_core::petgraph::graph::DiGraph<(), ()>, Box<dyn std::error::Error>> {
        let mut graph = rustworkx_core::petgraph::graph::DiGraph::<(), ()>::new();
        let node_count = self.size();

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
                    graph.add_edge(
                        rustworkx_core::petgraph::graph::NodeIndex::new(u),
                        rustworkx_core::petgraph::graph::NodeIndex::new(v.dest()),
                        (),
                    );
                });
            });

        Ok(graph)
    }

    #[inline(always)]
    pub fn check_neighbour(&self, u: usize, v: usize) -> Option<usize> {
        self.is_neighbour(u, v)
    }

    #[inline(always)]
    pub fn check_triangle(&self, u: usize, v: usize, w: usize) -> Option<(usize, usize)> {
        self.is_triangle(u, v, w)
    }

    #[inline(always)]
    pub(crate) fn edge_reciprocal(
        &self,
    ) -> Result<AbstractedProceduralMemory<usize>, Box<dyn std::error::Error>> {
        self.get_edge_reciprocal()
    }

    #[inline(always)]
    pub(crate) fn edge_over(
        &self,
    ) -> Result<AbstractedProceduralMemory<usize>, Box<dyn std::error::Error>> {
        self.get_edge_dest_id_over_source()
    }

    #[inline(always)]
    fn build_helper_filename(&self, seq: usize) -> Result<String, Box<dyn std::error::Error>> {
        self.graph_cache.build_helper_filename(seq)
    }

    #[inline(always)]
    fn cleanup_helpers(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.graph_cache.cleanup_helpers()
    }

    /// Build a cached (either `.mmap` or `.tmp`) file of a given [`CacheFile`] type for the [`GraphMemoryMap`] instance.
    ///
    /// [`CacheFile`]: ./enum.CacheFile.html#
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[inline(always)]
    pub fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.graph_cache.build_cache_filename(file_type, seq)
    }

    /// Remove [`GraphMemoryMap`] instance's cached `.tmp` files for a given [`CacheFile`] in the cache directory.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    /// [`CacheFile`]: ./enum.CacheFile.html#
    #[inline(always)]
    pub fn cleanup_cache(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>> {
        self.graph_cache.cleanup_cache(target)
    }

    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut this = ManuallyDrop::new(self);

        let er_fn = this.build_cache_filename(CacheFile::EdgeReciprocal, None)?;
        let eo_fn = this.build_cache_filename(CacheFile::EdgeOver, None)?;

        let _r = std::fs::remove_file(er_fn);
        let _r = std::fs::remove_file(eo_fn);

        this.graph_cache.drop_cache()
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
        // FIXME: this is really weird, most probably it is WRONG!!! Don't turn this in without replacing this ugly '* 8' for something that you understand and guarantee is right!!!
        unsafe {
            std::slice::from_raw_parts(
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
            std::slice::from_raw_parts(
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
            std::slice::from_raw_parts(
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
            "{{\n\tgraph filename: {}\n\tindex filename: {}\n\tmetalabel filename: {}\n}}",
            self.graph_filename, self.index_filename, self.metalabel_filename
        )
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
