pub(crate) mod cache;
pub mod edge;
mod export_induced_subgraph;
mod impl_miscelanious;
mod impl_partition;
mod impl_reciprocal_edges;
pub(crate) mod impl_traits;
pub mod label;

#[cfg(feature = "petgraph")]
mod export_petgraph;
#[cfg(feature = "rustworkx")]
mod export_rustworkx_core;
#[cfg(any(test, feature = "bench"))]
pub mod test_utils;

use crate::graph::cache::GraphCache;
use crate::shared_slice::AbstractedProceduralMemory;
pub use graph_derive::{GenericEdge, GenericEdgeType};

use label::VoidLabel;
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::{fmt::Debug, fs::File, sync::Arc};

pub use graph_derive::E;
pub use graph_derive::N;
use graph_derive::sparkx_label;

/// The default integer type for indexing.
///
/// The default type is `usize` to facilitate usage. As graphs that fit in `u32` or smaller unsizeds
/// won't have performance issues give nthe nature of our algorithms.
///
/// Used for node indices in memmaped `CSR` representation.
pub type DefaultIx = usize;

/// Trait for the unsigned integer type used for indexing garph elements.
///
/// # Safety
///
/// Marked `unsafe` because: the trait must faithfully preserve
/// and convert index values.
pub unsafe trait IndexType:
    Copy + Default + Hash + Ord + Debug + Send + Sync + 'static
{
    fn new(x: usize) -> Self;
    fn index(&self) -> usize;
    fn max() -> Self;
}

unsafe impl IndexType for usize {
    #[inline(always)]
    fn new(x: usize) -> Self {
        x
    }
    #[inline(always)]
    fn index(&self) -> Self {
        *self
    }
    #[inline(always)]
    fn max() -> Self {
        usize::MAX
    }
}

unsafe impl IndexType for u32 {
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as u32
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        u32::MAX
    }
}

unsafe impl IndexType for u16 {
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as u16
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        u16::MAX
    }
}

unsafe impl IndexType for u8 {
    #[inline(always)]
    fn new(x: usize) -> Self {
        x as u8
    }
    #[inline(always)]
    fn index(&self) -> usize {
        *self as usize
    }
    #[inline(always)]
    fn max() -> Self {
        u8::MAX
    }
}

/// Node identifier.
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct NodeIndex<Ix = DefaultIx>(Ix);

impl<Ix: IndexType> NodeIndex<Ix> {
    #[inline]
    pub fn new(x: usize) -> Self {
        NodeIndex(IndexType::new(x))
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0.index()
    }

    #[inline]
    pub fn end() -> Self {
        NodeIndex(IndexType::max())
    }

    fn _into_edge(self) -> EdgeIndex<Ix> {
        EdgeIndex(self.0)
    }
}

unsafe impl<Ix: IndexType> IndexType for NodeIndex<Ix> {
    fn index(&self) -> usize {
        self.0.index()
    }
    fn new(x: usize) -> Self {
        NodeIndex::new(x)
    }
    fn max() -> Self {
        NodeIndex(<Ix as IndexType>::max())
    }
}
/// Edge identifier.
#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct EdgeIndex<Ix = DefaultIx>(Ix);

impl<Ix: IndexType> EdgeIndex<Ix> {
    #[inline]
    pub fn new(x: usize) -> Self {
        EdgeIndex(IndexType::new(x))
    }

    #[inline]
    pub fn index(self) -> usize {
        self.0.index()
    }

    /// An invalid `EdgeIndex` used to denote absence of an edge, for example
    /// to end an adjacency list.
    #[inline]
    pub fn end() -> Self {
        EdgeIndex(IndexType::max())
    }

    fn _into_node(self) -> NodeIndex<Ix> {
        NodeIndex(self.0)
    }
}

impl<Ix: IndexType> From<Ix> for EdgeIndex<Ix> {
    fn from(ix: Ix) -> Self {
        EdgeIndex(ix)
    }
}

/// Short version of `NodeIndex::new`
pub fn node_index<Ix: IndexType>(index: usize) -> NodeIndex<Ix> {
    NodeIndex::new(index)
}

/// Short version of `EdgeIndex::new`
pub fn edge_index<Ix: IndexType>(index: usize) -> EdgeIndex<Ix> {
    EdgeIndex::new(index)
}

/// Describes the behavior node labels must exhibit to be used by the tool.
///
/// Given a repr C/transparent struct the required traits may be derived automatically by
/// annotating the struct with #[sparkx_label].
#[allow(dead_code)]
pub trait N:
    Copy + std::fmt::Debug + Ord + bytemuck::Pod + bytemuck::Zeroable + Send + Sync
{
    fn new(v: usize) -> Self;
    #[inline]
    fn is_labeled() -> bool {
        false
    }
    #[inline]
    fn is_weighted() -> bool {
        false
    }
    #[inline]
    fn is_colored() -> bool {
        false
    }
    #[inline]
    fn label(&self) -> usize {
        0
    }
    #[inline]
    fn color(&self) -> usize {
        0
    }
    #[inline]
    fn weigth(&self) -> usize {
        1
    }
}

/// Describes the behavior edge labels must exhibit to be used by the tool.
///
/// Given a repr C/transparent struct the required traits may be derived automatically by
/// annotating the struct with #[sparkx_label].
#[allow(dead_code)]
pub trait E:
    Copy + std::fmt::Debug + Ord + bytemuck::Pod + bytemuck::Zeroable + Send + Sync
{
    fn new(v: usize) -> Self;
    #[inline]
    fn is_labeled() -> bool {
        false
    }
    #[inline]
    fn is_weighted() -> bool {
        false
    }
    #[inline]
    fn is_colored() -> bool {
        false
    }
    #[inline]
    fn label(&self) -> usize {
        0
    }
    #[inline]
    fn color(&self) -> usize {
        0
    }
    #[inline]
    fn weigth(&self) -> usize {
        1
    }
}

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
    ExactCloseness,
    ExactHarmonic,
    ExactLin,
}

enum GraphFile {
    Neighbors,
    Offsets,
    NodeLabels,
    EdgeLabels,
    MetaLabels,
}

#[derive(Clone)]
pub struct GraphMemoryMap<
    N: crate::graph::N = VoidLabel,
    E: crate::graph::E = VoidLabel,
    Ix: crate::graph::IndexType = DefaultIx,
> {
    graph: Arc<memmap2::Mmap>,
    index: Arc<memmap2::Mmap>,
    metalabels: Arc<fst::Map<memmap2::Mmap>>,
    graph_cache: GraphCache<N, E, Ix>,
    edge_size: usize,
    thread_count: u8,
    exports: u8,
    _marker: PhantomData<Ix>,
}

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
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
        graph_cache: GraphCache<N, E, Ix>,
        thread_count: Option<u8>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if graph_cache.readonly {
            let graph = Arc::new(unsafe { memmap2::Mmap::map(&graph_cache.neighbors_file)? });
            let index = Arc::new(unsafe { memmap2::Mmap::map(&graph_cache.offsets_file)? });
            let metalabels = Arc::new(fst::Map::new(unsafe {
                memmap2::MmapOptions::new().map(&File::open(&graph_cache.metalabel_filename)?)?
            })?);
            let edge_size = std::mem::size_of::<usize>();
            let thread_count = thread_count.unwrap_or(1).max(1);
            let exports = 0u8;

            return Ok(Self {
                graph,
                index,
                metalabels,
                graph_cache,
                edge_size,
                thread_count,
                exports,
                _marker: PhantomData::<Ix>,
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
        let graph_cache = GraphCache::<N, E, Ix>::from_file(p, id, batch, in_fst)?;
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
        let graph_cache = GraphCache::<N, E, Ix>::open(filename, batch)?;
        Self::init_from_cache(graph_cache, thread_count)
    }

    #[inline(always)]
    pub(crate) fn index_ptr(&self) -> *const usize {
        self.index.as_ptr() as *const usize
    }

    #[inline(always)]
    pub(crate) fn neighbours_ptr(&self) -> *const usize {
        self.graph.as_ptr() as *const usize
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

    /// Returns the graph's neighbors' file's filename.
    #[inline(always)]
    pub fn cache_edges_filename(&self) -> String {
        self.graph_cache.neighbors_filename()
    }

    /// Returns the graph's offsets' file's filename.
    #[inline(always)]
    pub fn cache_index_filename(&self) -> String {
        self.graph_cache.offsets_filename()
    }

    /// Returns the graph's node labels' file's filename.
    #[inline(always)]
    pub fn cache_node_labels_filename(&self) -> String {
        self.graph_cache.nodelabels_filename()
    }

    /// Returns the graph's edge labels' file's filename.
    #[inline(always)]
    pub fn cache_edge_labels_filename(&self) -> String {
        self.graph_cache.edgelabels_filename()
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
    ) -> Result<NeighbourIter<N, E, Ix>, Box<dyn std::error::Error>> {
        if node_id >= self.size() {
            return Err(
                format!("error {node_id} must be smaller than |V| = {}", self.size()).into(),
            );
        }

        Ok(NeighbourIter::<N, E, Ix>::new(
            self.graph.as_ptr() as *const usize,
            self.index.as_ptr() as *const usize,
            node_id,
        ))
    }

    // /// Returns an [`EdgeIter`] iterator over all of the graph's edges.
    // ///
    // /// [`EdgeIter`]: ./struct.EdgeIter.html#
    // pub fn edges(&self) -> Result<EdgeIter<EdgeType, Edge>, Box<dyn std::error::Error>> {
    //     Ok(EdgeIter::<EdgeType, Edge>::new(
    //         self.graph.as_ptr() as *const Edge,
    //         self.index.as_ptr() as *const usize,
    //         0,
    //         self.size(),
    //     ))
    // }

    // /// Returns an [`EdgeIter`] iterator over the graph's edges in a given range.
    // ///
    // /// # Arguments
    // ///
    // /// * `begin_node` --- id of the node whose offset begin marks the beginning of the iterator's range.
    // /// * `end_node` ---  id of the node whose offset end marks the end of the iterator's range.
    // ///
    // /// [`EdgeIter`]: ./struct.EdgeIter.html#
    // pub fn edges_in_range(
    //     &self,
    //     begin_node: usize,
    //     end_node: usize,
    // ) -> Result<EdgeIter<EdgeType, Edge>, Box<dyn std::error::Error>> {
    //     if begin_node > end_node {
    //         return Err("error invalid range, beginning after end".into());
    //     }
    //     if begin_node > self.size() || end_node > self.size() {
    //         return Err("error invalid range".into());
    //     }
    //
    //     Ok(EdgeIter::<EdgeType, Edge>::new(
    //         self.graph.as_ptr() as *const Edge,
    //         self.index.as_ptr() as *const usize,
    //         begin_node,
    //         end_node,
    //     ))
    // }

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
    ) -> Result<Self, Box<dyn std::error::Error>> {
        self.apply_mask_to_nodes(mask, identifier)
    }

    #[inline(always)]
    pub fn check_neighbour(&self, u: usize, v: usize) -> Option<usize> {
        self.is_neighbour_impl(u, v)
    }

    #[inline(always)]
    pub fn check_triangle(&self, u: usize, v: usize, w: usize) -> Option<(usize, usize)> {
        self.is_triangle_impl(u, v, w)
    }

    #[inline(always)]
    pub(crate) fn edge_reciprocal(
        &self,
    ) -> Result<AbstractedProceduralMemory<usize>, Box<dyn std::error::Error>> {
        self.get_edge_reciprocal_impl()
    }

    #[inline(always)]
    pub(crate) fn edge_over(
        &self,
    ) -> Result<AbstractedProceduralMemory<usize>, Box<dyn std::error::Error>> {
        self.get_edge_over_impl()
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

    // /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format keeping all edge and node labelings[^1].
    // ///
    // /// [^1]: if none of the edge or node labeling is wanted consider using [`export_petgraph_stripped`].
    // ///
    // /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    // /// [`export_petgraph_stripped`]: ./struct.GraphMemoryMap.html#method.export_petgraph_stripped
    // #[cfg(feature = "petgraph")]
    // pub fn export_petgraph(
    //     &self,
    // ) -> Result<
    //     petgraph::graph::DiGraph<petgraph::graph::NodeIndex<usize>, EdgeType>,
    //     Box<dyn std::error::Error>,
    // > {
    //     self.export_petgraph_impl()
    // }

    /// Export the [`GraphMemoryMap`] instance to petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format stripping any edge or node labelings whatsoever.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[cfg(feature = "petgraph")]
    pub fn export_petgraph_stripped(
        &self,
    ) -> Result<petgraph::graph::DiGraph<(), ()>, Box<dyn std::error::Error>> {
        self.export_petgraph_stripped_impl()
    }

    // /// Export the [`GraphMemoryMap`] instance to rustworkx_core compatible petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format keeping all edge and node labelings[^1].
    // ///
    // /// [^1]: if none of the edge or node labeling is wanted consider using [`export_petgraph_stripped`].
    // ///
    // /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    // /// [`export_petgraph_stripped`]: ./struct.GraphMemoryMap.html#method.export_petgraph_stripped
    // #[cfg(feature = "rustworkx")]
    // pub fn export_rustworkx(
    //     &self,
    // ) -> Result<
    //     rustworkx_core::petgraph::graph::DiGraph<
    //         rustworkx_core::petgraph::graph::NodeIndex<usize>,
    //         EdgeType,
    //     >,
    //     Box<dyn std::error::Error>,
    // > {
    //     self.export_rustworkx_impl()
    // }

    /// Export the [`GraphMemoryMap`] instance to rustworkx_core compatible petgraph's [`DiGraph`](https://docs.rs/petgraph/latest/petgraph/graph/type.DiGraph.html) format stripping any edge or node labelings whatsoever.
    ///
    /// [`GraphMemoryMap`]: ./struct.GraphMemoryMap.html#
    #[cfg(feature = "rustworkx")]
    pub fn export_rustworkx_stripped(
        &self,
    ) -> Result<rustworkx_core::petgraph::graph::DiGraph<(), ()>, Box<dyn std::error::Error>> {
        self.export_rustworkx_stripped_impl()
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
pub struct NeighbourIter<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> {
    edge_ptr: *const usize,
    _orig_edge_ptr: *const usize,
    _orig_id_ptr: *const usize,
    id: usize,
    count: usize,
    offset: usize,
    _phantom1: std::marker::PhantomData<N>,
    _phantom2: std::marker::PhantomData<E>,
    _phantom3: std::marker::PhantomData<Ix>,
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> NeighbourIter<N, E, Ix> {
    fn new(edge_mmap: *const usize, id_mmap: *const usize, node_id: usize) -> Self {
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
            _phantom1: std::marker::PhantomData::<N>,
            _phantom2: std::marker::PhantomData::<E>,
            _phantom3: std::marker::PhantomData::<Ix>,
        }
    }

    #[inline(always)]
    fn _into_neighbour(&self) -> Self {
        NeighbourIter::new(self._orig_edge_ptr, self._orig_id_ptr, unsafe {
            self.edge_ptr.read_unaligned()
        })
    }

    fn _next_back_with_offset(&mut self) -> Option<(usize, usize)> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let next: (usize, usize);
        unsafe {
            next = (self.id, self.edge_ptr.add(self.count).read_unaligned());
        };
        Some(next)
    }

    pub fn remaining_neighbours(&self) -> usize {
        self.count
    }

    fn _next_with_offset(&mut self) -> Option<(usize, usize)> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let next: (usize, usize);
        self.edge_ptr = unsafe {
            next = (self.offset, self.edge_ptr.read_unaligned());
            self.edge_ptr.add(1)
        };
        self.offset += 1;
        Some(next)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> DoubleEndedIterator
    for NeighbourIter<N, E, Ix>
{
    #[inline(always)]
    fn next_back(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let next: usize;
        unsafe {
            next = self.edge_ptr.add(self.count).read_unaligned();
        };
        Some(next)
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Iterator
    for NeighbourIter<N, E, Ix>
{
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<usize> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        self.offset += 1;
        let next: usize;
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    std::ops::Index<std::ops::RangeFull> for GraphMemoryMap<N, E, Ix>
{
    type Output = [usize];
    #[inline]
    fn index(&self, _index: std::ops::RangeFull) -> &[usize] {
        // FIXME: this is really weird, most probably it is WRONG!!! Don't turn this in without replacing this ugly '* 8' for something that you understand and guarantee is right!!!
        unsafe {
            std::slice::from_raw_parts(
                self.graph.as_ptr() as *const usize,
                self.size() * 8 / self.edge_size,
            )
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    std::ops::Index<std::ops::Range<usize>> for GraphMemoryMap<N, E, Ix>
{
    type Output = [usize];
    #[inline]
    fn index(&self, index: std::ops::Range<usize>) -> &[usize] {
        unsafe {
            std::slice::from_raw_parts(
                self.graph.as_ptr().add(index.start * self.edge_size) as *const usize,
                index.end - index.start,
            )
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType>
    std::ops::Index<std::ops::Range<u64>> for GraphMemoryMap<N, E, Ix>
{
    type Output = [usize];
    #[inline]
    fn index(&self, index: std::ops::Range<u64>) -> &[usize] {
        let start = index.start as usize;
        let end = index.end as usize;

        unsafe {
            std::slice::from_raw_parts(
                self.graph.as_ptr().add(start * self.edge_size) as *const usize,
                end - start,
            )
        }
    }
}

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> Debug
    for GraphMemoryMap<N, E, Ix>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "MemoryMappedData {{\n\t
            neighbors_filename: {},\n\t
            offsets_filename: {},\n\t
            size: {},\n\t
            width: {},\n\t
            }}",
            self.graph_cache.neighbors_filename(),
            self.graph_cache.offsets_filename(),
            self.size(),
            self.width(),
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
