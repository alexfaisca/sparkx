use crate::graph::*;
use crate::shared_slice::*;

use std::mem::ManuallyDrop;

type ProceduralMemoryBFSDists = (
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<bool>,
);

/// For the computation of BFS in a [`GraphMemoryMap`] instance.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
pub struct BFSDists<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    /// Graph in which the BFS is to be performed.
    g: &'a GraphMemoryMap<EdgeType, Edge>,
    source: usize,
    /// Memmapped slice containing each nodes' distance to the source node.
    distances: AbstractedProceduralMemoryMut<usize>,
    /// Array containing the starting position of each euler trail.
    reacheable: usize,
    /// Sum of all nodes' distance to the source node.
    total_distances: f64,
}

#[allow(dead_code)]
impl<'a, EdgeType, Edge> BFSDists<'a, EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    pub fn new(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut bfs = Self::new_no_compute(g, source)?;
        let proc_mem = bfs.init_cache_mem()?;

        bfs.compute_with_proc_mem(proc_mem)?;

        Ok(bfs)
    }

    /// Returns the sum of all nodes' distances to the source node as determined by the BFS.
    pub fn total_distances(&self) -> f64 {
        self.total_distances
    }

    /// Returns the number of reacheable nodes from the source node as determined by the BFS.
    pub fn recheable(&self) -> usize {
        self.reacheable
    }

    /// Returns all nodes' distance to the source node as determined by the BFS.
    pub fn get_distances(&'a self) -> &'a [usize] {
        self.distances.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::BFS, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, EdgeType, Edge> BFSDists<'a, EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, source)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, source)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryBFSDists, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<ProceduralMemoryBFSDists, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBFSDists,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryBFSDists,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.width()
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
        source: usize,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        if source >= g.size() {
            return Err(format!(
                "error node with id {source} doesn't exist: max id is |V| = {}",
                g.size()
            )
            .into());
        }
        let out_fn = g.build_cache_filename(CacheFile::BFS, None)?;
        Ok(Self {
            g,
            source,
            distances: SharedSliceMut::<usize>::abst_mem_mut(&out_fn, g.size(), true)?,
            reacheable: 0,
            total_distances: 0.,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<ProceduralMemoryBFSDists, Box<dyn std::error::Error>> {
        let node_count = self.g.size();

        let q_fn = self.build_cache_filename(CacheFile::BFS, Some(0))?;
        let v_fn = self.build_cache_filename(CacheFile::BFS, Some(1))?;

        let queue = SharedSliceMut::<usize>::abst_mem_mut(&q_fn, node_count, true)?;
        let visited = SharedSliceMut::<bool>::abst_mem_mut(&v_fn, node_count, true)?;

        Ok((queue, visited))
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        proc_mem: ProceduralMemoryBFSDists,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let (queue, mut visited) = proc_mem;
        let mut dist = self.distances.shared_slice();

        let mut next = SharedQueueMut::from_shared_slice(queue.shared_slice());
        let mut reacheable = 0;
        let mut sum: f64 = 0.;

        next.push(self.source);

        // find cycles until no unused edges remain
        while let Some(u) = next.pop() {
            let du = *dist.get(u);
            let dup1 = (du + 1) as f64;
            for v in self.g.neighbours(u)? {
                let v = v.dest();
                if !*visited.get(v) {
                    *visited.get_mut(v) = true;
                    *dist.get_mut(v) = du + 1;
                    sum += dup1;
                    reacheable += 1;
                    next.push(v);
                }
            }
        }

        self.reacheable = reacheable;
        self.total_distances = sum;
        // cleanup cache
        self.g.cleanup_cache(CacheFile::BFS)?;

        Ok(())
    }
}
