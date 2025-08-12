use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use crossbeam::thread;
use num_cpus::get_physical;
use std::{
    collections::HashMap,
    io::Error,
    sync::{
        Arc, Barrier,
        atomic::{AtomicU8, Ordering},
    },
};

type ProceduralMemoryGVELouvain = (
    AbstractedProceduralMemoryMut<AtomicU8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<(usize, usize)>,
);

#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoGVELouvain<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    /// memmapped slice containing the coreness of each edge
    community: AbstractedProceduralMemoryMut<u16>,
}
#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>
    AlgoGVELouvain<'a, EdgeType, Edge>
{
    /// constants set according to the optimizeds parameters described in "GVE-Louvain: Fast Louvain Algorithm for
    /// Community Detection in Shared Memory Setting".
    /// Described in 4.1.2 Limiting the number of iterations per pass
    const MAX_ITERATIONS: usize = 20;
    /// Described in 4.1.3 Adjusting tolerance drop rate (threshold scaling)
    const TOLERANCE_DROP: f64 = 10.;
    /// Described in 4.1.4 Adjusting initial tolerance
    const INITIAL_TOLERANCE: f64 = 0.01;
    /// Described in 4.1.4 Adjusting aggregation tolerance
    const AGGREGATION_TOLERANCE: f64 = 0.8;

    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let output_filename = cache_file_name(graph.cache_fst_filename(), FileType::KTruss, None)?;
        let community =
            SharedSliceMut::<u16>::abst_mem_mut(output_filename.clone(), graph.width(), true)?;
        let gve_louvain = Self { graph, community };
        gve_louvain.compute(10)?;
        Ok(gve_louvain)
    }

    fn scan_communities(
        map: &mut HashMap<u16, usize>,
        edges: SharedSliceMut<Edge>,
        index: SharedSliceMut<usize>,
        communities: SharedSliceMut<u16>,
        i: usize,
        self_allowed: bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        for e_idx in *index.get(i)..*index.get(i + 1) {
            let j = edges.get(e_idx).dest();
            if !self_allowed && j == i {
                continue;
            }
            // 𝐻𝑡[𝐶'[𝑗]] ← 𝐻𝑡[𝐶'[𝑗]] + �
            match map.get_mut(communities.get(j)) {
                Some(h_t_j) => *h_t_j += 1,
                None => {
                    map.insert(*communities.get(j), 1);
                }
            };
        }
        Ok(())
    }

    fn compute(&self, mmap: u8) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }
}
