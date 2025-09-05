use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;
use crate::utils::OneOrMany;

use crossbeam::thread;
use portable_atomic::{AtomicF64, AtomicU8, AtomicUsize, Ordering};
use smallvec::SmallVec;
use std::{
    collections::{HashMap, HashSet},
    sync::{Arc, Barrier},
};

type ProceduralMemoryClusteringCoefficient = (AbstractedProceduralMemoryMut<AtomicU8>,);

/// For the computation of a [`GraphMemoryMap`] instance's local clustering coefficient, transitivity and average local clustering coefficient.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct ClusteringCoefficient<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which local clustering coefficient, transitivity and average local clustering coefficient.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's local clustering coefficient.
    local: AbstractedProceduralMemoryMut<f64>,
    /// Global clustering coefficient --- graph transitivity.
    transitivity: f64,
    /// Overall level of clustering in a network is measured by Watts and Strogatz as the average of the local clustering coefficients of all the nodes.
    local_average: f64,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ClusteringCoefficient<'a, N, E, Ix> {
    /// Computes a [`GraphMemoryMap`] instance's local clustering coefficient, transitivity and average local clustering
    /// coefficient.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance for which k-core decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::ClusteringCoefficient, None)?;
        let local = SharedSliceMut::<f64>::abst_mem_mut(&out_fn, g.size(), true)?;
        let mut clustering_coefficient = Self {
            g,
            local,
            transitivity: 0.,
            local_average: 0.,
        };
        clustering_coefficient.compute(10)?;
        Ok(clustering_coefficient)
    }

    /// Getter for the average of clustering coefficients of all nodes.
    pub fn get_average_clustering_coefficient(&self) -> f64 {
        self.local_average
    }

    /// Getter for individual node's clustering coefficient.
    pub fn get_node_clusteringcoefficient(&self, id: usize) -> f64 {
        assert!(id < self.g.size(), "id < |V| --- not met");
        *self.local.get(id)
    }

    /// Getter for graph's global clustering coefficient --- graph transitivity.
    pub fn get_graph_transitivity(&self) -> f64 {
        self.transitivity
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    fn init_procedural_memory_burkhardt_et_al(
        &self,
        mmap: u8,
    ) -> Result<ProceduralMemoryClusteringCoefficient, Box<dyn std::error::Error>> {
        let edge_count = self.g.width();

        let t_fn = self.build_cache_filename(CacheFile::ClusteringCoefficient, Some(0))?;

        let tri_count = SharedSliceMut::<AtomicU8>::abst_mem_mut(&t_fn, edge_count, mmap > 0)?;

        Ok((tri_count,))
    }

    /// Computes a [`GraphMemoryMap`] instance's local clustering coefficient, transitivity and average local clustering
    /// coefficient.
    ///
    /// # Arguments
    ///
    /// * `mmap` --- the level of memmapping to be used during the computation (*experimental feature*).
    ///
    pub fn compute(&mut self, mmap: u8) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let edge_count = self.g.width();

        let threads = self.g.thread_num();
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = SharedSlice::<usize>::new(self.g.offsets_ptr(), self.g.offsets_size());
        let neighbours_ptr = SharedSlice::<usize>::new(self.g.neighbours_ptr(), edge_count);

        // Shared atomic & simple arrays for counts and trussness
        let triangle_count = self.init_procedural_memory_burkhardt_et_al(mmap)?.0;

        let edge_reciprocal = self.g.edge_reciprocal()?;
        let edge_out = self.g.edge_over()?;

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        let total_possible_triangles = Arc::new(AtomicF64::new(0.));
        let local_clustering_sum = Arc::new(AtomicF64::new(0.));
        let total_triangles = Arc::new(AtomicUsize::new(0));

        thread::scope(|scope| {
            for tid in 0..threads {
                let eo = edge_out.shared_slice();
                let er = edge_reciprocal.shared_slice();

                let mut local = self.local.shared_slice();
                let mut tris = triangle_count.shared_slice();

                let total_possible_triangles = total_possible_triangles.clone();
                let total_triangles = total_triangles.clone();
                let local_clustering_sum = local_clustering_sum.clone();

                let synchronize = synchronize.clone();

                let begin = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);

                scope.spawn(move |_| {
                    // initialize triangle_count with zeroes
                    let edge_begin = *index_ptr.get(begin);
                    let edge_end = *index_ptr.get(end);
                    for idx in edge_begin..edge_end {
                        *tris.get_mut(idx) = AtomicU8::new(0);
                    }

                    synchronize.wait();

                    let mut all_neighbours = HashSet::<usize>::new();
                    let mut neighbours = HashMap::<usize, OneOrMany<usize>>::new();
                    for u in begin..end {
                        for j in *eo.get(u)..*index_ptr.get(u + 1) {
                            let w = *neighbours_ptr.get(j);
                            all_neighbours.insert(w);
                            match neighbours.entry(w) {
                                std::collections::hash_map::Entry::Vacant(e) => {
                                    e.insert(OneOrMany::One(j));
                                }
                                std::collections::hash_map::Entry::Occupied(mut e) => {
                                    match e.get_mut() {
                                        OneOrMany::One(i) => {
                                            let mut sv: SmallVec<[usize; 4]> = SmallVec::new();
                                            sv.push(*i);
                                            sv.push(j);
                                            *e.get_mut() = OneOrMany::Many(sv);
                                        }
                                        OneOrMany::Many(sv) => sv.push(j),
                                    }
                                }
                            }
                        }
                        for u_v in *index_ptr.get(u)..*eo.get(u) {
                            let v = *neighbours_ptr.get(u_v);
                            all_neighbours.insert(v);
                            if u == v {
                                continue;
                            }
                            for v_w in (*eo.get(v)..*index_ptr.get(v + 1)).rev() {
                                let w = *neighbours_ptr.get(v_w);
                                if w <= u {
                                    break;
                                }
                                match neighbours.get(&w) {
                                    Some(i) => match i {
                                        OneOrMany::One(u_w) => {
                                            let w_u = *er.get(*u_w);
                                            tris.get(u_v).fetch_add(1, Ordering::Relaxed);
                                            tris.get(v_w).fetch_add(1, Ordering::Relaxed);
                                            tris.get(*u_w).fetch_add(1, Ordering::Relaxed);
                                            tris.get(*er.get(u_v)).fetch_add(1, Ordering::Relaxed);
                                            tris.get(*er.get(v_w)).fetch_add(1, Ordering::Relaxed);
                                            tris.get(w_u).fetch_add(1, Ordering::Relaxed);
                                            total_triangles.fetch_add(2, Ordering::Relaxed);
                                        }
                                        OneOrMany::Many(u_ws) => {
                                            for &u_w in u_ws {
                                                let w_u = *er.get(u_w);
                                                tris.get(u_v).fetch_add(1, Ordering::Relaxed);
                                                tris.get(v_w).fetch_add(1, Ordering::Relaxed);
                                                tris.get(u_w).fetch_add(1, Ordering::Relaxed);
                                                tris.get(*er.get(u_v))
                                                    .fetch_add(1, Ordering::Relaxed);
                                                tris.get(*er.get(v_w))
                                                    .fetch_add(1, Ordering::Relaxed);
                                                tris.get(w_u).fetch_add(1, Ordering::Relaxed);
                                                total_triangles.fetch_add(2, Ordering::Relaxed);
                                            }
                                        }
                                    },
                                    None => continue,
                                };
                            }
                        }
                        let n_len = all_neighbours.len();
                        // #possible edges between neighbours = k * (k - 1) / 2
                        let n_possible_triangles = if n_len > 1 {
                            n_len as f64 * (n_len - 1) as f64 / 2.
                        } else {
                            0.
                        };
                        total_possible_triangles.fetch_add(n_possible_triangles, Ordering::Relaxed);
                        *local.get_mut(u) = n_possible_triangles;
                        all_neighbours.clear();
                        neighbours.clear();
                    }

                    synchronize.wait();

                    for u in begin..end {
                        let mut u_triangles: usize = 0;
                        for edge_idx in *index_ptr.get(u)..*index_ptr.get(u + 1) {
                            u_triangles += tris.get(edge_idx).load(Ordering::Relaxed) as usize;
                        }
                        let undirected_triangles = u_triangles.div_ceil(2) as f64;
                        let possible_triangles = *local.get(u);
                        let local_conductivity = if possible_triangles.is_normal() {
                            undirected_triangles / possible_triangles
                        } else {
                            0.
                        };
                        *local.get_mut(u) = local_conductivity;
                        local_clustering_sum.fetch_add(local_conductivity, Ordering::Relaxed);
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        self.local_average = local_clustering_sum.load(Ordering::Relaxed) / node_count as f64;
        self.transitivity = total_triangles.load(Ordering::Relaxed) as f64
            / 2.
            / total_possible_triangles.load(Ordering::Relaxed);

        // cleanup cache
        self.g.cleanup_cache(CacheFile::ClusteringCoefficient)?;

        Ok(())
    }
}
