use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use crossbeam::thread;
use num_cpus::get_physical;
use std::sync::{
    Arc, Barrier,
    atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering},
};

type ProceduralMemoryLiuEtAL = (
    AbstractedProceduralMemoryMut<AtomicU8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<AtomicBool>,
    AbstractedProceduralMemoryMut<u8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<usize>,
);

/// For the computation of a [`GraphMemoryMap`] instance's k-core decomposition as described in ["Parallel 𝑘-Core Decomposition: Theory and Practice"](https://doi.org/10.48550/arXiv.2502.08042) by Liu Y. et al.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoLiuEtAl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    /// Graph for which node/edge coreness is computed.
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    /// Memmapped slice containing the coreness of each edge.
    k_cores: AbstractedProceduralMemoryMut<u8>,
}

#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> AlgoLiuEtAl<'a, EdgeType, Edge> {
    /// Performs k-core decomposition as described in ["Parallel 𝑘-Core Decomposition: Theory and Practice"](https://doi.org/10.48550/arXiv.2502.08042) by Liu Y. et al.
    ///
    /// * Note: we did not implement the *Node Sampling*[^1] scheme optimization (used for high degree nodes), as our objective is the decomposition of very large sparse graphs.
    ///
    /// [^1]: details on how to implement this potimization and how it works can be found in the *4.1.2 Details about the Sampling Scheme.* section of the aforementioned paper in pp. 6-7.
    ///
    /// # Arguments
    ///
    /// * `graph` --- the [`GraphMemoryMap`] instance for which k-core decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let output_fn =
            cache_file_name(graph.cache_fst_filename(), FileType::KCoreLEA(H::H), None)?;
        let k_cores = SharedSliceMut::<u8>::abst_mem_mut(output_fn.clone(), graph.width(), true)?;
        let liu_et_al = Self { graph, k_cores };
        liu_et_al.compute(10)?;
        Ok(liu_et_al)
    }
    fn init_procedural_memory_liu_et_al(
        &self,
        mmap: u8,
    ) -> Result<ProceduralMemoryLiuEtAL, Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1;
        let edge_count = self.graph.width();

        let template_fn = self.graph.cache_edges_filename();
        let d_fn = cache_file_name(template_fn.clone(), FileType::KCoreLEA(H::H), Some(0))?;
        let ni_fn = cache_file_name(template_fn.clone(), FileType::KCoreLEA(H::H), Some(5))?;
        let a_fn = cache_file_name(template_fn.clone(), FileType::KCoreLEA(H::H), Some(1))?;
        let c_fn = cache_file_name(template_fn.clone(), FileType::KCoreLEA(H::H), Some(2))?;
        let f_fn = cache_file_name(template_fn.clone(), FileType::KCoreLEA(H::H), Some(3))?;
        let fs_fn = cache_file_name(template_fn.clone(), FileType::KCoreLEA(H::H), Some(4))?;

        let degree = SharedSliceMut::<AtomicU8>::abst_mem_mut(d_fn, node_count, mmap > 0)?;
        let node_index = SharedSliceMut::<usize>::abst_mem_mut(ni_fn, node_count, mmap > 3)?;
        let alive = SharedSliceMut::<AtomicBool>::abst_mem_mut(a_fn, node_count, mmap > 1)?;
        let coreness = SharedSliceMut::<u8>::abst_mem_mut(c_fn, node_count, mmap > 2)?;
        let frontier = SharedSliceMut::<usize>::abst_mem_mut(f_fn, edge_count, mmap > 3)?;
        let frontier_swap = SharedSliceMut::<usize>::abst_mem_mut(fs_fn, edge_count, mmap > 3)?;

        Ok((degree, node_index, alive, coreness, frontier, frontier_swap))
    }

    /// Computes the k-cores of a graph as described in ["Parallel 𝑘-Core Decomposition: Theory and Practice"](https://doi.org/10.48550/arXiv.2502.08042) by Liu Y. et al.
    ///
    /// The resulting k-core subgraphs are stored in memory (in a memmapped file) edgewise[^1].
    /// * Note: we did not implement the *Node Sampling* scheme optimization (used for high degree nodes), as our objective is the decomposition of very large sparse graphs. Details on how to implement this potimization and how it works can be found in the *4.1.2 Details about the Sampling Scheme.* section of the aforementioned paper in pp. 6-7.
    ///
    /// [^1]: for each edge of the graph it's coreness is stored in an array.
    ///
    /// # Arguments
    ///
    /// * `mmap` --- the level of memmapping to be used during the computation (*experimental feature*).
    ///
    pub fn compute(&self, mmap: u8) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1;
        let edge_count = self.graph.width();

        if node_count == 0 {
            return Ok(());
        }

        let threads = self.graph.thread_num().max(get_physical());
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = SharedSlice::<usize>::new(self.graph.index_ptr(), node_count + 1);
        let graph_ptr = SharedSlice::<Edge>::new(self.graph.edges_ptr(), edge_count);

        let (degree, _node_index, alive, coreness, frontier, swap) =
            self.init_procedural_memory_liu_et_al(mmap)?;

        // Initialize
        let total_dead_nodes = thread::scope(|s| -> usize {
            let mut dead_nodes = vec![];
            for tid in 0..threads {
                let mut degree = degree.shared_slice();
                let mut alive = alive.shared_slice();

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                dead_nodes.push(s.spawn(move |_| -> usize {
                    let mut dead_nodes = 0;

                    for u in start..end {
                        let deg_u = index_ptr.get(u + 1) - index_ptr.get(u);
                        *degree.get_mut(u) = AtomicU8::new(deg_u as u8);

                        if deg_u == 0 {
                            dead_nodes += 1;
                            *alive.get_mut(u) = AtomicBool::new(false);
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
            let _ = degree.flush_async();
            let _ = alive.flush_async();
            total_dead_nodes
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

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

        // --- core-peeling loop (Liu et al. algorithm) ---
        // for nodes with degree <= 16 no bucketing is used
        let mut k = 1u8;
        // number of vertices not yet peeled
        let remaining_global = Arc::new(AtomicUsize::new(node_count - total_dead_nodes));
        let frontier = SharedQueueMut::<usize>::from_shared_slice(frontier.shared_slice());
        let swap = SharedQueueMut::<usize>::from_shared_slice(swap.shared_slice());
        let synchronize = Arc::new(Barrier::new(threads));

        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            for tid in 0..threads {
                let degree = &degree;
                let alive = alive.shared_slice();
                let mut coreness = coreness.shared_slice();
                let mut frontier = frontier.clone();
                let mut swap = swap.clone();

                let remaining_global = Arc::clone(&remaining_global);
                let synchronize = Arc::clone(&synchronize);

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        let mut local_stack: Vec<usize> = vec![];
                        let max_len = 256;

                        while remaining_global.load(Ordering::Relaxed) > 0 {
                            synchronize.wait();
                            // build initial frontier = all vertices with degree <= k that are still active.
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
                                    _ => {
                                        return Err(format!(
                                            "error overflow when adding to k ({k} + 1)"
                                        )
                                        .into());
                                    }
                                };
                                continue;
                            }

                            // process subrounds for current k: peel all vertices with degree k.
                            while !frontier.is_empty() {
                                if tid == 0 {
                                    remaining_global.fetch_sub(frontier.len(), Ordering::Relaxed);
                                }

                                let chunk_size = frontier.len().div_ceil(threads);
                                let start = std::cmp::min(tid * chunk_size, frontier.len());
                                let end = std::cmp::min(start + chunk_size, frontier.len());

                                if let Some(chunk) = frontier.slice(start, end) {
                                    for i in 0..end - start {
                                        // set coreness and decrement neighbour degrees
                                        let u = *chunk.get(i);
                                        *coreness.get_mut(u) = k;

                                        // for each neighbor v of u:
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
                                                        if local_stack.len() < max_len {
                                                            local_stack.push(v);
                                                        } else {
                                                            let _ = swap.push(v);
                                                        }
                                                    }
                                                }
                                            }
                                        }

                                        // process local stack
                                        let mut read_in_stack: usize = 0;
                                        while let Some(u) = local_stack.pop() {
                                            // set coreness and decrement neighbour degrees
                                            read_in_stack += 1;
                                            *coreness.get_mut(u) = k;

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
                                                        let life = alive
                                                            .get(v)
                                                            .swap(false, Ordering::Relaxed);
                                                        if life {
                                                            if local_stack.len() < max_len {
                                                                local_stack.push(v);
                                                            } else {
                                                                let _ = swap.push(v);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                        remaining_global
                                            .fetch_sub(read_in_stack, Ordering::Relaxed);
                                    }
                                }

                                synchronize.wait();

                                swap = std::mem::replace(&mut frontier, swap).clear();

                                synchronize.wait();
                            }
                            k = match k.overflowing_add(1) {
                                (r, false) => r,
                                _ => {
                                    return Err(format!(
                                        "error overflow when adding to k ({k} + 1)"
                                    )
                                    .into());
                                }
                            };
                        }
                        Ok(())
                    },
                );
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;
        coreness.flush()?;

        // --- Compute per-edge core labels and write output ---
        // Create an output memory-mapped buffer for edge labels (u32 per directed edge).
        let out = self.k_cores.shared_slice();

        // parallel edge labeling: partition vertices among threads and write edge core values
        thread::scope(|scope| {
            let mut res = vec![];
            for tid in 0..threads {
                let coreness = coreness.shared_slice();
                let start = thread_load * tid;
                let end = std::cmp::min(start + thread_load, node_count);
                let mut edge_coreness = out;
                res.push(scope.spawn(move |_| -> Vec<usize> {
                    let mut res = vec![0usize; u8::MAX as usize];
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
            let joined_res: Vec<Vec<usize>> = res
                .into_iter()
                .map(|v| v.join().expect("error thread panicked"))
                .collect();
            let mut r = vec![0usize; u8::MAX as usize];
            for i in 0..u8::MAX as usize {
                for v in joined_res.clone() {
                    r[i] += v[i];
                }
            }
            r[0] += total_dead_nodes as usize;
            let mut max = 0;
            r.iter().enumerate().for_each(|(i, v)| {
                if *v != 0 && i > max {
                    max = i;
                }
            });
            r.resize(max + 1, 0);
            println!("k-cores {:?}", r);
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        // flush output to ensure all data is written to disk
        self.k_cores.flush_async()?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{k_core::verify_k_cores, test_common::get_or_init_dataset_cache_entry};

    use super::*;
    use paste::paste;
    use std::path::Path;

    macro_rules! graph_tests {
        ($($name:ident => $path:expr ,)*) => {
            $(
                paste! {
                    #[test]
                    fn [<k_cores_liu_et_al_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
                        generic_test($path)
                    }
                }
            )*
        }
    }

    fn generic_test<P: AsRef<Path> + Clone>(path: P) -> Result<(), Box<dyn std::error::Error>> {
        // let graph_cache =
        //     GraphCache::<TinyEdgeType, TinyLabelStandardEdge>::from_file(path, None, None, None)?;
        let graph_cache = get_or_init_dataset_cache_entry(path.as_ref())?;
        let graph = GraphMemoryMap::init(graph_cache, Some(16))?;
        let liu_et_al_k_cores = AlgoLiuEtAl::new(&graph)?;

        verify_k_cores(&graph, liu_et_al_k_cores.k_cores)?;
        Ok(())
    }

    // generate test cases from dataset
    graph_tests! {
        ggcat_1_5 => "../ggcat/graphs/random_graph_1_5.lz4",
        ggcat_2_5 => "../ggcat/graphs/random_graph_2_5.lz4",
        ggcat_3_5 => "../ggcat/graphs/random_graph_3_5.lz4",
        ggcat_4_5 => "../ggcat/graphs/random_graph_4_5.lz4",
        ggcat_5_5 => "../ggcat/graphs/random_graph_5_5.lz4",
        ggcat_6_5 => "../ggcat/graphs/random_graph_6_5.lz4",
        ggcat_7_5 => "../ggcat/graphs/random_graph_7_5.lz4",
        ggcat_8_5 => "../ggcat/graphs/random_graph_8_5.lz4",
        ggcat_9_5 => "../ggcat/graphs/random_graph_9_5.lz4",
        ggcat_1_10 => "../ggcat/graphs/random_graph_1_10.lz4",
        ggcat_2_10 => "../ggcat/graphs/random_graph_2_10.lz4",
        ggcat_3_10 => "../ggcat/graphs/random_graph_3_10.lz4",
        ggcat_4_10 => "../ggcat/graphs/random_graph_4_10.lz4",
        ggcat_5_10 => "../ggcat/graphs/random_graph_5_10.lz4",
        ggcat_6_10 => "../ggcat/graphs/random_graph_6_10.lz4",
        ggcat_7_10 => "../ggcat/graphs/random_graph_7_10.lz4",
        ggcat_8_10 => "../ggcat/graphs/random_graph_8_10.lz4",
        ggcat_9_10 => "../ggcat/graphs/random_graph_9_10.lz4",
        // ggcat_8_15 => "../ggcat/graphs/random_graph_8_15.lz4",
        // ggcat_9_15 => "../ggcat/graphs/random_graph_9_15.lz4",
        // … add the rest
    }
}
