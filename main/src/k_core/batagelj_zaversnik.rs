use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use crossbeam::thread;
use num_cpus::get_physical;

type ProceduralMemoryBZ = (
    AbstractedProceduralMemoryMut<u8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<u8>,
    AbstractedProceduralMemoryMut<usize>,
);

/// For the computation of a [`GraphMemoryMap`] instance's k-core decomposition as described in ["An O(m) Algorithm for Cores Decomposition of Networks"](https://doi.org/10.48550/arXiv.cs/0310049) by Batagelj V. and Zaversnik M.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoBatageljZaversnik<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    /// Graph for which node/edge coreness is computed.
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    /// Memmapped slice containing the coreness of each edge.
    k_cores: AbstractedProceduralMemoryMut<u8>,
}

#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>
    AlgoBatageljZaversnik<'a, EdgeType, Edge>
{
    /// Performs k-core decomposition as described in ["An O(m) Algorithm for Cores Decomposition of Networks"](https://doi.org/10.48550/arXiv.cs/0310049) by Batagelj V. and Zaversnik M.
    ///
    /// The resulting k-core subgraphs are stored in memory (in a memmapped file) edgewise[^1].
    ///
    /// [^1]: for each edge of the graph it's coreness is stored in an array.
    ///
    /// # Arguments
    ///
    /// * `graph` --- the [`GraphMemoryMap`] instance for which k-core decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let output_filename = cache_file_name(graph.cache_fst_filename(), FileType::KCoreBZ, None)?;
        let k_cores =
            SharedSliceMut::<u8>::abst_mem_mut(output_filename.clone(), graph.width(), true)?;
        let bz = Self { graph, k_cores };
        bz.compute(10)?;
        Ok(bz)
    }

    fn init_procedural_memory_bz(
        &self,
        mmap: u8,
    ) -> Result<ProceduralMemoryBZ, Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1;

        let template_fn = self.graph.cache_edges_filename();
        let d_fn = cache_file_name(template_fn.clone(), FileType::KCoreBZ, Some(0))?;
        let n_fn = cache_file_name(template_fn.clone(), FileType::KCoreBZ, Some(1))?;
        let c_fn = cache_file_name(template_fn.clone(), FileType::KCoreBZ, Some(2))?;
        let p_fn = cache_file_name(template_fn.clone(), FileType::KCoreBZ, Some(3))?;

        let degree = SharedSliceMut::<u8>::abst_mem_mut(d_fn, node_count, mmap > 0)?;
        let node = SharedSliceMut::<usize>::abst_mem_mut(n_fn, node_count, mmap > 1)?;
        let core = SharedSliceMut::<u8>::abst_mem_mut(c_fn, node_count, mmap > 2)?;
        let pos = SharedSliceMut::<usize>::abst_mem_mut(p_fn, node_count, mmap > 3)?;

        Ok((degree, node, core, pos))
    }

    /// Computes the k-cores of a graph as described in ["An O(m) Algorithm for Cores Decomposition of Networks"](https://doi.org/10.48550/arXiv.cs/0310049) by Batagelj V. and Zaversnik M.
    ///
    /// The resulting k-core subgraphs are stored in memory (in a memmapped file) edgewise, i.e.
    /// for each edge of the graph it's coreness is stored in an array.
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

        let (degree, mut node, mut core, mut pos) = self.init_procedural_memory_bz(mmap)?;
        // compute out-degrees in parallel
        let index_ptr = SharedSlice::<usize>::new(self.graph.index_ptr(), node_count + 1);
        let graph_ptr = SharedSlice::<Edge>::new(self.graph.edges_ptr(), edge_count);

        // initialize degree and bins count vecs
        let mut bins: Vec<usize> = thread::scope(
            |scope| -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync>> {
                let mut bins = vec![0usize; u8::MAX as usize];
                let mut max_vecs = vec![];

                for tid in 0..threads {
                    let mut deg_arr = degree.shared_slice();

                    let start = std::cmp::min(tid * thread_load, node_count);
                    let end = std::cmp::min(start + thread_load, node_count);

                    max_vecs.push(scope.spawn(
                        move |_| -> Result<Vec<usize>, Box<dyn std::error::Error + Send + Sync>> {
                            let mut bins: Vec<usize> = vec![0; u8::MAX as usize];
                            for v in start..end {
                                let deg = index_ptr.get(v + 1) - index_ptr.get(v);
                                if deg > u8::MAX as usize {
                                    return Err(format!(
                                        "error degree[{v}] == {deg} but max suported is {}",
                                        u8::MAX
                                    )
                                    .into());
                                }
                                bins[deg] += 1;
                                *deg_arr.get_mut(v) = deg as u8;
                            }
                            Ok(bins)
                        },
                    ));
                }
                // join results
                for handle_bin in max_vecs {
                    let joined_bin = handle_bin.join().map_err(
                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("{:?}", e).into()
                        },
                    )?;
                    for entries in joined_bin.into_iter() {
                        for (degree, count) in entries.iter().enumerate() {
                            bins[degree] += *count;
                        }
                    }
                }
                Ok(bins)
            },
        )
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        let (max_degree, _) = bins
            .iter()
            .enumerate()
            .max_by_key(|(deg, c)| *deg * (if **c != 0 { 1 } else { 0 }))
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "error couldn't get max degree".into()
            })?;
        bins.resize(max_degree + 1, 0);

        // println!()
        let dead_nodes = bins[0];

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
        let mut degree = degree.shared_slice();
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

        let out_slice = self.k_cores.shared_slice();

        thread::scope(|scope| {
            let mut res = vec![];
            for tid in 0..threads {
                let mut out_ptr = out_slice;
                let core = core.shared_slice();

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                res.push(scope.spawn(move |_| -> Vec<usize> {
                    let mut res = vec![0usize; u8::MAX as usize];
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
            // safe because max_degree is at least 0
            r[0] += dead_nodes;
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
                    fn [<k_cores_batagelj_zaversik_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
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
        let bz_k_cores = AlgoBatageljZaversnik::new(&graph)?;

        verify_k_cores(&graph, bz_k_cores.k_cores)?;
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
        // ggcat_6_10 => "../ggcat/graphs/random_graph_6_10.lz4",
        // ggcat_7_10 => "../ggcat/graphs/random_graph_7_10.lz4",
        // ggcat_8_10 => "../ggcat/graphs/random_graph_8_10.lz4",
        // ggcat_9_10 => "../ggcat/graphs/random_graph_9_10.lz4",
        // ggcat_8_15 => "../ggcat/graphs/random_graph_8_15.lz4",
        // ggcat_9_15 => "../ggcat/graphs/random_graph_9_15.lz4",
        // … add the rest
    }
}
