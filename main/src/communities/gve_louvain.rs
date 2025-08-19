use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use atomic_float::AtomicF64;
use crossbeam::thread;
use num_cpus::get_physical;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::Barrier;
use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

type ProceduralMemoryGVELouvain = (
    // K' --- renamed to k
    AbstractedProceduralMemoryMut<AtomicUsize>,
    // Σ' --- renamed to sigma
    AbstractedProceduralMemoryMut<AtomicUsize>,
    // first CSR
    // G'.index --- renamed to gdi
    AbstractedProceduralMemoryMut<usize>,
    // G'.edges --- renamed to gde
    AbstractedProceduralMemoryMut<(usize, usize)>,
    // second CSR
    // G''.index --- renamed to gddi
    AbstractedProceduralMemoryMut<usize>,
    // G''.edges --- renamed to gdde
    AbstractedProceduralMemoryMut<(usize, usize)>,
    // processed
    AbstractedProceduralMemoryMut<AtomicBool>,
    // C' --- renamed to coms (C is stored as a member of the struct)
    AbstractedProceduralMemoryMut<usize>,
    // Helper array for C' renumbering, dendrogram lookup & CSR aggregation
    AbstractedProceduralMemoryMut<usize>,
);

#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoGVELouvain<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    /// memmapped slice containing the coreness of each edge
    community: AbstractedProceduralMemoryMut<usize>,
    community_count: usize,
    modularity: f64,
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

    const MAX_PASSES: usize = 25;

    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let output_filename =
            cache_file_name(graph.cache_fst_filename(), FileType::GVELouvain, None)?;
        let community =
            SharedSliceMut::<usize>::abst_mem_mut(output_filename.clone(), graph.width(), true)?;
        let mut gve_louvain = Self {
            graph,
            community,
            community_count: 0,
            modularity: 0.,
        };
        gve_louvain.compute(10)?;
        Ok(gve_louvain)
    }

    pub fn community_count(&self) -> usize {
        self.community_count
    }

    pub fn partition_modularity(&self) -> f64 {
        self.modularity
    }

    fn init_procedural_memory_gve_louvain(
        &self,
        mmap: u8,
    ) -> Result<ProceduralMemoryGVELouvain, Box<dyn std::error::Error>> {
        let edge_count = self.graph.width();
        let node_count = self.graph.size() - 1;

        let template_fn = self.graph.cache_index_filename();
        let k_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(0))?;
        let s_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(1))?;
        let gdi_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(2))?;
        let gde_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(3))?;
        let gddi_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(4))?;
        let gdde_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(5))?;
        let a_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(6))?;
        let c_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(7))?;
        let nc_fn = cache_file_name(template_fn.clone(), FileType::GVELouvain, Some(8))?;

        let k = SharedSliceMut::<AtomicUsize>::abst_mem_mut(k_fn, node_count, mmap > 0)?;
        let sigma = SharedSliceMut::<AtomicUsize>::abst_mem_mut(s_fn, node_count, mmap > 0)?;
        let gdi = SharedSliceMut::<usize>::abst_mem_mut(gdi_fn, 2 * node_count, mmap > 0)?;
        let gde = SharedSliceMut::<(usize, usize)>::abst_mem_mut(gde_fn, edge_count, mmap > 0)?;
        let gddi = SharedSliceMut::<usize>::abst_mem_mut(gddi_fn, 2 * node_count, mmap > 0)?;
        let gdde = SharedSliceMut::<(usize, usize)>::abst_mem_mut(gdde_fn, edge_count, mmap > 0)?;
        let processed = SharedSliceMut::<AtomicBool>::abst_mem_mut(a_fn, node_count, mmap > 0)?;
        let coms = SharedSliceMut::<usize>::abst_mem_mut(c_fn, node_count, mmap > 0)?;
        // has to be size |E| as it is used to store the 'holey' CSR
        let helper = SharedSliceMut::<usize>::abst_mem_mut(nc_fn, edge_count, true)?;

        Ok((k, sigma, gdi, gde, gddi, gdde, processed, coms, helper))
    }

    /// Computes the differencial modularity upon moving a node `u`, from a community `d` to a
    /// community `c`:
    ///
    /// 𝛿𝑄 = (Ku->c - Ku->d) / m - (Ku * (Σc - Σd) / (2m^2)).
    ///
    /// # Arguments
    ///
    /// * `k_u_c`: f64 --- The weight of edges from node `u` to nodes in community `c`.
    /// * `k_u_d`: f64 --- The weight of edges from node `u` to nodes in community `d`.
    /// * `k_u`: f64 --- Total weight of edges from node `u`.
    /// * `sig_c`: f64 --- Total weight of edges in community `c`.
    /// * `sig_d`: f64 --- Total weight of edges in community `d`.
    /// * `m`: f64 --- Total weight of edges in the graph (or half of it if graphis directed).
    ///
    /// # Returns
    ///
    /// Difference in modularity upon moving node `u` from community `d` to community `c`.
    #[inline(always)]
    fn delta_q(k_u_c: f64, k_u_d: f64, k_u: f64, sig_c: f64, sig_d: f64, m: f64) -> f64 {
        assert!(k_u_c >= 0. && k_u_d >= 0. && k_u >= 0. && sig_c >= 0. && sig_d >= 0. && m >= 0.);
        assert!(
            k_u_c.is_finite()
                && k_u_d.is_finite()
                && k_u.is_finite()
                && sig_c.is_finite()
                && sig_d.is_finite()
                && m.is_normal()
        );
        let dq = (k_u_c - k_u_d) / m - k_u * (k_u + sig_c - sig_d) / (2. * m * m);
        if dq > 1. {
            println!(
                "Deviant dq: ({k_u_c} - {k_u_d}) / {m} - {k_u} * ({k_u} + {sig_c} - {sig_d}) / (2. *{m} * {m}) = {dq}"
            );
        }
        dq
    }

    /// Determines the total weight of edges from a given node `u` to every community `c` it is linked to.
    ///
    /// # Arguments
    ///
    /// * `nmap`: &mut HashMap<usize, usize> (updated) --- Hashmap for entries `(c, k_u_c)`, where `c` is a community and `k_u_c` is the total weight of edges from node u into nodes in `c`.
    /// * `edges`: SharedSliceMut<(usize, usize)> --- Edges vector, should consist of entries `(dest_node: usize, edge_weight: usize)`.
    /// * `index`: SharedSliceMut<usize> --- Index vector.
    /// * `coms`: SharedSliceMut<usize> --- Communities vector.
    /// * `u`: usize --- Index of node `u`.
    /// * `self_allowed`: bool --- Flag signaling whether self-loops should be accepted.
    #[inline(always)]
    fn scan_communities(
        comm_count: usize,
        nmap: &mut HashMap<usize, usize>,
        edges: SharedSliceMut<(usize, usize)>,
        index: SharedSliceMut<usize>,
        coms: SharedSliceMut<usize>,
        u: usize,
        self_allowed: bool,
    ) {
        for e_idx in *index.get(2 * u)..*index.get(2 * u + 1) {
            let e = edges.get(e_idx);
            if !self_allowed && e.0 == u {
                continue;
            }
            if *coms.get(e.0) > comm_count {
                println!(
                    "WARNING!! Found for node {u} (neighbour {}): {{{} > {comm_count}}} but c: {}",
                    e.0,
                    *coms.get(e.0),
                    *coms.get(*coms.get(e.0))
                );
                println!(
                    "{}..{} at {e_idx}\n",
                    *index.get(2 * u),
                    *index.get(2 * u + 1)
                );
            }
            // 𝐻𝑡[𝐶'[v]] ← 𝐻𝑡[𝐶'[v]] + w
            *nmap.entry(*coms.get(e.0)).or_insert(0) += e.1;
        }
    }

    /// Determines the best community a node `u` may be switched into from its neighbours community set.
    ///
    /// # Arguments
    ///
    /// * `nmap`: &mut HashMap<usize, usize> --- Hashmap of entries `(c, k_u_c)`, where `c` is a community and `k_u_c` is the total weight of edges from node `u` into nodes in `c`.
    /// * `sigma`: SharedSliceMut<AtomicUsize> --- Total community weights vector.
    /// * `c_u`: usize --- Community of node `u`.
    /// * `k:u`: f64 --- Total weight of node `u` <<f64>>.
    /// * `m`: f64 --- Total weight of edges in the graph (or half of it if graph is directed) <<f64>>.
    ///
    /// # Returns
    ///
    /// `(c*: usize, 𝛿𝑄*: f64)`, where:
    /// * `c*` is the best community node u may be switched into
    /// * `𝛿𝑄*` is the delta modularity arising thereof.
    #[inline(always)]
    fn choose_community(
        nmap: &mut HashMap<usize, usize>,
        sigma: SharedSliceMut<AtomicUsize>,
        c_u: usize,
        k_u: f64,
        m: f64,
    ) -> (usize, f64) {
        // houses c*
        let mut best_gain = 0.;
        // houses 𝛿𝑄*
        let mut best_comm = c_u;

        // Ku->d
        let k_u_d = nmap.get(&c_u).cloned().unwrap_or(0) as f64;
        // Σd
        let sig_d = sigma.get(c_u).load(Ordering::Relaxed) as f64;

        for (&c_v, &edge_weight) in nmap.iter() {
            if c_v == c_u {
                continue;
            }

            // Ku->c
            let k_u_c = edge_weight as f64;
            // Σc
            let sig_c = sigma.get(c_v).load(Ordering::Relaxed) as f64;

            // 𝛿𝑄 = (Ku->c - Ku->d) / m - (Ku * (Σc - Σd) / (2m^2))
            let dq = Self::delta_q(k_u_c, k_u_d, k_u, sig_c, sig_d, m);
            if dq > best_gain {
                // 𝛿𝑄∗ = 𝛿𝑄
                best_gain = dq;
                // *c = C'[v]
                best_comm = c_v;
            }
        }
        // (c*, 𝛿𝑄*)
        (best_comm, best_gain)
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn louvain_move(
        l_pass: usize,
        comm_count: usize,
        k: SharedSliceMut<AtomicUsize>,
        sigma: SharedSliceMut<AtomicUsize>,
        processed: SharedSliceMut<AtomicBool>,
        idx: SharedSliceMut<usize>,
        mut next_index: SharedSliceMut<usize>,
        e: SharedSliceMut<(usize, usize)>,
        mut c: SharedSliceMut<usize>,
        mut helper: SharedSliceMut<usize>,
        m: f64,
        tolerance: f64,
        threads: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        // recalculate node load per thread
        let node_load = comm_count.div_ceil(threads);

        let mut l_iter = 0;
        let synchronize = Arc::new(Barrier::new(threads));
        for i in 0..Self::MAX_ITERATIONS {
            let global_delta_q = Arc::new(AtomicF64::new(0.0));

            // reset iteration metrics
            thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
                // use dynamic style scheduling by splitting range [0, cur_node_count) for threads
                let mut thread_res = Vec::with_capacity(threads);
                for tid in 0..threads {
                    let k = k.clone();
                    let sigma = sigma.clone();
                    let processed = processed.clone();

                    let global_delta_q = global_delta_q.clone();
                    let synchronize = synchronize.clone();

                    let begin = std::cmp::min(tid * node_load, comm_count);
                    let end = std::cmp::min(begin + node_load, comm_count);

                    // each thread gets its own local HashMap for accumulating neighbor community weights
                    thread_res.push(scope.spawn(
                        move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                            // if pass > 0, we need to initialize communities for the new graph (each super-vertex in its own community)
                            if l_pass != 0 && i == 0 {
                                for u in begin..end {
                                    // unprune all nodes
                                    *helper.get_mut(u) = usize::MAX;
                                    *next_index.get_mut(u * 2) = 0;
                                    // reassign node community to itself
                                    *c.get_mut(u) = u;
                                }
                            }

                            synchronize.wait();

                            let mut local_delta_q_sum = 0.0;
                            let mut nmap: HashMap<usize, usize> = HashMap::new();
                            for u in begin..end {
                                // skip this vertex if it was marked as pruned (no need to process)
                                if processed.get(u).swap(true, Ordering::Relaxed) {
                                    continue;
                                }
                                let c_u = *c.get(u);
                                // get weight of u k[u]
                                let k_u = k.get(u).load(Ordering::Relaxed);
                                // gather neighbour communities and edge weight sums using the adjacency list of u

                                nmap.clear();
                                let nm = &mut nmap;
                                Self::scan_communities(comm_count, nm, e, idx, c, u, false);

                                let s = sigma.clone();
                                let (b_c, b_g) = Self::choose_community(nm, s, c_u, k_u as f64, m);

                                // move u to the best community if it yields positive gain
                                if b_c != c_u && b_g > 1e-12 {
                                    // atomic updates to community totals --- if node has already
                                    // been moved don't do anything
                                    if sigma
                                        .get(c_u)
                                        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |s| {
                                            if s < k_u {
                                                None
                                            } else {
                                                Some(s - k_u)
                                            }
                                        })
                                        .is_ok()
                                    {
                                        sigma.get(b_c).fetch_add(k_u, Ordering::Relaxed);
                                        // update community assignment of u
                                        *c.get_mut(u) = b_c;
                                        local_delta_q_sum += b_g;
                                        // mark all neighbors of u to be processed in the next iteration
                                        for idx in *idx.get(u * 2)..*idx.get(u * 2 + 1) {
                                            processed
                                                .get(e.get(idx).0)
                                                .store(false, Ordering::Relaxed);
                                        }
                                    } else {
                                        return Err(
                                            format!("error louvain_move(): {} {u} can't be removed from its community {c_u} as node weight is {k_u} but community total weight is only {}", 
                                                    if l_pass == 0 {"node"} else {"super-node"}, sigma.get(c_u).load(Ordering::Relaxed)
                                                    ).into()
                                            );
                                    }
                                }
                            }
                            // atomically accumulate thread-local result to global
                            global_delta_q.fetch_add(local_delta_q_sum, Ordering::Relaxed);
                            Ok(())
                        },
                    ));
                }
                for (i, r) in thread_res.into_iter().enumerate() {
                    let t_res = r.join();
                    if t_res.is_err() {
                        return Err(
                            format!("error joining {i}, couldn't get 𝛿𝑄: {:?}", t_res).into()
                        );
                    }
                }
                Ok(())
            })
            .unwrap()?;

            // check convergence criteria for the local-moving phase
            if global_delta_q.load(Ordering::Relaxed).abs() < tolerance {
                break;
            }
            l_iter += 1;
        }
        Ok(l_iter)
    }

    fn compute(&mut self, mmap: u8) -> Result<(), Box<dyn std::error::Error>> {
        let edge_count = self.graph.width();
        let node_count = self.graph.size() - 1;

        let threads = self.graph.thread_num().max(get_physical());
        let node_load = node_count.div_ceil(threads);

        let (k, sigma, gdi, gde, gddi, gdde, processed, coms, helper) =
            self.init_procedural_memory_gve_louvain(mmap)?;

        // C --- the community vector of the original nodes
        let mut communities = self.community.shared_slice();
        let index_ptr = SharedSlice::<usize>::new(self.graph.index_ptr(), node_count + 1);
        let graph_ptr = SharedSlice::<Edge>::new(self.graph.edges_ptr(), edge_count);

        // initialize
        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let processed = processed.shared_slice();
                let mut gdi = gdi.shared_slice();
                let mut gddi = gddi.shared_slice();
                let mut gde = gde.shared_slice();
                let mut k = k.shared_slice();
                let mut sigma = sigma.shared_slice();
                let mut coms = coms.shared_slice();
                let mut helper = helper.shared_slice();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                threads_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        for u in begin..end {
                            let u_start = *index_ptr.get(u);
                            let u_end = *index_ptr.get(u + 1);
                            let out_offset = u * 2;
                            let deg_u = match u_end.overflowing_sub(u_start) {
                                (r, false) => r,
                                (_, true) => {
                                    return Err(format!(
                                        "overflow calculating degree of {u}: {u_end} - {u_start}"
                                    )
                                    .into());
                                }
                            };
                            let edges = match graph_ptr.slice(u_start, u_end) {
                                Some(e) => e
                                    .iter()
                                    .map(|e| (e.dest(), 1))
                                    .collect::<Vec<(usize, usize)>>(),
                                None => {
                                    return Err(format!("error reading node {u} in init").into());
                                }
                            };

                            // mark all nodes unprocessed
                            processed.get(u).swap(false, Ordering::Relaxed);
                            *helper.get_mut(u) = usize::MAX;
                            *communities.get_mut(u) = u;
                            *coms.get_mut(u) = u;
                            *gdi.get_mut(out_offset) = u_start;
                            *gdi.get_mut(out_offset + 1) = u_end;
                            *gddi.get_mut(out_offset) = 0;
                            // copy G onto G'
                            gde.write_slice(u_start, edges.as_slice());
                            // sum of edge weights, unweighted edges (in pass 0, edge weight is 1)
                            // Σ′ ← 𝐾′ ← 𝑣𝑒𝑟𝑡𝑒𝑥_w𝑒𝑖𝑔ℎ𝑡𝑠(𝐺′)
                            *k.get_mut(u) = AtomicUsize::new(deg_u);
                            *sigma.get_mut(u) = AtomicUsize::new(deg_u);
                        }
                        Ok(())
                    },
                ));
            }
            for (tid, r) in threads_res.into_iter().enumerate() {
                let t_res = r.join();
                if t_res.is_err() {
                    return Err(format!("error (thread {tid}): {:?}", t_res).into());
                }
            }
            Ok(())
        })
        .unwrap()?;

        let mut tolerance = Self::INITIAL_TOLERANCE;

        let mut prev_comm_count = node_count;
        let mut new_comm_count: usize = 0;

        // 2m (since each undirected edge has weight 2)
        let m2 = edge_count as f64;
        let m = m2 / 2.;

        let mut coms = coms.shared_slice();
        let mut helper = helper.shared_slice();

        let mut index = gdi.shared_slice();
        let mut next_index = gddi.shared_slice();
        let mut edges = gde.shared_slice();
        let mut next_edges = gdde.shared_slice();

        let k = k.shared_slice();
        let sigma = sigma.shared_slice();
        let processed = processed.shared_slice();

        // outer loop: louvain passes
        for l_pass in 0..Self::MAX_PASSES {
            if prev_comm_count <= 1 {
                break;
            }

            // Σ′ ← 𝐾′ ← 𝑣𝑒𝑟𝑡𝑒𝑥_w𝑒𝑖𝑔ℎ𝑡𝑠(𝐺′) ; 𝐶′ ← [0..|𝑉′|) (initialization inside louvain_move())
            // li <- louvain_move(G', C', K', Σ′)
            // if 𝑙𝑖 ≤ 1 then break --- adjusted to 𝑙𝑖 ≤ 0 as we don't increment before loop ends
            // as is teh case with the algorithm author
            let count_iter = Self::louvain_move(
                l_pass,
                prev_comm_count,
                k.clone(),
                sigma.clone(),
                processed.clone(),
                index,
                next_index,
                edges,
                coms,
                helper,
                m,
                tolerance,
                threads,
            )?;
            if count_iter == 0 {
                println!("Finished at {l_pass}th pass iters were <= 1");
                break;
            } else {
                println!("Continue, {l_pass}th pass had {count_iter} iterations");
            }

            // // WARN: Test weight conservation over louvain_move()
            // let mut sum_k = 0usize;
            // let mut sum_s = 0usize;
            // for c in 0..prev_comm_count {
            //     let kc = k.get(c).load(Ordering::Relaxed);
            //     let sc = sigma.get(c).load(Ordering::Relaxed);
            //     sum_k += kc;
            //     sum_s += sc;
            // }
            // // For a directed-symmetric representation, sum_k must equal m2.
            // assert_eq!(sum_k, m2 as usize, "∑k != m2");
            // assert_eq!(sum_s, m2 as usize, "∑s != m2");

            // to simplify code dendrogram lookup is always done after louvain_move()
            // (if global conversion did not happen) this way no dendrogram lookup
            // is necessary after louvain() terminates

            // |Γ|, |Γ𝑜𝑙𝑑 | ← Number of communities in 𝐶′, C;
            // C′ ← Renumber communities in 𝐶
            // count communities while at it
            println!("community renumbering");

            for u in 0..prev_comm_count {
                let degree = *index.get(u * 2 + 1) - *index.get(u * 2);
                let c = *coms.get(u);
                let new_c = *helper.get(c);
                if new_c == usize::MAX {
                    *helper.get_mut(c) = new_comm_count;
                    *coms.get_mut(u) = new_comm_count;
                    *next_index.get_mut(new_comm_count * 2) = degree;
                    new_comm_count += 1;
                } else {
                    *coms.get_mut(u) = new_c;
                    *next_index.get_mut(new_c * 2) += degree;
                }
            }

            // 𝐶 ← Lookup dendrogram using 𝐶 to 𝐶
            for orig in 0..node_count {
                let c = *communities.get(orig);
                if c < prev_comm_count {
                    *communities.get_mut(orig) = *coms.get(c);
                } else {
                    return Err(
                        format!(
                            "error performing dendrogram lookup: node {orig} is in community {c} but max_id for communities is {prev_comm_count} in the {l_pass}th pass"
                            ).into()
                        );
                }
            }

            // if |Γ|/|Γ𝑜𝑙𝑑 | > 𝜏_𝑎𝑔𝑔 then break
            if new_comm_count as f64 / prev_comm_count as f64 > Self::AGGREGATION_TOLERANCE {
                prev_comm_count = new_comm_count;
                break;
            }

            // build the aggregated graph (super-vertex graph) in the 'next_index' and 'next_edges' buffers
            {
                println!("aggregation");
                // exclusive scan
                let mut sum = 0;
                for u in 0..new_comm_count {
                    let degree = *next_index.get(u * 2);
                    *next_index.get_mut(u * 2) = sum;
                    *next_index.get_mut(u * 2 + 1) = sum;
                    sum += degree;
                }
                // add edge atomically
                for u in 0..prev_comm_count {
                    let degree = *index.get(u * 2 + 1) - *index.get(u * 2);
                    // guard against isolated nodes to not go over com boundary
                    if degree > 0 {
                        // add edge (C'[u], u)
                        *helper.get_mut(*next_index.get(*coms.get(u) * 2 + 1)) = u;
                        *next_index.get_mut(*coms.get(u) * 2 + 1) += 1;
                    }
                }
                // fill edges of new graph
                let mut nm: HashMap<usize, usize> = HashMap::new();
                for c in 0..new_comm_count {
                    processed.get(c).store(false, Ordering::Relaxed);
                    nm.clear();
                    for e_idx in *next_index.get(c * 2)..*next_index.get(c * 2 + 1) {
                        let u = *helper.get(e_idx);
                        Self::scan_communities(
                            prev_comm_count,
                            &mut nm,
                            edges,
                            index,
                            coms,
                            u,
                            true,
                        );
                    }
                    let mut total = 0;
                    let mut total_weight = 0;
                    for (&d, &w) in nm.iter() {
                        if d > new_comm_count {
                            return Err(
                                format!("error building supergraph: found edge to community {d} but max_id for communities is {prev_comm_count} in the {l_pass}th pass"
                                        ).into()
                                );
                        }
                        *next_edges.get_mut(*next_index.get(c * 2) + total) = (d, w);
                        total += 1;
                        total_weight += w;
                    }
                    // println!("({c} -> {{|edges| = {total}, |weight| = {total_weight}}})");
                    *next_index.get_mut(c * 2 + 1) = *next_index.get(c * 2) + total;
                    if c > 0 && *next_index.get(c * 2) < *next_index.get(c * 2 - 1) {
                        return Err(
                                format!("error building supergraph: index missalignment, node {} ends at {} and node {c} starts at {}",
                                        c - 1, *next_index.get(c * 2 - 1), *next_index.get(c * 2)
                                        ).into()
                                );
                    }
                    k.get(c).store(total_weight, Ordering::Relaxed);
                    sigma.get(c).store(total_weight, Ordering::Relaxed);
                }
            }

            let mut sum_k = 0usize;
            for c in 0..new_comm_count {
                // WARN: Don't remove community id reassignment!!!
                let kc = k.get(c).load(Ordering::Relaxed);
                let sc = sigma.get(c).load(Ordering::Relaxed);
                assert_eq!(kc, sc, "k[c] != sigma[c] at c={}", c);
                sum_k += kc;
            }
            // For a directed-symmetric representation, sum_k must equal m2.
            assert_eq!(sum_k, m2 as usize, "∑k != m2");

            for c in 0..new_comm_count {
                let s = *next_index.get(2 * c);
                let e = *next_index.get(2 * c + 1);
                assert!(
                    s <= e && e <= edge_count /* of the buffer you’re writing */
                );
                // also walk a few entries defensively
                for pos in s..e {
                    let (dst, w) = *next_edges.get(pos);
                    assert!(dst < new_comm_count, "edge dst OOB");
                    assert!(w > 0, "zero/garbage weight");
                }
            }

            // check aggregation tolerance: if communities did not reduce sufficiently, break cycle
            if new_comm_count as f64 >= (prev_comm_count as f64 * Self::AGGREGATION_TOLERANCE) {
                prev_comm_count = new_comm_count;
                break;
            }

            // tighten tolerance for next pass (threshold scaling)
            // 𝜏 = 𝜏 / TOLERANCE_DROP
            tolerance /= Self::TOLERANCE_DROP;

            // prepare for next loop by resetting community count and switching CSRs
            std::mem::swap(&mut index, &mut next_index);
            std::mem::swap(&mut edges, &mut next_edges);

            println!(
                "{l_pass}th pass comms {prev_comm_count} -> {new_comm_count} (aggregation rate = {})",
                new_comm_count as f64 / prev_comm_count as f64
            );

            prev_comm_count = new_comm_count;
            new_comm_count = 0;
        } // end of passes loop

        self.community_count = prev_comm_count;

        if m2 == 0. {
            self.modularity = 0.0;
            return Ok(());
        }

        for u in 0..node_count {
            k.get(*communities.get(u)).store(0, Ordering::Relaxed);
            sigma.get(*communities.get(u)).store(0, Ordering::Relaxed);
        }

        // sum internal degree ---> sigma | sum total degree ---> k
        for u in 0..node_count {
            let comm_u = *communities.get(u);
            let iter = self.graph.neighbours(u)?;
            k.get(*communities.get(u))
                .fetch_add(iter.remaining_neighbours(), Ordering::Relaxed);
            for e in iter {
                let v = e.dest();
                if v >= node_count {
                    continue;
                } // safety
                if *communities.get(v) == comm_u {
                    sigma.get(comm_u).fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        // modularity: sum_c( internal_dir_edges[c]/m2 - (Kc[c]/m2)^2 )
        let mut partition_modularity = 0.0f64;
        for c in 0..prev_comm_count {
            let lc_over_m2 = (sigma.get(c).load(Ordering::Relaxed) as f64) / m2;
            let kc_over_m2 = (k.get(c).load(Ordering::Relaxed) as f64) / m2;
            partition_modularity += lc_over_m2 - kc_over_m2 * kc_over_m2;
        }

        self.modularity = partition_modularity;

        cleanup_cache()?;
        Ok(())
    }
}
