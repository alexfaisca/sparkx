use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use crossbeam::thread;
use num_cpus::get_physical;
use std::sync::Arc;
use std::sync::Barrier;
use std::{
    collections::HashMap,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
};

type ProceduralMemoryGVELouvain<Edge> = (
    // K' --- renamed to k
    AbstractedProceduralMemoryMut<AtomicUsize>,
    // Σ' --- renamed to sigma
    AbstractedProceduralMemoryMut<AtomicUsize>,
    // first CSR
    // G'.index --- renamed to gdi
    AbstractedProceduralMemoryMut<usize>,
    // G'.edges --- renamed to gde
    AbstractedProceduralMemoryMut<Edge>,
    // second CSR
    // G''.index --- renamed to gddi
    AbstractedProceduralMemoryMut<usize>,
    // G''.edges --- renamed to gdde
    AbstractedProceduralMemoryMut<Edge>,
    // processed -- renamed to alive
    AbstractedProceduralMemoryMut<AtomicBool>,
    // C' --- renamed to coms (C is stored as a member of the struct)
    AbstractedProceduralMemoryMut<usize>,
    // Helper array for C' renumbering and dendrogram loopup
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
    const MAX_ITERATIONS: usize = 40;
    /// Described in 4.1.3 Adjusting tolerance drop rate (threshold scaling)
    const TOLERANCE_DROP: f64 = 10.;
    /// Described in 4.1.4 Adjusting initial tolerance
    const INITIAL_TOLERANCE: f64 = 0.01;
    /// Described in 4.1.4 Adjusting aggregation tolerance
    const AGGREGATION_TOLERANCE: f64 = 1.0;

    const MAX_PASSES: usize = 20;

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
    ) -> Result<ProceduralMemoryGVELouvain<Edge>, Box<dyn std::error::Error>> {
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
        let gde = SharedSliceMut::<Edge>::abst_mem_mut(gde_fn, edge_count, mmap > 0)?;
        let gddi = SharedSliceMut::<usize>::abst_mem_mut(gddi_fn, 2 * node_count, mmap > 0)?;
        let gdde = SharedSliceMut::<Edge>::abst_mem_mut(gdde_fn, edge_count, mmap > 0)?;
        let alive = SharedSliceMut::<AtomicBool>::abst_mem_mut(a_fn, node_count * 2, mmap > 0)?;
        let coms = SharedSliceMut::<usize>::abst_mem_mut(c_fn, node_count, mmap > 0)?;
        let new_coms = SharedSliceMut::<usize>::abst_mem_mut(nc_fn, node_count, true)?;

        Ok((k, sigma, gdi, gde, gddi, gdde, alive, coms, new_coms))
    }

    #[deprecated]
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

    fn compute(&mut self, mmap: u8) -> Result<(), Box<dyn std::error::Error>> {
        let edge_count = self.graph.width();
        let node_count = self.graph.size() - 1;

        let threads = self.graph.thread_num().max(get_physical());
        let node_load = node_count.div_ceil(threads);

        let (k, sigma, gdi, gde, gddi, gdde, alive, coms, new_comms) =
            self.init_procedural_memory_gve_louvain(mmap)?;

        // C --- the community vector of the original nodes
        let mut communities = self.community.shared_slice();
        let index_ptr = SharedSlice::<usize>::new(self.graph.index_ptr(), node_count + 1);
        let graph_ptr = SharedSlice::<Edge>::new(self.graph.edges_ptr(), edge_count);

        // Initialize
        thread::scope(|scope| {
            for tid in 0..threads {
                // eid is unnecessary as graph + index alwready do the job

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                let alive = alive.shared_slice();
                let mut gdi = gdi.shared_slice();
                let mut gde = gde.shared_slice();
                let mut k = k.shared_slice();
                let mut sigma = sigma.shared_slice();
                let mut coms = coms.shared_slice();

                scope.spawn(move |_| {
                    for u in begin..end {
                        // mark all nodes unprocessed
                        alive.get(u).swap(false, Ordering::Relaxed);
                        *communities.get_mut(u) = u;
                        *coms.get_mut(u) = u;
                        let u_index_start = *index_ptr.get(u);
                        let u_index_end = *index_ptr.get(u + 1);
                        let out_offset = u * 2;
                        *gdi.get_mut(out_offset) = u_index_start;
                        *gdi.get_mut(out_offset + 1) = u_index_end;
                        // copy G onto G'
                        gde.write_shared_slice(
                            graph_ptr,
                            u_index_start,
                            u_index_start,
                            u_index_end - u_index_start,
                        );
                        // sum of edge weights, unweighted edges (w = 1)
                        // Σ′ ← 𝐾′ ← 𝑣𝑒𝑟𝑡𝑒𝑥_w𝑒𝑖𝑔ℎ𝑡𝑠(𝐺′)
                        let deg_u = u_index_end - u_index_start;
                        *k.get_mut(u) = AtomicUsize::new(deg_u);
                        *sigma.get_mut(u) = AtomicUsize::new(deg_u);
                    }
                });
            }
        })
        .unwrap();

        let mut tolerance = Self::INITIAL_TOLERANCE;
        let mut use_first_csr = true; // toggle between G' and G'' buffers for each pass
        let mut prev_comm_count = node_count;
        let mut new_comm_count: usize = 0;
        let mut coms = coms.shared_slice();
        let mut new_coms = new_comms.shared_slice();
        let synchronize = Arc::new(Barrier::new(threads));

        // Outer loop: Louvain passes
        for l_pass in 0..Self::MAX_PASSES {
            // Determine which CSR buffers to use for this pass (alternate between gdi/gde and gddi/gdde)
            let (cur_index, cur_edges, mut next_index, mut next_edges) = if use_first_csr {
                (
                    gdi.shared_slice(),
                    gde.shared_slice(),
                    gddi.shared_slice(),
                    gdde.shared_slice(),
                )
            } else {
                (
                    gddi.shared_slice(),
                    gdde.shared_slice(),
                    gdi.shared_slice(),
                    gde.shared_slice(),
                )
            };

            // li <- louvain_move(G', C', K', Σ′) (adapted so initialization is multithreaded)
            let mut total_delta_q = 0.0f64;
            let mut l_iterations = 0;
            loop {
                if l_iterations == Self::MAX_ITERATIONS {
                    break;
                }
                // Reset iteration metrics
                total_delta_q = thread::scope(|scope| -> Result<f64, Box<dyn std::error::Error>> {
                    // recalculate node load per thread
                    let node_load = prev_comm_count.div_ceil(threads);
                    // Use dynamic style scheduling by splitting range [0, cur_node_count) for threads
                    let mut delta_q_local = Vec::with_capacity(threads);
                    for tid in 0..threads {
                        let start = std::cmp::min(tid * node_load, prev_comm_count);
                        let end = std::cmp::min(start + node_load, prev_comm_count);
                        let k = k.shared_slice().clone();
                        let sigma = sigma.shared_slice().clone();
                        let alive = alive.shared_slice().clone();
                        let synchronize = synchronize.clone();
                        // Each thread gets its own local HashMap for accumulating neighbor community weights
                        delta_q_local.push(scope.spawn(move |_| {
                            // If pass > 0, we need to initialize communities for the new graph (each super-vertex in its own community)
                            if l_pass != 0 && l_iterations == 0 {
                                for u in start..end {
                                    // Σ′ ← 𝐾′ ← 𝑣𝑒𝑟𝑡𝑒𝑥_w𝑒𝑖𝑔ℎ𝑡𝑠(𝐺′)
                                    // (These were computed during aggregation; here we just ensure sigma and k are up-to-date in memory)
                                    let deg_u = sigma.get(u).load(Ordering::Relaxed); // sigma was set to total degree of each new community in prev aggregation
                                    k.get(u).store(deg_u, Ordering::Relaxed);
                                    // 𝐶′ ← [0..|𝑉′|)
                                    *coms.get_mut(u) = u;
                                    // mark all nodes unprocessed
                                    alive.get(u).store(false, Ordering::Relaxed);
                                    *new_coms.get_mut(u) = usize::MAX;
                                }
                            } else if l_iterations == 0 {
                                for u in start..end {
                                    *new_coms.get_mut(u) = usize::MAX;
                                }
                            }
                            synchronize.wait();

                            let mut local_delta_q_sum = 0.0;
                            let mut neighbor_map: HashMap<usize, usize> = HashMap::new();
                            for u in start..end {
                                // Skip this vertex if it was marked as pruned (no need to process)
                                if alive.get(u).swap(true, Ordering::Relaxed) {
                                    continue;
                                }
                                let u_com_old = *coms.get(u);
                                // Calculate total incident weight of u (k[u]) and remove u's contribution from its community
                                let k_u = k.get(u).load(Ordering::Relaxed);
                                let k_u_f64 = k_u as f64;
                                // Decrease the community total weight temporarily (as if u is removed from its community)
                                let sigma_old =
                                    sigma.get(u_com_old).fetch_sub(k_u, Ordering::Relaxed);
                                let sigma_old_f64 = sigma_old as f64;
                                // Gather neighbor communities and edge weight sums using the adjacency list of u
                                neighbor_map.clear();
                                let idx_start = *cur_index.get(u * 2);
                                let idx_end = *cur_index.get(u * 2 + 1);
                                for e_idx in idx_start..idx_end {
                                    let v = cur_edges.get(e_idx).dest();
                                    if v == u {
                                        continue; // ignore self-loop if any
                                    }
                                    let comm_v = *coms.get(v);
                                    // Accumulate weight to neighbor's community
                                    *neighbor_map.entry(comm_v).or_insert(0) += 1;
                                }
                                // Ensure the current community appears in the map (for ΔQ calculation) with weight 0 if not present
                                neighbor_map.entry(u_com_old).or_insert(0);
                                // Determine best community for u (including the possibility of staying put)
                                let mut best_comm = u_com_old;
                                let mut best_gain = 0.0;
                                for (&comm, &edge_weight) in neighbor_map.iter() {
                                    let w_u_to_comm = edge_weight as f64;
                                    if comm == u_com_old {
                                        // For the current community, treat u as removed: effectively K_i->d = w_u_to_comm
                                        continue;
                                    }
                                    // Total weight of community 'comm' (Σ_comm)
                                    let sigma_comm = sigma.get(comm).load(Ordering::Relaxed) as f64;
                                    // Compute ΔQ = (w_u->comm - w_u->oldComm) / m - (K_u * (Σ_comm - Σ_oldComm) / (2m^2))
                                    // Here m (total edge weight) = edge_count/2 for undirected unweighted graph
                                    let m2 = edge_count as f64; // 2m (since each undirected edge counted twice in edge_count)
                                    let delta_q = (w_u_to_comm
                                        - neighbor_map.get(&u_com_old).cloned().unwrap_or(0)
                                            as f64)
                                        / m2
                                        - (k_u_f64 * ((sigma_comm - sigma_old_f64) / (m2 * m2)));
                                    if delta_q > best_gain {
                                        best_gain = delta_q;
                                        best_comm = comm;
                                    }
                                }
                                // Move u to the best community if it yields positive gain
                                if best_comm != u_com_old && best_gain > 1e-12 {
                                    // Atomic updates to community totals
                                    sigma.get(best_comm).fetch_add(k_u, Ordering::Relaxed);
                                    // Update community assignment of u
                                    *coms.get_mut(u) = best_comm;
                                    local_delta_q_sum += best_gain;
                                    // Mark all neighbors of u to be processed in the next iteration (they might be affected by u's move)
                                    for e_idx in idx_start..idx_end {
                                        let v = cur_edges.get(e_idx).dest();
                                        if v != u {
                                            alive.get(v).store(false, Ordering::Relaxed);
                                        }
                                    }
                                } else {
                                    sigma.get(u_com_old).fetch_add(k_u, Ordering::Relaxed);
                                }
                            }
                            // Atomically accumulate thread-local results to global
                            // Use an atomic or mutex to add local_delta_q_sum to total_delta_q (to avoid data races)
                            // Here we choose to use an atomic f64 by converting to an AtomicU64 representation if needed.
                            // For simplicity, we'll use a critical section via a mutex (not shown in snippet) or just sum after joining threads.
                            // (In this simplified snippet, we ignore fine-grained atomic accumulation of delta Q for brevity.)
                            local_delta_q_sum
                        }));
                    }
                    for q in delta_q_local {
                        match q.join() {
                            Ok(local_delta) => total_delta_q += local_delta,
                            Err(e) => {
                                return Err(format!(
                                    "error joining thread, couldn't get delat_q_local: {:?}",
                                    e
                                )
                                .into());
                            }
                        };
                    }
                    Ok(total_delta_q)
                })
                .unwrap()?;
                // End of parallel section for this iteration

                // Check convergence criteria for the local-moving phase
                // (In a fully atomic implementation, total_delta_q would be summed across threads above.)
                // For this outline, assume total_delta_q was accumulated correctly.
                if total_delta_q.abs() < tolerance {
                    // No vertex moved or improvement below tolerance: local-moving phase converged
                    break;
                }
                l_iterations += 1;
            }

            // communities have converged
            if l_iterations <= 1 {
                break;
            }

            // |Γ|, |Γ𝑜𝑙𝑑 | ← Number of communities in 𝐶′, C;
            for u in 0..prev_comm_count {
                let old_c = *coms.get(u);
                let new_c = *new_coms.get(old_c);
                if new_c == usize::MAX {
                    *new_coms.get_mut(old_c) = new_comm_count;
                    *coms.get_mut(u) = new_comm_count;
                    new_comm_count += 1;
                } else {
                    *coms.get_mut(u) = new_c;
                }
            }
            // println!(
            //     " 1 -> #prev comms = {prev_comm_count}\n 2 -> #new comms = {new_comm_count}\n 3 -> #nodes = {node_count}\n"
            // );

            // let mut i = 0;
            // let mut j = 0;
            // let mut z = 0;

            for orig in 0..node_count {
                let old_comm = *communities.get(orig);
                if old_comm < prev_comm_count {
                    // Map old community (super-vertex) to new community ID
                    // println!("dendrogram {orig} -> {}", *new_coms.get_mut(old_comm));
                    *communities.get_mut(orig) = *coms.get(old_comm);
                    // if *coms.get(old_comm) == usize::MAX {
                    //     j += 1;
                    // }
                } /*else {
                if old_comm == usize::MAX {
                z += 1;
                }
                // println!(
                //     "for prev_comm_count {prev_comm_count} -> {old_comm} at node {orig} {}",
                //     self.graph.node_degree(orig)
                // );
                i += 1;
                } */
            }
            // println!(
            //     " 4 -> #nodes no re-assignment = {i}\n 5 -> #nodes bad re-assignment = {j}\n 6 -> #nodes past bad re-assignment = {z}\n\n\n\n", /*AAAAAAAAAAAAAAAA -> {{{l_pass} delta_communities: {}}}",*/
            //                                                                                                                                     // new_comm_count as f64 / prev_comm_count as f64
            // );

            // if |Γ|/|Γ𝑜𝑙𝑑 | > 𝜏_𝑎𝑔𝑔 then break
            if new_comm_count as f64 / prev_comm_count as f64 > Self::AGGREGATION_TOLERANCE {
                // println!(
                //     "Passes stopped at {l_pass} as delta_communities bigger than aggregation tolerance: delta_com {} prev_comm_count {prev_comm_count}",
                //     new_comm_count as f64 / prev_comm_count as f64
                // );
                prev_comm_count = new_comm_count;
                break;
            }

            // Build the aggregated graph (super-vertex graph) in the 'next_index' and 'next_edges' buffers
            // Compute degree of each new community in the super-graph
            for u in 0..new_comm_count {
                *new_coms.get_mut(u) = 0;
            }
            {
                // Iterate over edges of current graph
                for u in 0..prev_comm_count {
                    let u_start = *cur_index.get(u * 2);
                    let u_end = *cur_index.get(u * 2 + 1);
                    for e_idx in u_start..u_end {
                        let v = cur_edges.get(e_idx).dest();
                        let cu = *coms.get(u);
                        let cv = *coms.get(v);
                        // skip internal edges (self-loops in super-graph)
                        if cu == cv {
                            continue;
                        }
                        *new_coms.get_mut(u) += 1;
                    }
                }
            }
            // Fill edges of new graph
            {
                // Convert to prefix offsets for ease of placement
                let mut sum = 0;
                for c in 0..new_comm_count {
                    let community_degree = *new_coms.get_mut(c);
                    *new_coms.get_mut(c) = 0;
                    *next_index.get_mut(c * 2) = sum;
                    sum += community_degree;
                    *next_index.get_mut(c * 2 + 1) = sum;
                }
                // Iterate edges again to populate new edge list
                for u in 0..prev_comm_count {
                    let u_start = *cur_index.get(u * 2);
                    let u_end = *cur_index.get(u * 2 + 1);
                    let cu = *coms.get(u) as u64;
                    for e_idx in u_start..u_end {
                        let v = cur_edges.get(e_idx).dest();
                        let cv = *coms.get(v) as u64;
                        if cu == cv {
                            continue;
                        }
                        // Write edge for cu -> cv in new graph
                        let pos = *next_index.get((cu * 2) as usize) + *new_coms.get(cu as usize);
                        *next_edges.get_mut(pos) = Edge::new(cu, cv);
                        *new_coms.get_mut(cu as usize) += 1;
                    }
                }
            }
            // edge weights (k --- super-nodes, sigma --- communities)
            for c in 0..new_comm_count {
                let deg_c = *next_index.get(c * 2) - *next_index.get(c * 2 + 1);
                k.get(c).store(deg_c, Ordering::Relaxed);
                sigma.get(c).store(deg_c, Ordering::Relaxed);
            }

            // Decide whether to continue to next pass
            use_first_csr = !use_first_csr; // swap CSR buffers for next iteration
            tolerance /= Self::TOLERANCE_DROP; // tighten tolerance for next pass (threshold scaling)
            // Check aggregation tolerance: if communities did not reduce significantly, stop
            if new_comm_count as f64 >= (prev_comm_count as f64 * Self::AGGREGATION_TOLERANCE) {
                prev_comm_count = new_comm_count;
                break;
            }
            prev_comm_count = new_comm_count;
            if new_comm_count <= 1 {
                // Only one community remains, algorithm has fully converged
                break;
            }
            new_comm_count = 0;
        } // end of passes loop

        self.community_count = prev_comm_count;

        let m2 = self.graph.width();
        if m2 == 0 {
            self.modularity = 0.0;
            return Ok(());
        }

        for u in 0..node_count {
            k.get(*communities.get(u)).store(0, Ordering::Relaxed);
        }

        // count internal edges (community weight already in k)
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

        // 3) modularity: sum_c( internal_dir_edges[c]/m2 - (Kc[c]/m2)^2 )
        let m2f = m2 as f64;
        let mut partition_modularity = 0.0f64;
        for c in 0..prev_comm_count {
            let lc_over_m2 = (sigma.get(c).load(Ordering::Relaxed) as f64) / m2f;
            let kc_over_m2 = (k.get(c).load(Ordering::Relaxed) as f64) / m2f;
            partition_modularity += lc_over_m2 - kc_over_m2 * kc_over_m2;
        }

        self.modularity = partition_modularity;

        cleanup_cache()?;
        Ok(())
    }
}
