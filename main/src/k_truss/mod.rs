pub mod burkhardt_et_al;
pub mod clustering_coefficient;
pub mod pkt;

#[cfg(test)]
pub(self) use _verify::verify_k_trusses;

#[cfg(test)]
mod _verify {
    use crate::{
        generic_edge::{GenericEdge, GenericEdgeType},
        generic_memory_map::GraphMemoryMap,
        shared_slice::{AbstractedProceduralMemoryMut, SharedSlice, SharedSliceMut},
        utils::{FileType, cache_file_name},
    };

    use crossbeam::thread;
    use num_cpus::get_physical;
    use std::{
        collections::HashMap,
        sync::{
            Arc, Barrier,
            atomic::{AtomicU8, Ordering},
        },
    };

    #[allow(dead_code)]
    pub(super) fn verify_k_trusses<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>(
        graph: &GraphMemoryMap<EdgeType, Edge>,
        edge_trussness: AbstractedProceduralMemoryMut<u8>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads = get_physical() * 2;
        let node_count = graph.size() - 1;
        let edge_count = graph.width();
        let node_load = node_count.div_ceil(threads);
        // trussness length == graph width
        if edge_trussness.len() != graph.width() {
            return Err(format!(
                "k-trusses invalid: graph width is {} but trussness has length {}",
                graph.width(),
                edge_trussness.len()
            )
            .into());
        }
        if graph.width() == 0 {
            return Ok(());
        }

        let trussness = edge_trussness.shared_slice();

        let index_ptr = SharedSlice::<usize>::new(graph.index_ptr(), node_count + 1);
        let graph_ptr = SharedSlice::<Edge>::new(graph.edges_ptr(), edge_count);

        let support = SharedSliceMut::<AtomicU8>::abst_mem_mut(
            cache_file_name("".to_string(), FileType::Test, None)?,
            edge_count,
            true,
        )?;
        let edge_reciprocal = graph.get_edge_reciprocal()?;
        let edge_out = graph.get_edge_dest_id_over_source()?;

        let synchronize = Arc::new(Barrier::new(threads));

        let count_support =
            |u: usize, v: usize, tau: u8| -> Result<u8, Box<dyn std::error::Error>> {
                let mut n_u = graph.neighbours(u)?.enumerate();
                let mut n_v = graph.neighbours(v)?.enumerate();
                let u_offset = *index_ptr.get(u);
                let v_offset = *index_ptr.get(v);
                let mut u_w = n_u.next();
                let mut v_w = n_v.next();
                let mut cnt = 0u8;

                while u_w.is_some() && v_w.is_some() {
                    let (u_w_idx, u_w_edge) = u_w.unwrap();
                    let (v_w_idx, v_w_edge) = v_w.unwrap();
                    match u_w_edge.dest().cmp(&v_w_edge.dest()) {
                        std::cmp::Ordering::Less => u_w = n_u.next(),
                        std::cmp::Ordering::Greater => v_w = n_v.next(),
                        std::cmp::Ordering::Equal => {
                            let w = u_w_edge.dest();
                            if w != u
                                && w != v
                                && *trussness.get(u_offset + u_w_idx) >= tau
                                && *trussness.get(v_offset + v_w_idx) >= tau
                            {
                                cnt += 1;
                            }
                            u_w = n_u.next();
                            v_w = n_v.next();
                        }
                    }
                }
                Ok(cnt)
            };

        thread::scope(
        |scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut res = vec![];
            for tid in 0..threads {
                let eo = edge_out.shared_slice();
                let er = edge_reciprocal.shared_slice();

                let mut tris = support.shared_slice();

                let synchronize = Arc::clone(&synchronize);

                let start = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(start + node_load, node_count);

                res.push(scope.spawn(move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                    let edge_begin = *index_ptr.get(start);
                    let edge_end = *index_ptr.get(end);

                    // initialize tris with zeroes
                    for idx in edge_begin..edge_end {
                        *tris.get_mut(idx) = AtomicU8::new(0);
                    }

                    synchronize.wait();

                    let mut neighbours = HashMap::<usize, usize>::new();

                    // compute edge support (stored in tris)
                    for u in start..end {
                        for j in *eo.get(u)..*index_ptr.get(u + 1) {
                            let w = *graph_ptr.get(j);
                            neighbours.insert(w.dest(), j);
                        }
                        for u_v in *index_ptr.get(u)..*eo.get(u) {
                            let v = *graph_ptr.get(u_v);
                            let v = v.dest();
                            if u == v {
                                continue;
                            }
                            for v_w in (*eo.get(v)..*index_ptr.get(v + 1)).rev() {
                                let w = graph_ptr.get(v_w).dest();
                                if w <= u {
                                    break;
                                }
                                let w_u = match neighbours.get(&w) {
                                    Some(i) => *i,
                                    None => continue,
                                };

                                tris.get(u_v).fetch_add(1, Ordering::Relaxed);
                                tris.get(v_w).fetch_add(1, Ordering::Relaxed);
                                tris.get(w_u).fetch_add(1, Ordering::Relaxed);
                                tris.get(*er.get(u_v)).fetch_add(1, Ordering::Relaxed);
                                tris.get(*er.get(v_w)).fetch_add(1, Ordering::Relaxed);
                                tris.get(*er.get(w_u)).fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        neighbours.clear();
                    }

                    synchronize.wait();

                    for u in start..end {
                        for edge_idx in *eo.get(u)..*index_ptr.get(u + 1) {
                            let reciprocal_idx = *er.get(edge_idx);

                            let k_uv = *trussness.get(edge_idx);
                            let k_vu = *trussness.get(reciprocal_idx);

                            // test for trussness symmetry
                            if k_uv != k_vu {
                                return Err(format!(
                                    "k-trusses invalid: for node {u}: t(u->v)={k_uv}, t(v->u)={k_vu}"
                                ).into());
                            }
                            let k = k_uv;

                            let v = graph_ptr.get(edge_idx).dest();
                            // test membership: edges with trussness == k must have support > k + 2
                            if k > 0 {
                                let support_k = count_support(u, v, k).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {format!("{e}").into()}
                                    )?;
                                // check if hasenough support to be in k-truss
                                if support_k < k {
                                    return Err(format!(
                                        "Edge ({u},{v}) fails {k}-truss membership: support_≥{k} = {support_k} < {k}"
                                    ).into());
                                }
                            }

                            // test maximality: support within edges with trussness >= k+1 must be < (k + 1)
                            let kp1 = k.saturating_add(1);
                            let support_k1 = count_support(u, v, kp1).map_err(
                                |e| -> Box<dyn std::error::Error + Send + Sync> {format!("{e}").into()}
                                )?;
                            // check if hasenough support to be in (k+1)-truss
                            if support_k1 >= kp1 {
                                return Err(format!(
                                    "Edge ({u},{v}) fails {k}-truss maximality: support_≥{kp1} = {support_k1} (should be < {kp1})",
                                ).into());
                            }
                        }
                    }
                    Ok(())
                }));
            }
            // join results
            for r in res {
                let joined_r = match r.join() {
                    Ok(b) => b,
                    Err(e) => {
                        return Err(format!("{:?}", e).into());
                    }
                };
                joined_r.map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;
            }
            Ok(())
        },
    ).map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
    }
}
