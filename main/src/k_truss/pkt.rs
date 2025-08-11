use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use crossbeam::thread;
use num_cpus::get_physical;
use std::{
    io::Error,
    sync::{
        Arc, Barrier,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
};

type ProceduralMemoryPKT = (
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<AtomicU8>,
);

#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoPKT<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    /// memmapped slice containing the coreness of each edge
    k_trusses: AbstractedProceduralMemoryMut<u8>,
}
#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> AlgoPKT<'a, EdgeType, Edge> {
    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let output_filename = cache_file_name(graph.cache_fst_filename(), FileType::KTruss, None)?;
        let k_trusses =
            SharedSliceMut::<u8>::abst_mem_mut(output_filename.clone(), graph.width(), true)?;
        let pkt = Self { graph, k_trusses };
        pkt.compute(10)?;
        Ok(pkt)
    }

    fn init_procedural_memory_pkt(
        &self,
        mmap: u8,
    ) -> Result<ProceduralMemoryPKT, Box<dyn std::error::Error>> {
        let edge_count = self.graph.width();

        let template_fn = self.graph.cache_fst_filename();
        let c_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(1))?;
        let n_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(2))?;
        let p_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(3))?;
        let ic_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(4))?;
        let in_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(5))?;
        let s_fn = cache_file_name(template_fn.clone(), FileType::KTruss, None)?;

        let curr = SharedSliceMut::<usize>::abst_mem_mut(c_fn, edge_count, mmap > 2)?;
        let next = SharedSliceMut::<usize>::abst_mem_mut(n_fn, edge_count, mmap > 2)?;
        let processed = SharedSliceMut::<bool>::abst_mem_mut(p_fn, edge_count, mmap > 3)?;
        let in_curr = SharedSliceMut::<bool>::abst_mem_mut(ic_fn, edge_count, mmap > 3)?;
        let in_next = SharedSliceMut::<bool>::abst_mem_mut(in_fn, edge_count, mmap > 3)?;
        let s = SharedSliceMut::<AtomicU8>::abst_mem_mut(s_fn, edge_count, true)?;

        Ok((curr, next, processed, in_curr, in_next, s))
    }

    pub fn compute(&self, mmap: u8) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1;
        let edge_count = self.graph.width();

        let threads = self.graph.thread_num().max(get_physical());
        let edge_load = edge_count.div_ceil(threads);
        let node_load = node_count.div_ceil(threads);

        let index_ptr = Arc::new(SharedSlice::<usize>::new(
            self.graph.index_ptr(),
            node_count + 1,
        ));
        let graph_ptr = Arc::new(SharedSlice::<Edge>::new(self.graph.edges_ptr(), edge_count));

        // Shared arrays
        let (curr, next, processed, in_curr, in_next, s) = self.init_procedural_memory_pkt(mmap)?;
        let edge_reciprocal = self.graph.get_edge_reciprocal()?;
        let edge_out = self.graph.get_edge_dest_id_over_source()?;

        // Allocate memory for thread local arrays
        let template_fn = self.graph.cache_fst_filename();
        let mut x: Vec<AbstractedProceduralMemoryMut<usize>> = Vec::new();
        for i in 0..threads {
            let x_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(8 + i))?;
            x.push(SharedSliceMut::<usize>::abst_mem_mut(
                x_fn,
                node_count,
                mmap > 0,
            )?)
        }
        let x = Arc::new(x);

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        // ParTriangle-AM4
        thread::scope(|scope| {
            for tid in 0..threads {
                // eid is unnecessary as graph + index alwready do the job
                let graph_ptr = Arc::clone(&graph_ptr);
                let index_ptr = Arc::clone(&index_ptr);
                let mut x = x[tid].shared_slice();
                let eo = edge_out.shared_slice();
                let mut s = s.shared_slice();
                let er = edge_reciprocal.shared_slice();
                let mut curr = curr.shared_slice();
                let mut next = next.shared_slice();
                let mut in_curr = in_curr.shared_slice();
                let mut in_next = in_next.shared_slice();
                let mut processed = processed.shared_slice();
                let synchronize = Arc::clone(&synchronize);

                let init_begin = std::cmp::min(tid * edge_load, edge_count);
                let init_end = std::cmp::min(init_begin + edge_load, edge_count);
                let begin = std::cmp::min(tid * node_load, node_count - 1); // is node and edge
                let end = std::cmp::min(begin + node_load, node_count - 1); // limit accurate?
                scope.spawn(move |_| {
                    // initialize s, edge_out, x, curr, next, in_curr, in_next & processed
                    for edge_offset in init_begin..init_end {
                        *s.get_mut(edge_offset) = AtomicU8::new(0);
                        *curr.get_mut(edge_offset) = 0;
                        *next.get_mut(edge_offset) = 0;
                        *in_curr.get_mut(edge_offset) = false;
                        *in_next.get_mut(edge_offset) = false;
                        *processed.get_mut(edge_offset) = false;
                        *x.get_mut(graph_ptr.get(edge_offset).dest()) = 0;
                    }

                    synchronize.wait();

                    for u in begin..end {
                        let edges_stop = *index_ptr.get(u + 1);
                        let eo_u = *eo.get(u);
                        for j in eo_u..edges_stop {
                            *x.get_mut(graph_ptr.get(j).dest()) = j + 1;
                        }
                        for u_v in *index_ptr.get(u)..eo_u {
                            let v = graph_ptr.get(u_v).dest();
                            if v == u {
                                break;
                            }
                            for v_w in (*eo.get(v)..*index_ptr.get(v + 1)).rev() {
                                let w = graph_ptr.get(v_w).dest();
                                if w <= u {
                                    break;
                                }

                                let w_u = match x.get(w).cmp(&0) {
                                    std::cmp::Ordering::Equal => continue,
                                    _ => *x.get(w) - 1,
                                };

                                s.get(u_v).fetch_add(1, Ordering::Relaxed);
                                s.get(v_w).fetch_add(1, Ordering::Relaxed);
                                s.get(w_u).fetch_add(1, Ordering::Relaxed);
                                s.get(*er.get(u_v)).fetch_add(1, Ordering::Relaxed);
                                s.get(*er.get(v_w)).fetch_add(1, Ordering::Relaxed);
                                s.get(*er.get(w_u)).fetch_add(1, Ordering::Relaxed);
                            }
                        }

                        for j in eo_u..edges_stop {
                            *x.get_mut(graph_ptr.get(j).dest()) = 0;
                        }
                    }
                });
            }
        })
        .unwrap();

        let mut l: u8 = 1;
        let buff_size = 4096;
        let total_duds = Arc::new(AtomicUsize::new(0));
        let curr = SharedQueueMut::from_shared_slice(curr.shared_slice());
        let next = SharedQueueMut::from_shared_slice(next.shared_slice());

        thread::scope(|scope| {
            let mut res = Vec::new();
            for tid in 0..threads {
                let graph_ptr = Arc::clone(&graph_ptr);
                let index_ptr = Arc::clone(&index_ptr);

                let mut todo = edge_count;
                let mut x = x[tid].shared_slice();

                let s = s.shared_slice();
                let mut curr = curr.clone();
                let mut next = next.clone();
                let er = edge_reciprocal.shared_slice();
                let mut in_curr = in_curr.shared_slice();
                let mut in_next = in_next.shared_slice();
                let mut processed = processed.shared_slice();

                let total_duds = Arc::clone(&total_duds);
                let synchronize = Arc::clone(&synchronize);

                let begin = std::cmp::min(tid * edge_load, edge_count);
                let end = std::cmp::min(begin + edge_load, edge_count);

                res.push(scope.spawn(move |_| -> Result<Vec<u64>, Box<dyn std::error::Error + Send + Sync>> {
                    let mut res = vec![0_u64; 20];
                    let mut buff = vec![0; buff_size];
                    let mut i = 0;

                    // Remove 0-triangle edges
                    for e in begin..end {
                        if s.get(e).load(Ordering::Relaxed) == 0 {
                            *processed.get_mut(e) = true;
                            res[0] += 1;
                            i += 1;
                        }
                    }
                    total_duds.fetch_add(i, Ordering::SeqCst);
                    i = 0;

                    synchronize.wait();

                    todo = match todo.overflowing_sub(total_duds.load(Ordering::Relaxed)) {
                        (r, false) => r,
                        _ => {
                            return Err(Box::new(Error::new(
                                        std::io::ErrorKind::Other,
                                        format!("error overflow when decrementing todo ({todo} - {})", total_duds.load(Ordering::Relaxed))))
                                );
                        }
                    };

                    // println!("triangles removed");
                    while todo > 0 {
                        for e in begin..end {
                            if s.get(e).load(Ordering::Relaxed) == l {
                                buff[i] = e;
                                *in_curr.get_mut(e) = true;
                                i += 1;
                            }
                            if i == buff_size {
                                curr.push_slice(buff.as_slice());
                                i = 0;
                            }
                        }
                        if i > 0 {
                            curr.push_slice(&buff[0..i]);
                            i = 0;
                        }
                        synchronize.wait();

                        let mut to_process = match curr.slice(0, curr.len()) {
                            Some(i) => i,
                            None => {
                                return Err(Box::new(Error::new(
                                            std::io::ErrorKind::Other,
                                            "error reading curr in pkt"
                                            )));
                            }
                        };
                        // println!("new cicle initialized {} {:?}", todo, curr.ptr);
                        while !to_process.is_empty() {
                            todo = match todo.overflowing_sub(to_process.len()) {
                                (r, false) => r,
                                _ => {
                                    return Err(Box::new(Error::new(
                                                std::io::ErrorKind::Other,
                                                format!("error overflow when decrementing todo ({todo} - {})", to_process.len())))
                                        );
                                }
                            };
                            synchronize.wait();

                            // ProcessSubLevel
                            let thread_load = curr.len().div_ceil(threads);
                            let begin = tid * thread_load;
                            let end = std::cmp::min(begin + thread_load, curr.len());

                            for e_idx in begin..end {
                                let u_v = *to_process.get(e_idx);

                                let u = graph_ptr.get(*er.get(u_v)).dest();
                                let v = graph_ptr.get(u_v).dest();

                                let edges_start = *index_ptr.get(u);
                                let edges_stop = *index_ptr.get(u + 1);

                                // mark u neighbours
                                for u_w in edges_start..edges_stop {
                                    let w = graph_ptr.get(u_w).dest();
                                    if w != u {
                                        *x.get_mut(w) = *er.get(u_w) + 1;
                                    }
                                }

                                for v_w in *index_ptr.get(v)..*index_ptr.get(v + 1) {
                                    let w = graph_ptr.get(v_w).dest();
                                    if *x.get(w) == 0 {
                                        continue;
                                    }
                                    let w_u = *x.get(w) - 1;
                                    if *processed.get(v_w) || *processed.get(w_u) {
                                        continue;
                                    }

                                    if s.get(v_w).load(Ordering::Relaxed) > l
                                        && s.get(w_u).load(Ordering::Relaxed) > l
                                    {
                                        let prev_l_v_w = s.get(v_w).fetch_sub(1, Ordering::SeqCst);
                                        if prev_l_v_w == l + 1 {
                                            *in_next.get_mut(v_w) = true;
                                            buff[i] = v_w;
                                            i += 1;
                                            if i == buff_size {
                                                next.push_slice(&buff[..]);
                                                i = 0;
                                            }
                                        }
                                        if prev_l_v_w <= l {
                                            s.get(v_w).fetch_add(1, Ordering::SeqCst);
                                        }
                                        let prev_l_w_u = s.get(w_u).fetch_sub(1, Ordering::SeqCst);
                                        if prev_l_w_u == l + 1 {
                                            *in_next.get_mut(w_u) = true;
                                            buff[i] = w_u;
                                            i += 1;
                                            if i == buff_size {
                                                next.push_slice(&buff[..]);
                                                i = 0;
                                            }
                                        }
                                        if prev_l_w_u <= l {
                                            s.get(w_u).fetch_add(1, Ordering::SeqCst);
                                        }
                                    } else if s.get(v_w).load(Ordering::Relaxed) > l
                                        && ((u_v < w_u && *in_curr.get(w_u)) || !*in_curr.get(w_u))
                                    {
                                        let prev_l_v_w = s.get(v_w).fetch_sub(1, Ordering::SeqCst);
                                        if prev_l_v_w == l + 1 {
                                            *in_next.get_mut(v_w) = true;
                                            buff[i] = v_w;
                                            i += 1;
                                            if i == buff_size {
                                                next.push_slice(&buff[..]);
                                                i = 0;
                                            }
                                        }
                                        if prev_l_v_w <= l {
                                            s.get(v_w).fetch_add(1, Ordering::SeqCst);
                                        }
                                    } else if s.get(w_u).load(Ordering::Relaxed) > l
                                        && ((u_v < v_w && *in_curr.get(v_w)) || !*in_curr.get(v_w))
                                    {
                                        let prev_l_w_u = s.get(w_u).fetch_sub(1, Ordering::SeqCst);
                                        if prev_l_w_u == l + 1 {
                                            *in_next.get_mut(w_u) = true;
                                            buff[i] = w_u;
                                            i += 1;
                                            if i == buff_size {
                                                next.push_slice(&buff[..]);
                                                i = 0;
                                            }
                                        }
                                        if prev_l_w_u <= l {
                                            s.get(w_u).fetch_add(1, Ordering::SeqCst);
                                        }
                                    }
                                }

                                // unmark u neighbours
                                for u_w in edges_start..edges_stop {
                                    *x.get_mut(graph_ptr.get(u_w).dest()) = 0;
                                }
                            }
                            if i > 0 {
                                next.push_slice(&buff[0..i]);
                                i = 0;
                            }
                            for e_idx in begin..end {
                                let edge = *to_process.get(e_idx);
                                *processed.get_mut(edge) = true;
                                *in_curr.get_mut(edge) = false; // FIXME: this can be removed?
                            }
                            // println
                            for _e in begin..end {
                                res[l as usize] += 1;
                            }

                            synchronize.wait();
                            next = std::mem::replace(&mut curr, next).clear();
                            in_next = std::mem::replace(&mut in_curr, in_next);

                            synchronize.wait();
                            to_process = match curr.slice(0, curr.len()) {
                                Some(i) => i,
                                None => {
                                    return Err(Box::new(Error::new(
                                                std::io::ErrorKind::Other,
                                                "error couldn't get new to_process vec"))
                                        );
                                }
                            };
                            synchronize.wait();
                        }
                        l = match l.overflowing_add(1) {
                            (r, false) => r,
                            _ => {
                                return Err(Box::new(Error::new(
                                        std::io::ErrorKind::Other,
                                        format!("error overflow when adding to l ({l} - 1)")))
                                    );
                            }
                        };
                        synchronize.wait();
                    }
                    Ok(res)
                }));
            }
            let joined_res: Vec<Vec<u64>> = res
                .into_iter()
                .map(|v| v.join().expect("error thread panicked").expect("error ??1"))
                .collect();
            let mut r = vec![0u64; 16];
            for i in 0..16 {
                for v in joined_res.clone() {
                    r[i] += v[i];
                }
            }
            println!("k-trussness {:?}", r);
        })
        .unwrap();

        Ok(())
    }
}
