use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use crossbeam::thread;
use num_cpus::get_physical;
use std::{
    collections::HashMap,
    sync::{
        Arc, Barrier,
        atomic::{AtomicU8, Ordering},
    },
};

type ProceduralMemoryBurkhardtEtAl = (
    AbstractedProceduralMemoryMut<AtomicU8>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<(usize, usize)>,
);

#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoBurkhardtEtAl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    /// memmapped slice containing the coreness of each edge
    k_trusses: AbstractedProceduralMemoryMut<u8>,
}
#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>
    AlgoBurkhardtEtAl<'a, EdgeType, Edge>
{
    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let output_filename = cache_file_name(graph.cache_fst_filename(), FileType::KTruss, None)?;
        let k_trusses =
            SharedSliceMut::<u8>::abst_mem_mut(output_filename.clone(), graph.width(), true)?;
        let burkhardt_et_al = Self { graph, k_trusses };
        burkhardt_et_al.compute(10)?;
        Ok(burkhardt_et_al)
    }

    fn init_procedural_memory_burkhardt_et_al(
        &self,
        mmap: u8,
    ) -> Result<ProceduralMemoryBurkhardtEtAl, Box<dyn std::error::Error>> {
        let edge_count = self.graph.width();

        let template_fn = self.graph.cache_index_filename();
        let t_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(0))?;
        let el_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(1))?;
        let ei_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(2))?;
        let s_fn = cache_file_name(template_fn.clone(), FileType::KTruss, Some(3))?;

        let tri_count = SharedSliceMut::<AtomicU8>::abst_mem_mut(t_fn, edge_count, mmap > 0)?;
        let edge_list = SharedSliceMut::<usize>::abst_mem_mut(el_fn, edge_count, mmap > 1)?;
        let edge_index = SharedSliceMut::<usize>::abst_mem_mut(ei_fn, edge_count, mmap > 1)?;
        let stack = SharedSliceMut::<(usize, usize)>::abst_mem_mut(s_fn, edge_count * 2, mmap > 2)?;

        Ok((tri_count, edge_list, edge_index, stack))
    }

    pub fn compute(&self, mmap: u8) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1;
        let edge_count = self.graph.width();

        let threads = self.graph.thread_num().max(get_physical());
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = SharedSlice::<usize>::new(self.graph.index_ptr(), node_count + 1);
        let graph_ptr = SharedSlice::<Edge>::new(self.graph.edges_ptr(), edge_count);

        // Shared atomic & simple arrays for counts and trussness
        let (triangle_count, edges, edge_index, edge_stack) =
            self.init_procedural_memory_burkhardt_et_al(mmap)?;
        let mut trussness = self.k_trusses.shared_slice();

        let edge_reciprocal = self.graph.get_edge_reciprocal()?;
        let edge_out = self.graph.get_edge_dest_id_over_source()?;

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        // Algorithm 1 - adjusted for directed scheme
        thread::scope(|scope| {
            for tid in 0..threads {
                let eo = edge_out.shared_slice();
                let er = edge_reciprocal.shared_slice();

                let mut tris = triangle_count.shared_slice();
                let mut edges = edges.shared_slice();
                let mut edge_index = edge_index.shared_slice();

                let synchronize = Arc::clone(&synchronize);

                let start = std::cmp::min(tid * thread_load, node_count);
                let end = std::cmp::min(start + thread_load, node_count);

                scope.spawn(move |_| {
                    // initialize triangle_count with zeroes
                    let edge_begin = *index_ptr.get(start);
                    let edge_end = *index_ptr.get(end);
                    for idx in edge_begin..edge_end {
                        *trussness.get_mut(idx) = 0;
                        *tris.get_mut(idx) = AtomicU8::new(0);
                    }

                    synchronize.wait();

                    let mut neighbours = HashMap::<usize, usize>::new();
                    for u in start..end {
                        for j in *eo.get(u)..*index_ptr.get(u + 1) {
                            let w = *graph_ptr.get(j);
                            *edges.get_mut(j) = j;
                            *edge_index.get_mut(j) = j;
                            neighbours.insert(w.dest(), j);
                        }
                        for u_v in *index_ptr.get(u)..*eo.get(u) {
                            *edges.get_mut(u_v) = u_v;
                            *edge_index.get_mut(u_v) = u_v;
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
                });
            }
        })
        .unwrap();

        let stack = SharedQueueMut::<(usize, usize)>::from_shared_slice(edge_stack.shared_slice());

        // Algorithm 2 - sentinel value is 0
        // blank (u64, usize) value
        let mut edges = edges.shared_slice();
        let mut edge_index = edge_index.shared_slice();
        let mut edge_count = edge_count;
        let er = edge_reciprocal.shared_slice();
        let tris = triangle_count.shared_slice();
        let mut stack = stack.clone();
        let mut test = vec![0usize; u8::MAX as usize];

        // max node degree is 16
        for k in 1..u8::MAX {
            if edge_count == 0 {
                break;
            }
            let mut idx = 0;
            while idx < edge_count {
                let edge_offset = *edge_index.get(idx);
                let t_count = tris.get(edge_offset).load(Ordering::Relaxed);
                if t_count == k {
                    let u = graph_ptr.get(*er.get(edge_offset)).dest();
                    stack.push((u, edge_offset));
                    idx = match idx.overflowing_add(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!("error overflow add to idx ({idx} + 1)",).into());
                        }
                    };
                } else if t_count == 0 {
                    edge_count = match edge_count.overflowing_sub(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!(
                                "error overflow add to edge_count ({edge_count} + 1)",
                            )
                            .into());
                        }
                    };
                    let e_index = *edge_index.get(edge_count);
                    let r_index = *edges.get(edge_offset);
                    *edge_index.get_mut(edge_count) = *edge_index.get(r_index);
                    *edge_index.get_mut(r_index) = e_index;
                    *edges.get_mut(edge_offset) = edge_count;
                    *edges.get_mut(e_index) = r_index;
                    // store edge trussness
                    *trussness.get_mut(edge_offset) = k - 1;
                    test[k as usize - 1] += 1;
                    continue;
                } else {
                    idx = match idx.overflowing_add(1) {
                        (r, false) => r,
                        _ => {
                            return Err(format!("error overflow add to idx ({idx} + 1)",).into());
                        }
                    };
                }
            }

            let mut neighbours = HashMap::<usize, usize>::new();

            while let Some((u, offset)) = stack.pop() {
                tris.get(offset).store(0, Ordering::Relaxed);
                let v = graph_ptr.get(offset).dest();
                for u_w in *index_ptr.get(u)..*index_ptr.get(u + 1) {
                    let w = graph_ptr.get(u_w).dest();
                    if w != u && w != v {
                        neighbours.insert(w, u_w);
                    }
                }
                for v_w in *index_ptr.get(v)..*index_ptr.get(v + 1) {
                    let w = graph_ptr.get(v_w).dest();
                    if w == v {
                        continue;
                    }
                    let u_w = match neighbours.get(&w) {
                        Some(i) => *i,
                        None => continue,
                    };

                    let w_u = *er.get(u_w);
                    if tris.get(v_w).load(Ordering::Relaxed) != 0
                        && tris.get(w_u).load(Ordering::Relaxed) != 0
                    {
                        let prev_w_u = tris.get(w_u).fetch_sub(1, Ordering::Relaxed);
                        let prev_v_w = tris.get(v_w).fetch_sub(1, Ordering::Relaxed);
                        if prev_w_u == k + 1 {
                            stack.push((w, w_u));
                        }
                        if prev_v_w == k + 1 {
                            stack.push((v, v_w));
                        }
                    }
                }
                neighbours.clear();
            }
        }
        {
            let mut max = 0;
            test.iter().enumerate().for_each(|(i, v)| {
                if *v != 0 && i > max {
                    max = i;
                }
            });
            test.resize(max + 1, 0);
            println!("k-trussness {:?}", test);
        }
        self.k_trusses.flush_async()?;

        Ok(())
    }
}
