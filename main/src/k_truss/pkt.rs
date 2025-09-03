use crate::graph::*;
use crate::shared_slice::*;

use crossbeam::thread;
use portable_atomic::{AtomicU8, AtomicUsize, Ordering};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Barrier};

type ProceduralMemoryPKT = (
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<usize>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<bool>,
    AbstractedProceduralMemoryMut<AtomicU8>,
);

/// For the computation of a [`GraphMemoryMap`] instance's k-truss decomposition as described in ["Shared-memory Graph Truss Decomposition"](https://doi.org/10.48550/arXiv.1707.02000) by Kamir H. and Madduri K.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct AlgoPKT<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    /// Graph for which edge trussness is computed.
    g: &'a GraphMemoryMap<EdgeType, Edge>,
    /// Memmapped slice containing the trussness of each edge.
    k_trusses: AbstractedProceduralMemoryMut<u8>,
}
#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> AlgoPKT<'a, EdgeType, Edge> {
    /// Performs the *PKT Alrgorithm*'s 'k-truss decomposition as described in ["Shared-memory Graph Truss Decomposition"](https://doi.org/10.48550/arXiv.1707.02000) by Kamir H. and Madduri K.
    ///
    /// # Arguments
    ///
    /// * `g` --- the  [`GraphMemoryMap`] instance for which k-truss decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<EdgeType, Edge>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut pkt = Self::new_no_compute(g)?;
        let proc_mem = pkt.init_cache_mem()?;

        pkt.compute_with_proc_mem(proc_mem)?;

        Ok(pkt)
    }

    /// Returns the trussness of a given edge of a [`GraphMemoryMap`] instance.
    ///
    /// # Arguments
    ///
    /// * `e_idx` --- the index of the edge whose trussness is to be returned.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn trussness(&self, e_idx: usize) -> u8 {
        assert!(e_idx < self.g.width());
        *self.k_trusses.get(e_idx)
    }

    /// Returns a slice containing the trussness of each edge of the [`GraphMemoryMap`] instance.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn k_trusses(&self) -> &[u8] {
        self.k_trusses.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::KTrussPKT, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> AlgoPKT<'a, EdgeType, Edge> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(&self) -> Result<ProceduralMemoryPKT, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<ProceduralMemoryPKT, Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryPKT,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: ProceduralMemoryPKT,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.width()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::KTrussPKT, None)?;
        let k_trusses = SharedSliceMut::<u8>::abst_mem_mut(&out_fn, g.width(), true)?;
        Ok(Self { g, k_trusses })
    }

    fn init_cache_mem_impl(&self) -> Result<ProceduralMemoryPKT, Box<dyn std::error::Error>> {
        let edge_count = self.g.width();

        let c_fn = self.g.build_cache_filename(CacheFile::KTrussPKT, Some(1))?;
        let n_fn = self.g.build_cache_filename(CacheFile::KTrussPKT, Some(2))?;
        let p_fn = self.g.build_cache_filename(CacheFile::KTrussPKT, Some(3))?;
        let ic_fn = self.g.build_cache_filename(CacheFile::KTrussPKT, Some(4))?;
        let in_fn = self.g.build_cache_filename(CacheFile::KTrussPKT, Some(5))?;
        let s_fn = self.g.build_cache_filename(CacheFile::KTrussPKT, None)?;

        let curr = SharedSliceMut::<usize>::abst_mem_mut(&c_fn, edge_count, true)?;
        let next = SharedSliceMut::<usize>::abst_mem_mut(&n_fn, edge_count, true)?;
        let processed = SharedSliceMut::<bool>::abst_mem_mut(&p_fn, edge_count, true)?;
        let in_curr = SharedSliceMut::<bool>::abst_mem_mut(&ic_fn, edge_count, true)?;
        let in_next = SharedSliceMut::<bool>::abst_mem_mut(&in_fn, edge_count, true)?;
        let s = SharedSliceMut::<AtomicU8>::abst_mem_mut(&s_fn, edge_count, true)?;

        // pre-initialize the memmapped files if they don't exist
        let _edge_reciprocal = self.g.edge_reciprocal()?;
        let _edge_out = self.g.edge_over()?;

        Ok((curr, next, processed, in_curr, in_next, s))
    }

    fn compute_with_proc_mem_impl(
        &self,
        proc_mem: ProceduralMemoryPKT,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let edge_count = self.g.width();

        let threads = self.g.thread_num();
        let edge_load = edge_count.div_ceil(threads);

        let index_ptr = SharedSlice::<usize>::new(self.g.index_ptr(), self.g.offsets_size());
        let graph_ptr = SharedSlice::<Edge>::new(self.g.edges_ptr(), edge_count);

        // Shared arrays
        let (curr, next, processed, in_curr, in_next, s) = proc_mem;
        let edge_reciprocal = self.g.edge_reciprocal()?;
        let edge_out = self.g.edge_over()?;

        // Allocate memory for thread local arrays
        let mut x: Vec<AbstractedProceduralMemoryMut<usize>> = Vec::new();
        for i in 0..threads {
            let x_fn = self
                .g
                .build_cache_filename(CacheFile::KTrussPKT, Some(8 + i))?;
            x.push(SharedSliceMut::<usize>::abst_mem_mut(
                &x_fn, node_count, true,
            )?)
        }
        let x = Arc::new(x);

        // Thread syncronization
        let synchronize = Arc::new(Barrier::new(threads));

        // ParTriangle-AM4
        thread::scope(|scope| {
            let node_load = node_count.div_ceil(threads);
            let edge_load = node_count.div_ceil(threads);

            for tid in 0..threads {
                // eid is unnecessary as graph + index alwready do the job
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
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        let mut l: u8 = 1;
        let buff_size = 4096;
        let total_duds = Arc::new(AtomicUsize::new(0));
        let curr = SharedQueueMut::from_shared_slice(curr.shared_slice());
        let next = SharedQueueMut::from_shared_slice(next.shared_slice());

        thread::scope(|scope| {
            let mut res = Vec::new();
            for tid in 0..threads {
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

                res.push(scope.spawn(
                    move |_| -> Result<Box<[usize]>, Box<dyn std::error::Error + Send + Sync>> {
                        let mut res = vec![0usize; u8::MAX as usize];
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
                                return Err(format!(
                                    "error overflow when decrementing todo ({todo} - {})",
                                    total_duds.load(Ordering::Relaxed)
                                )
                                .into());
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

                            let mut to_process = curr.slice(0, curr.len()).ok_or_else(
                                || -> Box<dyn std::error::Error + Send + Sync> {
                                    "error reading curr in pkt".into()
                                },
                            )?;
                            // println!("new cicle initialized {} {:?}", todo, curr.ptr);
                            while !to_process.is_empty() {
                                todo = match todo.overflowing_sub(to_process.len()) {
                                    (r, false) => r,
                                    _ => {
                                        return Err(format!(
                                            "error overflow when decrementing todo ({todo} - {})",
                                            to_process.len()
                                        )
                                        .into());
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
                                            let prev_l_v_w =
                                                s.get(v_w).fetch_sub(1, Ordering::SeqCst);
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
                                            let prev_l_w_u =
                                                s.get(w_u).fetch_sub(1, Ordering::SeqCst);
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
                                            && ((u_v < w_u && *in_curr.get(w_u))
                                                || !*in_curr.get(w_u))
                                        {
                                            let prev_l_v_w =
                                                s.get(v_w).fetch_sub(1, Ordering::SeqCst);
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
                                            && ((u_v < v_w && *in_curr.get(v_w))
                                                || !*in_curr.get(v_w))
                                        {
                                            let prev_l_w_u =
                                                s.get(w_u).fetch_sub(1, Ordering::SeqCst);
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
                                }
                                // println
                                for _e in begin..end {
                                    res[l as usize] += 1;
                                }

                                synchronize.wait();
                                next = std::mem::replace(&mut curr, next).clear();
                                in_next = std::mem::replace(&mut in_curr, in_next);

                                synchronize.wait();
                                to_process = curr.slice(0, curr.len()).ok_or_else(
                                    || -> Box<dyn std::error::Error + Send + Sync> {
                                        "error reading curr in pkt".into()
                                    },
                                )?;
                                synchronize.wait();
                            }
                            l = match l.overflowing_add(1) {
                                (r, false) => r,
                                _ => {
                                    return Err(format!(
                                        "error overflow when adding to l ({l} - 1)"
                                    )
                                    .into());
                                }
                            };
                            synchronize.wait();
                        }
                        Ok(res.into_boxed_slice())
                    },
                ));
            }
            let joined_res: Vec<Box<[usize]>> = res
                .into_iter()
                .map(|v| v.join().expect("error thread panicked").expect("error ??1"))
                .collect();
            let env_verbose_val =
                std::env::var("BRUIJNX_VERBOSE").unwrap_or_else(|_| "0".to_string());
            let verbose: bool = env_verbose_val == "1";
            if verbose {
                let mut r = vec![0usize; u8::MAX as usize];
                for i in 0..u8::MAX as usize {
                    for v in joined_res.clone() {
                        r[i] += v[i];
                    }
                }

                let mut max = 0;
                r.iter().enumerate().for_each(|(i, v)| {
                    if *v != 0 && i > max {
                        max = i;
                    }
                });
                r.resize(max + 1, 0);
                println!("k-trussness {:?}", r);
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        // cleanup cache
        self.g.cleanup_cache(CacheFile::KTrussPKT)?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{k_truss::verify_k_trusses, test_common::get_or_init_dataset_cache_entry};

    use super::*;
    use paste::paste;
    use std::path::Path;

    macro_rules! graph_tests {
        ($($name:ident => $path:expr ,)*) => {
            $(
                paste! {
                    #[test]
                    fn [<k_trusses_pkt_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
                        generic_test($path)
                    }
                }
            )*
        }
    }

    fn generic_test<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn std::error::Error>> {
        let graph = GraphMemoryMap::init_from_cache(
            get_or_init_dataset_cache_entry(path.as_ref())?,
            Some(16),
        )?;
        let pkt_k_trusses = AlgoPKT::new(&graph)?;

        verify_k_trusses(&graph, pkt_k_trusses.k_trusses)
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
        // ggcat_9_5 => "../ggcat/graphs/random_graph_9_5.lz4",
        // ggcat_1_10 => "../ggcat/graphs/random_graph_1_10.lz4",
        // ggcat_2_10 => "../ggcat/graphs/random_graph_2_10.lz4",
        // ggcat_3_10 => "../ggcat/graphs/random_graph_3_10.lz4",
        // ggcat_4_10 => "../ggcat/graphs/random_graph_4_10.lz4",
        // ggcat_5_10 => "../ggcat/graphs/random_graph_5_10.lz4",
        // ggcat_6_10 => "../ggcat/graphs/random_graph_6_10.lz4",
        // ggcat_7_10 => "../ggcat/graphs/random_graph_7_10.lz4",
        // ggcat_8_10 => "../ggcat/graphs/random_graph_8_10.lz4",
        // ggcat_9_10 => "../ggcat/graphs/random_graph_9_10.lz4",
        // ggcat_8_15 => "../ggcat/graphs/random_graph_8_15.lz4",
        // ggcat_9_15 => "../ggcat/graphs/random_graph_9_15.lz4",
        // … add the rest
    }
}
