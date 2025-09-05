use std::mem::ManuallyDrop;

use crate::graph;
use crate::graph::*;
use crate::shared_slice::*;
use crate::trails::bfs::BFSDists;

use crossbeam::thread;

pub struct ExactClosenessCentrality<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<f64>,
    /// Normalization flag.
    normalized: Option<bool>,
}

#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ExactClosenessCentrality<'a, N, E, Ix> {
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g, normalized)?;

        hyper_ball.compute_with_proc_mem(())?;

        hyper_ball.g.cleanup_cache(CacheFile::HyperBall)?;

        Ok(hyper_ball)
    }

    pub fn get_node_centrality(&self, idx: usize) -> f64 {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[f64] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactCloseness, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ExactClosenessCentrality<'a, N, E, Ix> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactCloseness, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
            normalized,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.g.thread_num();
        let node_load = node_count.div_ceil(threads);
        let index = Self::index_by_normalization(self.normalized);

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let graph = &self.g;

                let mut e = self.centralities.shared_slice();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                threads_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        if index == 0 {
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
                                    *e.get_mut(u) = 0.0;
                                } else {
                                    *e.get_mut(u) = bfs.total_distances();
                                }
                            }
                        } else if index == 1 {
                            let norm = node_count.saturating_sub(1) as f64;
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
                                    *e.get_mut(u) = 0.0;
                                } else {
                                    *e.get_mut(u) = norm / bfs.total_distances();
                                }
                            }
                        } else {
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
                                    *e.get_mut(u) = 0.0;
                                } else {
                                    *e.get_mut(u) = bfs.recheable() as f64 / bfs.total_distances();
                                }
                            }
                        }
                        Ok(())
                    },
                ));
            }

            // check for errors
            for (tid, r) in threads_res.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in initialization (thread {tid}): {:?}", e).into()
                    })?;
            }

            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        self.g.cleanup_cache(CacheFile::ExactCloseness)?;

        Ok(())
    }
}

pub struct ExactHarmonicCentrality<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<f64>,
    /// Normalization flag.
    normalized: Option<bool>,
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ExactHarmonicCentrality<'a, N, E, Ix> {
    pub fn new(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g, normalized)?;

        hyper_ball.compute_with_proc_mem(())?;

        hyper_ball.g.cleanup_cache(CacheFile::HyperBall)?;

        Ok(hyper_ball)
    }

    pub fn get_node_centrality(&self, idx: usize) -> f64 {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[f64] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactHarmonic, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ExactHarmonicCentrality<'a, N, E, Ix> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, normalized)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
        normalized: Option<bool>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactHarmonic, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
            normalized,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.g.thread_num();
        let node_load = node_count.div_ceil(threads);
        let index = Self::index_by_normalization(self.normalized);

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let graph = &self.g;

                let mut e = self.centralities.shared_slice();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                threads_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        if index == 0 {
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 {
                                    *e.get_mut(u) = 0.0;
                                } else {
                                    let inv_dist: f64 = bfs
                                        .get_distances()
                                        .iter()
                                        .map(|&d| {
                                            let d = d as f64;
                                            if d.is_normal() { 1. / d } else { 0. }
                                        })
                                        .sum();
                                    *e.get_mut(u) = inv_dist;
                                }
                            }
                        } else if index == 1 {
                            let norm = node_count.saturating_sub(1) as f64;
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 {
                                    *e.get_mut(u) = 0.0;
                                } else {
                                    let inv_dist: f64 = bfs
                                        .get_distances()
                                        .iter()
                                        .map(|&d| {
                                            let d = d as f64;
                                            if d.is_normal() { 1. / d } else { 0. }
                                        })
                                        .sum();
                                    *e.get_mut(u) = norm / inv_dist;
                                }
                            }
                        } else {
                            for u in begin..end {
                                let bfs = BFSDists::new(graph, u).map_err(
                                    |e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error: {:?}", e).into()
                                    },
                                )?;
                                if bfs.recheable() <= 1 {
                                    *e.get_mut(u) = 0.0;
                                } else {
                                    let inv_dist: f64 = bfs
                                        .get_distances()
                                        .iter()
                                        .map(|&d| {
                                            let d = d as f64;
                                            if d.is_normal() { 1. / d } else { 0. }
                                        })
                                        .sum();
                                    *e.get_mut(u) = bfs.recheable() as f64 / inv_dist;
                                }
                            }
                        }
                        Ok(())
                    },
                ));
            }

            // check for errors
            for (tid, r) in threads_res.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in initialization (thread {tid}): {:?}", e).into()
                    })?;
            }

            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        self.g.cleanup_cache(CacheFile::ExactHarmonic)?;

        Ok(())
    }
}

pub struct ExactLinCentrality<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<N, E, Ix>,
    /// Memmapped slice containing each node's centrality.
    centralities: AbstractedProceduralMemoryMut<f64>,
}
#[allow(dead_code)]
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ExactLinCentrality<'a, N, E, Ix> {
    pub fn new(g: &'a GraphMemoryMap<N, E, Ix>) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g)?;

        hyper_ball.compute_with_proc_mem(())?;

        hyper_ball.g.cleanup_cache(CacheFile::HyperBall)?;

        Ok(hyper_ball)
    }

    pub fn get_node_centrality(&self, idx: usize) -> f64 {
        *self.centralities.get(idx)
    }

    pub fn centralities(&self) -> &[f64] {
        self.centralities.as_slice()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::ExactLin, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, N: graph::N, E: graph::E, Ix: graph::IndexType> ExactLinCentrality<'a, N, E, Ix> {
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(&self) -> Result<(), Box<dyn std::error::Error>> {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    fn compute_with_proc_mem(&mut self, proc_mem: ()) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(proc_mem)
    }

    #[inline(always)]
    fn build_cache_filename(
        &self,
        file_type: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.g.build_cache_filename(file_type, seq)
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.size()
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<N, E, Ix>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        let e_fn = g.build_cache_filename(CacheFile::ExactCloseness, None)?;

        Ok(Self {
            g,
            centralities: SharedSliceMut::abst_mem_mut(&e_fn, node_count, true)?,
        })
    }

    fn init_cache_mem_impl(&self) -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        _proc_mem: (),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.g.thread_num();
        let node_load = node_count.div_ceil(threads);

        thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
            let mut threads_res = vec![];
            for tid in 0..threads {
                let graph = &self.g;

                let mut e = self.centralities.shared_slice();

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                threads_res.push(scope.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        for u in begin..end {
                            let bfs = BFSDists::new(graph, u).map_err(
                                |e| -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error: {:?}", e).into()
                                },
                            )?;
                            if bfs.total_distances() == 0. {
                                *e.get_mut(u) = 0.0;
                            } else {
                                *e.get_mut(u) =
                                    (bfs.recheable() as f64).powi(2) / bfs.total_distances();
                            }
                        }
                        Ok(())
                    },
                ));
            }

            // check for errors
            for (tid, r) in threads_res.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in initialization (thread {tid}): {:?}", e).into()
                    })?;
            }

            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        self.g.cleanup_cache(CacheFile::ExactLin)?;

        Ok(())
    }
}
