use crate::graph::*;
use crate::shared_slice::*;
#[cfg(test)]
use crate::utils::{FileType::ExactClosenessCentrality, H};

use crossbeam::thread;
use hyperloglog_rs::prelude::WordType;
use hyperloglog_rs::prelude::*;
use hyperloglog_rs::prelude::{HyperLogLog, HyperLogLogTrait};
use portable_atomic::{AtomicUsize, Ordering};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Barrier};

type ProceduralMemoryHB<P, const B: usize> = (
    AbstractedProceduralMemoryMut<HyperLogLog<P, B>>,
    AbstractedProceduralMemoryMut<f64>,
    AbstractedProceduralMemoryMut<f64>,
);

pub type HyperBall4<'a, ET, E> = HyperBallInner<'a, ET, E, Precision4, 6>;
pub type HyperBall5<'a, ET, E> = HyperBallInner<'a, ET, E, Precision5, 6>;
pub type HyperBall6<'a, ET, E> = HyperBallInner<'a, ET, E, Precision6, 6>;
pub type HyperBall7<'a, ET, E> = HyperBallInner<'a, ET, E, Precision7, 6>;
pub type HyperBall8<'a, ET, E> = HyperBallInner<'a, ET, E, Precision8, 6>;
pub type HyperBall9<'a, ET, E> = HyperBallInner<'a, ET, E, Precision9, 6>;
pub type HyperBall10<'a, ET, E> = HyperBallInner<'a, ET, E, Precision10, 6>;
pub type HyperBall11<'a, ET, E> = HyperBallInner<'a, ET, E, Precision11, 6>;
pub type HyperBall12<'a, ET, E> = HyperBallInner<'a, ET, E, Precision12, 6>;
pub type HyperBall13<'a, ET, E> = HyperBallInner<'a, ET, E, Precision13, 6>;
pub type HyperBall14<'a, ET, E> = HyperBallInner<'a, ET, E, Precision14, 6>;
pub type HyperBall15<'a, ET, E> = HyperBallInner<'a, ET, E, Precision15, 6>;
pub type HyperBall16<'a, ET, E> = HyperBallInner<'a, ET, E, Precision16, 6>;
pub type HyperBall17<'a, ET, E> = HyperBallInner<'a, ET, E, Precision17, 6>;
pub type HyperBall18<'a, ET, E> = HyperBallInner<'a, ET, E, Precision18, 6>;

/// Enum for centralities' caching filenames' creation manager logic.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
enum Centrality {
    Harmonic,
    NHarmonic,
    NCHarmonic,
    Closeness,
    NCloseness,
    NCCloseness,
    Lin,
}

/// For the *HyperBall Algorithm* described in ["In-Core Computation of Geometric Centralities with HyperBall: A Hundred Billion Nodes and Beyond"](https://doi.org/10.48550/arXiv.1308.2144) by Boldi P. and Vigna S. on [`GraphMemoryMap`] instances.
///
/// * Note: De Bruijn graphs are symmetric graphs, so running the HyperBall algorithm on the graph is the same as running it on its transpose. Hence, we can run HyperBall ona De Bruinjn graph to obtain the necessary components to determine the closeness, harmonic and Lin's centralities for every node of the graph.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
#[derive(Debug)]
pub struct HyperBallInner<
    'a,
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
    P: WordType<B> = Precision8,
    const B: usize = 6,
> {
    /// Graph for which *HyperBall* is computed.
    g: &'a GraphMemoryMap<EdgeType, Edge>,
    /// Memmapped slice containing each node's *HyperLogLog++* counter.
    counters: AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
    /// Memmapped slice containing each node's distance accumulator.
    distances: AbstractedProceduralMemoryMut<f64>,
    /// Memmapped slice containing each node's inverse distance accumulator.
    inverse_distances: AbstractedProceduralMemoryMut<f64>,
    /// Precision of the *HyperLogLog++* counters.
    precision: u8,
    /// Maximum number of iteration --- default is 128, max is 1024 (details on how these values are ludicrously big are found in the abovementioned paper).
    max_t: usize,
    /// Closeness centralities' cached values --- under various degrees of normalization.
    closeness: [Option<AbstractedProceduralMemoryMut<f64>>; 3],
    /// Harmonic centralities' cached values --- under various degrees of normalization.
    harmonic: [Option<AbstractedProceduralMemoryMut<f64>>; 3],
    /// Lin's centrality's cached values.
    lin: Option<AbstractedProceduralMemoryMut<f64>>,
    #[cfg(feature = "bench")]
    iters: usize,
}

#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>, P: WordType<B>, const B: usize>
    HyperBallInner<'a, EdgeType, Edge, P, B>
{
    const DEAFULT_PRECISION: u8 = 8;
    const MIN_PRECISION: u8 = 4;
    const MAX_PRECISION: u8 = 18;
    const DEAFULT_MAX_DEPTH: usize = 100;
    const MAX_MAX_DEPTH: usize = 1024;

    /// Performs the *HyperBall Algorithm* as described in ["In-Core Computation of Geometric Centralities with HyperBall: A Hundred Billion Nodes and Beyond"](https://doi.org/10.48550/arXiv.1308.2144) by Boldi P. and Vigna S.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which the *HyperBall Algorithm* is to be performed in.
    /// * `precision` --- the precision to be used for each nodes *HyperLogLog++* counter (defaults to 8, equivalent to 2⁸-register bits per node).
    /// * `max_depth` --- the maximum number of iterations of the *HyperBall Algorithm* to tolerate before convergence is achieved (defaults to 128, max is 1024).
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
        precision: Option<u8>,
        max_depth: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let mut hyper_ball = Self::new_no_compute(g, precision, max_depth)?;

        let proc_mem = hyper_ball.init_cache_mem()?;
        hyper_ball.compute_with_proc_mem(proc_mem)?;

        hyper_ball.g.cleanup_cache(CacheFile::HyperBall)?;

        Ok(hyper_ball)
    }

    /// Manager logic for centralities' caching filenames' creation.
    fn centrality_cache_file_name(
        &self,
        centrality: Centrality,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let (cache_file_type, seq) = match centrality {
            Centrality::Closeness => (CacheFile::HyperBallClosenessCentrality, Some(0)),
            Centrality::NCloseness => (CacheFile::HyperBallClosenessCentrality, Some(1)),
            Centrality::NCCloseness => (CacheFile::HyperBallClosenessCentrality, Some(2)),
            Centrality::Harmonic => (CacheFile::HyperBallHarmonicCentrality, Some(0)),
            Centrality::NHarmonic => (CacheFile::HyperBallHarmonicCentrality, Some(1)),
            Centrality::NCHarmonic => (CacheFile::HyperBallHarmonicCentrality, Some(2)),
            Centrality::Lin => (CacheFile::HyperBallLinCentrality, Some(0)),
        };

        self.build_cache_filename(cache_file_type, seq)
    }

    fn index_by_normalization(normalization: Option<bool>) -> usize {
        match normalization {
            None => 0,
            Some(false) => 1,
            Some(true) => 2,
        }
    }

    /// Gets (from a previously cached result) the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- tri-state flag determining the type of normalization that was used for the computation.
    ///     - [`None`] --- no normalization.
    ///     - [`Some`] ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - [`Some`] ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn get_closeness_centrality(
        &self,
        normalization: Option<bool>,
    ) -> Result<&[f64], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        let mem = self.closeness[idx]
            .as_ref() // borrow Option, do not move it
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "error closeness centrality isn't yet in cache".into()
            })?;
        Ok(mem.as_slice())
    }

    /// Gets (from a previously cached result) or computes the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- tri-state flag determining the type of normalization that was used for the computation.
    ///     - [`None`] --- no normalization.
    ///     - [`Some`] ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - [`Some`] ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn get_or_compute_closeness_centrality(
        &'a mut self,
        normalization: Option<bool>,
    ) -> Result<&'a [f64], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if self.harmonic[idx].is_none() {
            self.compute_closeness_centrality(normalization)
        } else {
            self.get_closeness_centrality(normalization)
        }
    }

    /// Computes the approximation of each node's *Closeness Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalize` --- tri-state flag determining the type of normalization that was used for the computation.
    ///     - [`None`] --- no normalization.
    ///     - [`Some`] ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - [`Some`] ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn compute_closeness_centrality(
        &mut self,
        normalize: Option<bool>,
    ) -> Result<&[f64], Box<dyn std::error::Error>> {
        let node_count = self.g.size(); // |V|

        let c_fn =
            self.centrality_cache_file_name(normalize.map_or(Centrality::Closeness, |local| {
                if local {
                    Centrality::NCCloseness
                } else {
                    Centrality::NCloseness
                }
            }))?;
        let mut mem = SharedSliceMut::<f64>::abst_mem_mut(&c_fn, node_count, true)?;

        // unnormalized
        if normalize.is_none() {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = 1. / *self.distances.get(idx);
                }
            }
            self.closeness[0] = Some(mem);
        // normalized by number of reacheable nodes
        } else if normalize.unwrap() {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = (self.counters.get_mut(idx).estimate_cardinality() as f64
                        - 1.)
                        / *self.distances.get(idx);
                }
            }
            self.closeness[1] = Some(mem);
        // normalized by node count (|V| - 1)
        } else {
            let normalize_factor = node_count as f64 - 1.; // |V| - 1
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = normalize_factor / *self.distances.get(idx);
                }
            }
            self.closeness[2] = Some(mem);
        }
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        self.get_closeness_centrality(normalize)
    }

    /// Gets (from a previously cached result) the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization was used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- tri-state flag determining the type of normalization that was used for the computation.
    ///     - [`None`] --- no normalization.
    ///     - [`Some`] ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - [`Some`] ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn get_hamonic_centrality(
        &self,
        normalization: Option<bool>,
    ) -> Result<&[f64], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        let mem = self.harmonic[idx]
            .as_ref() // borrow Option, do not move it
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "error harmonic centrality isn't yet in cache".into()
            })?;
        Ok(mem.as_slice())
    }

    /// Gets (from a previously cached result) or computes the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalization` --- tri-state flag determining the type of normalization that was used for the computation.
    ///     - [`None`] --- no normalization.
    ///     - [`Some`] ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - [`Some`] ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn get_or_compute_harmonic_centrality(
        &'a mut self,
        normalization: Option<bool>,
    ) -> Result<&'a [f64], Box<dyn std::error::Error>> {
        let idx = Self::index_by_normalization(normalization);
        if self.harmonic[idx].is_none() {
            self.compute_harmonic_centrality(normalization)
        } else {
            self.get_hamonic_centrality(normalization)
        }
    }

    /// Computes the approximation of each node's *Harmonic Centrality* from their respective distance accumulator (and if normalization is used, possibly, their respective *HyperLogLog++* counter estimation).
    ///
    /// # Arguments
    ///
    /// * `normalize` --- tri-state flag determining the type of normalization that was used for the computation.
    ///     - [`None`] --- no normalization.
    ///     - [`Some`] ([`false`]) --- centrality normalized by each node's estimate of reacheable nodes, effectively, normalization by containing connected component size.
    ///     - [`Some`] ([`true`]) --- centrality normalized by total number of nodes, `|V|`.
    ///
    pub fn compute_harmonic_centrality(
        &mut self,
        normalize: Option<bool>,
    ) -> Result<&[f64], Box<dyn std::error::Error>> {
        let node_count = self.g.size(); // |V|

        let c_fn =
            self.centrality_cache_file_name(normalize.map_or(Centrality::Harmonic, |local| {
                if local {
                    Centrality::NCHarmonic
                } else {
                    Centrality::NHarmonic
                }
            }))?;
        let mut mem = SharedSliceMut::<f64>::abst_mem_mut(&c_fn, node_count, true)?;

        // unnormalized
        if normalize.is_none() {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = *self.inverse_distances.get(idx);
                }
            }
            self.harmonic[0] = Some(mem);
        // normalized by number of reacheable nodes
        } else if normalize.unwrap() {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = *self.inverse_distances.get(idx)
                        / (self.counters.get_mut(idx).estimate_cardinality() as f64 - 1.);
                }
            }
            self.harmonic[1] = Some(mem);
        // normalized by node count (|v| - 1)
        } else {
            let normalize_factor = node_count as f64 - 1.; // |V| - 1
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = *self.inverse_distances.get(idx) / normalize_factor;
                }
            }
            self.harmonic[2] = Some(mem);
        }
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        self.get_hamonic_centrality(normalize)
    }

    /// Gets (from a previously cached result) the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog++* counter estimation.
    pub fn get_lins_centrality(&self) -> Result<&[f64], Box<dyn std::error::Error>> {
        let mem = self
            .lin
            .as_ref() // borrow Option, do not move it
            .ok_or_else(|| -> Box<dyn std::error::Error> {
                "error lin's centrality isn't yet in cache".into()
            })?;
        Ok(mem.as_slice())
    }

    /// Gets (from a previously cached result) or computes the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog++* counter estimation.
    pub fn get_or_compute_lins_centrality(
        &'a mut self,
    ) -> Result<&'a [f64], Box<dyn std::error::Error>> {
        if self.lin.is_none() {
            self.compute_lins_centrality()
        } else {
            self.get_lins_centrality()
        }
    }

    /// Computes the approximation of each node's *Lin's Centrality* from their respective distance accumulator and their respective *HyperLogLog++* counter estimation.
    pub fn compute_lins_centrality(&mut self) -> Result<&[f64], Box<dyn std::error::Error>> {
        let node_count = self.g.size(); // |V|

        let c_fn = self.centrality_cache_file_name(Centrality::Lin)?;
        let mut mem = SharedSliceMut::<f64>::abst_mem_mut(&c_fn, node_count, true)?;

        // lin's centrality is like a closeness centraility doubly normalized by number of
        // reacheable nodes
        for idx in 0..node_count {
            if !self.distances.get(idx).is_normal() {
                *mem.get_mut(idx) = 1.;
            } else {
                *mem.get_mut(idx) = (self.counters.get_mut(idx).estimate_cardinality() as f64 - 1.)
                    .powi(2)
                    / *self.distances.get(idx);
            }
        }

        self.lin = Some(mem);
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        self.get_lins_centrality()
    }

    /// Removes all cached files pertaining to this algorithm's execution's results, as well as any
    /// files containing centrailty values computed through them.
    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.build_cache_filename(CacheFile::HyperBall, None)?;
        let d_fn = this.build_cache_filename(CacheFile::HyperBallDistances, None)?;
        let i_fn = this.build_cache_filename(CacheFile::HyperBallInvDistances, None)?;
        std::fs::remove_file(out_fn)?;
        std::fs::remove_file(d_fn)?;
        std::fs::remove_file(i_fn)?;

        // remove any closeness centralities' cached values
        for (i, t) in [
            (0, Centrality::Closeness),
            (1, Centrality::NCloseness),
            (2, Centrality::NCloseness),
        ] {
            if this.closeness[i].is_some() {
                std::fs::remove_file(this.centrality_cache_file_name(t)?)?;
            }
        }

        // remove any harmonic centralities' cached values
        for (i, t) in [
            (0, Centrality::Harmonic),
            (1, Centrality::NHarmonic),
            (2, Centrality::NCHarmonic),
        ] {
            if this.closeness[i].is_some() {
                std::fs::remove_file(this.centrality_cache_file_name(t)?)?;
            }
        }

        // remove any lin's centrality's cached values
        if this.lin.is_some() {
            std::fs::remove_file(this.centrality_cache_file_name(Centrality::Lin)?)?;
        }

        Ok(())
    }
}

#[allow(dead_code)]
/// HyperBall engine functions.
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>, P: WordType<B>, const B: usize>
    HyperBallInner<'a, EdgeType, Edge, P, B>
{
    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
        precision: Option<u8>,
        max_depth: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, precision, max_depth)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn new_no_compute(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
        precision: Option<u8>,
        max_depth: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::new_no_compute_impl(g, precision, max_depth)
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn init_cache_mem(
        &self,
    ) -> Result<
        AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
        Box<dyn std::error::Error>,
    > {
        self.init_cache_mem_impl()
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn init_cache_mem(
        &self,
    ) -> Result<
        AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
        Box<dyn std::error::Error>,
    > {
        self.init_cache_mem_impl()
    }

    #[cfg(feature = "bench")]
    #[inline(always)]
    pub fn compute_with_proc_mem(
        &mut self,
        swap: AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(swap)
    }

    #[cfg(not(feature = "bench"))]
    #[inline(always)]
    pub(crate) fn compute_with_proc_mem(
        &mut self,
        swap: AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.compute_with_proc_mem_impl(swap)
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
        self.iters
    }

    fn init_cache(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<ProceduralMemoryHB<P, B>, Box<dyn std::error::Error>> {
        let node_count = g.size();

        let c_fn = g.build_cache_filename(CacheFile::HyperBall, None)?;
        let d_fn = g.build_cache_filename(CacheFile::HyperBallDistances, None)?;
        let i_fn = g.build_cache_filename(CacheFile::HyperBallInvDistances, None)?;

        let counters = SharedSliceMut::<hyperloglog_rs::prelude::HyperLogLog<P, B>>::abst_mem_mut(
            &c_fn, node_count, true,
        )?;
        let distances = SharedSliceMut::<f64>::abst_mem_mut(&d_fn, node_count, true)?;
        let inverse_distances = SharedSliceMut::<f64>::abst_mem_mut(&i_fn, node_count, true)?;

        Ok((counters, distances, inverse_distances))
    }

    fn new_no_compute_impl(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
        precision: Option<u8>,
        max_depth: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let node_count = g.size();
        // make sure presision is within bounds
        let precision =
            precision.map_or(HyperBallInner::<EdgeType, Edge>::DEAFULT_PRECISION, |p| {
                p.clamp(
                    HyperBallInner::<EdgeType, Edge>::MIN_PRECISION,
                    HyperBallInner::<EdgeType, Edge>::MAX_PRECISION,
                )
            });
        // make sure depth is within bounds
        let max_t = max_depth.map_or(HyperBallInner::<EdgeType, Edge>::DEAFULT_MAX_DEPTH, |p| {
            std::cmp::max(HyperBallInner::<EdgeType, Edge>::MAX_MAX_DEPTH, p)
        });
        // init cached vecs for distances and inverse distances accumulation
        let (mut counters, mut distances, mut inverse_distances) = Self::init_cache(g)?;
        // init counters: foreach v in 0..=n { add(c[v], v) } & distances and inverse_distances
        (0..node_count).for_each(|u| {
            *counters.get_mut(u) = hyperloglog_rs::prelude::HyperLogLog::<P, B>::default();
            counters.get_mut(u).insert(u);
            *distances.get_mut(u) = 0f64;
            *inverse_distances.get_mut(u) = 0f64;
        });

        Ok(Self {
            g,
            counters,
            distances,
            inverse_distances,
            precision,
            max_t,
            closeness: [None, None, None],
            harmonic: [None, None, None],
            lin: None,
            #[cfg(feature = "bench")]
            iters: 0,
        })
    }

    fn init_cache_mem_impl(
        &self,
    ) -> Result<
        AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
        Box<dyn std::error::Error>,
    > {
        let node_count = self.g.size();

        let s_fn = self.g.build_cache_filename(CacheFile::HyperBall, Some(0))?;
        let swap = SharedSliceMut::<hyperloglog_rs::prelude::HyperLogLog<P, B>>::abst_mem_mut(
            &s_fn, node_count, true,
        )?;

        Ok(swap)
    }

    fn compute_with_proc_mem_impl(
        &mut self,
        swap: AbstractedProceduralMemoryMut<hyperloglog_rs::prelude::HyperLogLog<P, B>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.g.size();
        let threads = self.g.thread_num();
        let node_load = node_count.div_ceil(threads);

        let global_changed = Arc::new(AtomicUsize::new(0));
        let synchronize = Arc::new(Barrier::new(threads));

        let counters = self.counters.shared_slice();
        let swap = swap.shared_slice();

        let mut t_f64: f64 = 1.; // first iteration is initialization
        let mut inv_t_f64: f64 = 1.; // first iteration is initialization
        let mut changed = 0;

        #[cfg(feature = "bench")]
        let iters = Arc::new(AtomicUsize::new(0));

        thread::scope(
            |scope| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                for tid in 0..threads {
                    let graph = &self.g;

                    let mut counters = counters;
                    let mut swap = swap;
                    let mut distance_accumulator = self.distances.shared_slice();
                    let mut inverse_distance_accumulator = self.inverse_distances.shared_slice();

                    let global_changed = Arc::clone(&global_changed);
                    let synchronize = Arc::clone(&synchronize);
                    #[cfg(feature = "bench")]
                    let iters = iters.clone();

                    let begin = std::cmp::min(tid * node_load, node_count);
                    let end = std::cmp::min(begin + node_load, node_count);
                    scope.spawn(
                        move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                            loop {
                                #[cfg(feature = "bench")]
                                if tid == 0 {
                                    iters.add(1, Ordering::Relaxed);
                                }
                                {
                                    // println!("HyperBall tid {tid} iteration {t_f64}");
                                }

                                for u in begin..end {
                                    let mut a: HyperLogLog<P, B> = *counters.get_mut(u);

                                    let prev_count = a.estimate_cardinality();

                                    let u_n = graph.neighbours(u).map_err(
                                        |e| -> Box<dyn std::error::Error + Send + Sync> {
                                            format!("couldn't find neighbours of {u}: {e}").into()
                                        },
                                    )?;

                                    for v in u_n {
                                        a |= counters.get(v.dest());
                                    }

                                    let curr_count = a.estimate_cardinality();
                                    let count_diff = (curr_count - prev_count) as f64;

                                    if count_diff > 0f64 {
                                        // update accumulators
                                        *distance_accumulator.get_mut(u) += t_f64 * count_diff;
                                        *inverse_distance_accumulator.get_mut(u) +=
                                            inv_t_f64 * count_diff;
                                        // update local changed count
                                        changed += 1;
                                    }

                                    *swap.get_mut(u) = a;
                                }

                                global_changed.fetch_add(changed, Ordering::Relaxed);
                                synchronize.wait();

                                if changed == 0 && global_changed.load(Ordering::Relaxed) == 0 {
                                    break Ok(());
                                }

                                swap = std::mem::replace(&mut counters, swap);
                                changed = 0;
                                t_f64 += 1.;
                                inv_t_f64 = 1. / t_f64;

                                synchronize.wait();

                                if tid == 0 {
                                    global_changed.store(0, Ordering::Relaxed);
                                }
                            }
                        },
                    );
                }
                Ok(())
            },
        )
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?
        .map_err(|e| -> Box<dyn std::error::Error> { format!("error HyperBall {:?}", e).into() })?;

        #[cfg(feature = "bench")]
        {
            self.iters = iters.load(Ordering::Relaxed);
        }

        self.counters.flush()?;
        self.g.cleanup_cache(CacheFile::HyperBall)?;

        // for i in 0..node_count {
        //     print!("\t|id: {i} ==> counter: {}|", self.counters.get(i).estimate_cardinality());
        // }
        // println!();

        Ok(())
    }
}

#[cfg(test)]
#[allow(dead_code)]
mod test {
    use hyperloglog_rs::prelude::*;

    use super::*;
    use crate::test_common::{get_or_init_dataset_cache_entry, get_or_init_dataset_exact_value};
    use paste::paste;
    use std::path::Path;

    fn mae(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>() / (a.len() as f64)
    }
    fn mape(a: &[f64], b: &[f64]) -> f64 {
        let eps = 1e-12;
        100.0
            * a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs() / (y.abs().max(eps)))
                .sum::<f64>()
            / (a.len() as f64)
    }

    fn ranks(v: &[f64]) -> Vec<usize> {
        // descending rank; stable
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[j].partial_cmp(&v[i]).unwrap_or(std::cmp::Ordering::Equal));
        let mut r = vec![0; v.len()];
        for (rank, i) in idx.into_iter().enumerate() {
            r[i] = rank;
        }
        r
    }

    fn spearman_rho(a: &[f64], b: &[f64]) -> f64 {
        let ra = ranks(a);
        let rb = ranks(b);
        let n = a.len() as f64;
        let ssd = ra
            .iter()
            .zip(rb.iter())
            .map(|(x, y)| {
                let d = (*x as f64) - (*y as f64);
                d * d
            })
            .sum::<f64>();
        1.0 - (6.0 * ssd) / (n * (n * n - 1.0).max(1.0))
    }

    macro_rules! graph_tests {
        ($($name:ident => $path:expr ,)*) => {
            $(
                paste! {
                    #[test]
                    fn [<hyperball_closeness_centrality_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
                        generic_test($path)
                    }
                }
            )*
        }
    }

    fn generic_test<P: AsRef<Path>>(p: P) -> Result<(), Box<dyn std::error::Error>> {
        let g = GraphMemoryMap::init(get_or_init_dataset_cache_entry(p.as_ref())?, Some(16))?;

        let e_fn = get_or_init_dataset_exact_value(p.as_ref(), &g, ExactClosenessCentrality(H::H))?;
        let exact = SharedSliceMut::<f64>::abst_mem_mut(&e_fn, g.size(), true)?;
        let e = exact.shared_slice();

        let mut hyperball = HyperBallInner::<_, _, Precision12, 6>::new(&g, None, None)?;
        let approx = hyperball.compute_closeness_centrality(Some(false))?;

        // metrics
        let e_mae = mae(approx, e.as_slice());
        let e_mape = mape(approx, e.as_slice());
        let rho = spearman_rho(approx, e.as_slice());

        eprintln!(
            "dataset={:?}  MAE={e_mae:.4e}  MAPE={e_mape:.2}%  Spearman={rho:.4}",
            p.as_ref()
        );

        // guardrails to fail regressions
        assert!(rho > 0.95, "Spearman too low on {:?}", p.as_ref());
        Ok(())
    }

    // generate test cases from dataset
    graph_tests! {
        // ggcat_1_5 => "../ggcat/graphs/random_graph_1_5.lz4",
        // ggcat_2_5 => "../ggcat/graphs/random_graph_2_5.lz4",
        // ggcat_3_5 => "../ggcat/graphs/random_graph_3_5.lz4",
        // ggcat_4_5 => "../ggcat/graphs/random_graph_4_5.lz4",
        // ggcat_5_5 => "../ggcat/graphs/random_graph_5_5.lz4",
        // ggcat_6_5 => "../ggcat/graphs/random_graph_6_5.lz4",
        // ggcat_7_5 => "../ggcat/graphs/random_graph_7_5.lz4",
        // ggcat_8_5 => "../ggcat/graphs/random_graph_8_5.lz4",
        // ggcat_9_5 => "../ggcat/graphs/random_graph_9_5.lz4",
        // ggcat_1_10 => "../ggcat/graphs/random_graph_1_10.lz4",
        // ggcat_2_10 => "../ggcat/graphs/random_graph_2_10.lz4",
        // ggcat_3_10 => "../ggcat/graphs/random_graph_3_10.lz4",
        ggcat_4_10 => "../ggcat/graphs/random_graph_4_10.lz4",
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
