use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use crossbeam::thread;
use hyperloglog_rs::prelude::{HyperLogLog, HyperLogLogTrait};
use num_cpus::get_physical;
use std::{
    io::Error,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
};

type ProceduralMemoryHB = (
    AbstractedProceduralMemoryMut<
        hyperloglog_rs::prelude::HyperLogLog<hyperloglog_rs::prelude::Precision6, 8>,
    >,
    AbstractedProceduralMemoryMut<f64>,
    AbstractedProceduralMemoryMut<f64>,
);

#[allow(dead_code)]
#[allow(clippy::upper_case_acronyms)]
#[derive(Debug)]
enum Centrality {
    HYPERBALL,
    HARMONIC,
    NHARMONIC,
    NCHARMONIC,
    CLOSENESS,
    NCLOSENESS,
    NCCLOSENESS,
    LIN,
}

/// De Bruijn graphs are simmetric graphs, so running the HyperBall algorithm on the graph is
/// the same as running it on its transpose. Hence, we can run HyperBall ona De Bruinjn graph
/// to obtain the necessary components to determine the closeness, harmonic and Lin's centralities
/// for every node of the graph.
#[allow(dead_code)]
#[derive(Debug)]
pub struct HyperBallInner<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    /// vec with a hyperloglog++ counter for each node of the graph
    counters: AbstractedProceduralMemoryMut<
        hyperloglog_rs::prelude::HyperLogLog<hyperloglog_rs::prelude::Precision6, 8>,
    >,
    // counters: AbstractedProceduralMemoryMut<HyperLogLogPlus<usize, RandomState>>,
    /// cached slice of the sum of the distances for each node of the graph
    distances: AbstractedProceduralMemoryMut<f64>,
    /// cached slice of the sum of the inverse of the distances for each node of the graph
    inverse_distances: AbstractedProceduralMemoryMut<f64>,
    /// precision of the hyperloglog++ counters used in the algorithm
    precision: u8,
    /// maximum number of iteration --- default is 128, max is 1024 (arbitrarily large values)
    max_t: usize,
}

#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>
    HyperBallInner<'a, EdgeType, Edge>
{
    const DEAFULT_PRECISION: u8 = 8;
    const MIN_PRECISION: u8 = 4;
    const MAX_PRECISION: u8 = 18;
    const DEAFULT_MAX_DEPTH: usize = 100;
    const MAX_MAX_DEPTH: usize = 1024;

    fn init_cache(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<ProceduralMemoryHB, Box<dyn std::error::Error>> {
        let c_fn = cache_file_name(graph.cache_fst_filename(), FileType::HyperBall, None)?;
        let d_fn = cache_file_name(
            graph.cache_fst_filename(),
            FileType::HyperBallDistances,
            None,
        )?;
        let i_fn = cache_file_name(
            graph.cache_fst_filename(),
            FileType::HyperBallInvDistances,
            None,
        )?;
        let counters = SharedSliceMut::<
            hyperloglog_rs::prelude::HyperLogLog<hyperloglog_rs::prelude::Precision6, 8>,
        >::abst_mem_mut(c_fn, graph.size() - 1, true)?;
        let distances = SharedSliceMut::<f64>::abst_mem_mut(d_fn, graph.size() - 1, true)?;
        let inverse_distances = SharedSliceMut::<f64>::abst_mem_mut(i_fn, graph.size() - 1, true)?;

        Ok((counters, distances, inverse_distances))
    }

    fn centrality_cache_file_name(
        template_fn: String,
        centrality: Centrality,
    ) -> Result<String, Box<dyn std::error::Error>> {
        match centrality {
            Centrality::HYPERBALL => Ok(cache_file_name(
                template_fn,
                FileType::HyperBallHarmonicCentrality,
                Some(0),
            )?),
            Centrality::CLOSENESS => Ok(cache_file_name(
                template_fn,
                FileType::HyperBallClosenessCentrality,
                Some(0),
            )?),
            Centrality::NCLOSENESS => Ok(cache_file_name(
                template_fn,
                FileType::HyperBallClosenessCentrality,
                Some(1),
            )?),
            Centrality::NCCLOSENESS => Ok(cache_file_name(
                template_fn,
                FileType::HyperBallClosenessCentrality,
                Some(2),
            )?),
            Centrality::HARMONIC => Ok(cache_file_name(
                template_fn,
                FileType::HyperBallHarmonicCentrality,
                Some(0),
            )?),
            Centrality::NHARMONIC => Ok(cache_file_name(
                template_fn,
                FileType::HyperBallHarmonicCentrality,
                Some(1),
            )?),
            Centrality::NCHARMONIC => Ok(cache_file_name(
                template_fn,
                FileType::HyperBallHarmonicCentrality,
                Some(1),
            )?),
            Centrality::LIN => Ok(cache_file_name(
                template_fn,
                FileType::HyperBallLinCentrality,
                None,
            )?),
        }
    }

    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
        precision: Option<u8>,
        max_depth: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let (node_count, overflow) = graph.size().overflowing_sub(1);
        if overflow || node_count == 0 {
            return Err(Box::new(Error::new(
                std::io::ErrorKind::Other,
                "error initiating hyperball graph is empty",
            )));
        }
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
        let (mut counters, mut distances, mut inverse_distances) = Self::init_cache(graph)?;
        // init counters: foreach v in 0..=n { add(c[v], v) } & distances and inverse_distances
        (0..node_count).for_each(|u| {
            *counters.get_mut(u) = hyperloglog_rs::prelude::HyperLogLog::<
                hyperloglog_rs::prelude::Precision6,
                8,
            >::default();
            counters.get_mut(u).insert(u);
            *distances.get_mut(u) = 0f64;
            *inverse_distances.get_mut(u) = 0f64;
        });

        let mut hyper_ball = Self {
            graph,
            counters,
            distances,
            inverse_distances,
            precision,
            max_t,
        };

        hyper_ball.compute()?;

        Ok(hyper_ball)
    }

    #[deprecated]
    fn _compute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let mut t_f64: f64 = 1.; // first iteration is initialization
        let mut inv_t_f64: f64 = 1.; // first iteration is initialization
        let mut changed = 0;
        let s_fn = Self::centrality_cache_file_name(
            self.graph.cache_fst_filename(),
            Centrality::HYPERBALL,
        )?;
        let mut swap =
            SharedSliceMut::<HyperLogLog<hyperloglog_rs::prelude::Precision6, 8>>::abst_mem_mut(
                s_fn,
                self.graph.size() - 1,
                true,
            )?;

        let mut counters = self.counters.shared_slice();
        let mut counters = counters.mut_slice(0, self.graph.size() - 1).unwrap();
        let mut swap = swap.mut_slice(0, self.graph.size() - 1).unwrap();

        loop {
            for u in 0..self.graph.size() - 1 {
                let mut a: HyperLogLog<hyperloglog_rs::prelude::Precision6, 8> = counters[u];

                let prev_count = a.estimate_cardinality();

                for v in self.graph.neighbours(u)? {
                    a |= counters[v.dest()];
                }

                let curr_count = a.estimate_cardinality();
                let count_diff = (curr_count - prev_count) as f64;

                if count_diff > 0f64 {
                    *self.distances.get_mut(u) += t_f64 * count_diff;
                    *self.inverse_distances.get_mut(u) += inv_t_f64 * count_diff;
                    changed += 1;
                }

                swap[u] = a;
            }

            if changed == 0 {
                break;
            }

            swap = std::mem::replace(&mut counters, swap);
            changed = 0;
            t_f64 += 1.;
            inv_t_f64 = 1. / t_f64;
        }

        Ok(())
    }

    fn compute(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1;
        let threads = self.graph.thread_num().max(get_physical());
        let node_load = node_count.div_ceil(threads);

        let global_changed = Arc::new(AtomicUsize::new(0));
        let synchronize = Arc::new(Barrier::new(threads));

        let mut t_f64: f64 = 1.; // first iteration is initialization
        let mut inv_t_f64: f64 = 1.; // first iteration is initialization
        let mut changed = 0;

        let s_fn = Self::centrality_cache_file_name(
            self.graph.cache_fst_filename(),
            Centrality::HYPERBALL,
        )?;
        let swap = SharedSliceMut::<
            hyperloglog_rs::prelude::HyperLogLog<hyperloglog_rs::prelude::Precision6, 8>,
        >::abst_mem_mut(s_fn, self.graph.size() - 1, true)?;

        let counters = self.counters.shared_slice();
        let swap = swap.shared_slice();

        match thread::scope(|scope| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
            for tid in 0..threads {
                let graph = self.graph.clone();

                let mut counters = counters;
                let mut swap = swap;
                let mut distance_accumulator = self.distances.shared_slice();
                let mut inverse_distance_accumulator = self.inverse_distances.shared_slice();

                let global_changed = Arc::clone(&global_changed);
                let synchronize = Arc::clone(&synchronize);

                let begin = std::cmp::min(tid * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                scope.spawn(move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

                    loop {

                        println!("HyperBall tid {tid} iteration {t_f64}");

                        for u in begin..end {

                            let mut a: HyperLogLog::<hyperloglog_rs::prelude::Precision6, 8> = *counters.get_mut(u);

                            let prev_count = a.estimate_cardinality();

                            let u_n = match graph.neighbours(u) {
                                Ok(n) => n,
                                Err(_) => {
                                    return Err(Box::new(Error::new(
                                                std::io::ErrorKind::NotFound,
                                                format!("error HyperBall couldn't find neighbours of {u}")
                                                )));
                                }
                            };

                            for v in u_n {
                                a |= counters.get(v.dest());
                            }

                            let curr_count = a.estimate_cardinality();
                            let count_diff = (curr_count - prev_count) as f64;


                            if count_diff > 0f64 {
                                // update accumulators
                                *distance_accumulator.get_mut(u) += t_f64 * count_diff;
                                *inverse_distance_accumulator.get_mut(u) += inv_t_f64 * count_diff;
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
                });
            }
            Ok(())
        })
        .unwrap() {
            Ok(_) => {}
            Err(e) => {
                return Err(Box::new(Error::new(
                            std::io::ErrorKind::NotFound,
                            format!("error HyperBall: {e}")
                            ))
                    );
            }

        };
        self.counters.flush()?;
        cleanup_cache()?;

        // for i in 0..node_count {
        //     print!("\t|id: {i} ==> counter: {}|", self.counters.get(i).estimate_cardinality());
        // }
        // println!();

        Ok(())
    }

    // FIXME: for all centralities: what happens when the distance is not  normal and: is zero, or
    // is infinite
    pub fn compute_closeness_centrality(
        &mut self,
        normalize: Option<bool>,
    ) -> Result<AbstractedProceduralMemoryMut<f64>, Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1; // |V|
        let node_count_f64 = self.graph.size() as f64 - 2.; // |V| - 1

        let c_fn = Self::centrality_cache_file_name(
            self.graph.cache_fst_filename(),
            normalize.map_or(Centrality::CLOSENESS, |local| {
                if local {
                    Centrality::NCCLOSENESS
                } else {
                    Centrality::NCLOSENESS
                }
            }),
        )?;
        let mut mem = SharedSliceMut::<f64>::abst_mem_mut(c_fn, node_count, true)?;

        // unnormalized
        if normalize.is_none() {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = 1. / *self.distances.get(idx);
                }
            }
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
        // normalized by node count
        } else {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = node_count_f64 / *self.distances.get(idx);
                }
            }
        }
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        Ok(mem)
    }

    pub fn compute_harmonic_centrality(
        &mut self,
        normalize: Option<bool>,
    ) -> Result<AbstractedProceduralMemoryMut<f64>, Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1; // |V|
        let node_count_f64 = self.graph.size() as f64 - 2.; // |V| - 1

        let c_fn = Self::centrality_cache_file_name(
            self.graph.cache_fst_filename(),
            normalize.map_or(Centrality::HARMONIC, |local| {
                if local {
                    Centrality::NCHARMONIC
                } else {
                    Centrality::NHARMONIC
                }
            }),
        )?;
        let mut mem = SharedSliceMut::<f64>::abst_mem_mut(c_fn, node_count, true)?;

        // unnormalized
        if normalize.is_none() {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = *self.inverse_distances.get(idx);
                }
            }
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
        // normalized by node count
        } else {
            for idx in 0..node_count {
                if !self.distances.get(idx).is_normal() {
                    *mem.get_mut(idx) = 0.;
                } else {
                    *mem.get_mut(idx) = *self.inverse_distances.get(idx) / node_count_f64;
                }
            }
        }
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        Ok(mem)
    }

    pub fn compute_lins_centrality(
        &mut self,
    ) -> Result<AbstractedProceduralMemoryMut<f64>, Box<dyn std::error::Error>> {
        let node_count = self.graph.size() - 1; // |V|

        let c_fn =
            Self::centrality_cache_file_name(self.graph.cache_fst_filename(), Centrality::LIN)?;
        let mut mem = SharedSliceMut::<f64>::abst_mem_mut(c_fn, node_count, true)?;

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
        // if let Some(s) = mem.slice(0, node_count) {
        //     println!("{:?}\n centrality", s);
        // }
        Ok(mem)
    }
}
