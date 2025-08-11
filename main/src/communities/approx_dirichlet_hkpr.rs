use crate::generic_edge::*;
use crate::generic_memory_map::*;

use ordered_float::OrderedFloat;
use rand::{Rng, rng};
use rand_distr::{Distribution, Poisson};
use std::{collections::HashMap, io::Error};

#[allow(dead_code)]
#[derive(Clone)]
pub struct ApproxDirHKPR<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    pub t: f64,
    pub eps: f64,
    pub seed: usize,
    pub target_size: usize,
    pub target_vol: usize,
    pub target_conductance: f64,
}

/// describes the type of limiter to be used for the number of steps to take for each
/// random walk
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum ApproxDirichletHeatKernelK {
    /// too stringent of a limitation -- results will be no good
    None,
    /// optimization described in SolverApproxDirHKPR's algorithm
    Mean,
    /// no limit --- probably won't run for eps < 0.005
    Unlim,
}

#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> ApproxDirHKPR<'a, EdgeType, Edge> {
    #[inline(always)]
    fn f64_is_nomal(val: f64, op_description: &str) -> Result<f64, Error> {
        if !val.is_normal() {
            panic!(
                "error hk-relax abnormal value at {} = {}",
                op_description, val
            );
        }
        Ok(val)
    }

    fn f64_to_usize_safe(x: f64) -> Option<usize> {
        if x.is_normal() && x > 0f64 && x <= usize::MAX as f64 {
            Some(x as usize) // truncates toward zero
        } else {
            None
        }
    }

    fn evaluate_params(
        graph: GraphMemoryMap<EdgeType, Edge>,
        seed_node: usize,
        eps: f64,
        _target_size: usize,
        _target_vol: usize,
        target_conductance: f64,
    ) -> Result<(), Error> {
        let node_count = match graph.size().overflowing_sub(1) {
            (_, true) => {
                panic!("error hk-relax invalid parameters: |V| == 0, the graph is empty");
            }
            (i, _) => {
                if i == 0 {
                    panic!(
                        "error hk-relax invalid parameters: actual |V| == 0, the graph is empty"
                    );
                }
                i
            }
        };
        if !eps.is_normal() || eps <= 0f64 || eps >= 1f64 {
            panic!(
                "error hk-relax invalid parameters: ε == {} doesn't satisfy 0.0 < ε 1.0",
                eps
            );
        }
        if !target_conductance.is_normal()
            || target_conductance <= 0f64
            || target_conductance >= 1f64
        {
            panic!(
                "error hk-relax invalid parameters: target_conductance == {} doesn't satisfy 0.0 < target_conductance < 1.0",
                target_conductance
            );
        }
        if seed_node > node_count - 1 {
            panic!(
                "error hk-relax invalid parameters: id(seed_nodes) == {} but max_id(v in V) == {}",
                seed_node,
                node_count - 1
            );
        }
        Ok(())
    }

    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
        eps: f64,
        seed: usize,
        target_size: usize,
        target_vol: usize,
        target_conductance: f64,
    ) -> Result<Self, Error> {
        let () = Self::evaluate_params(
            graph.clone(),
            seed,
            eps,
            target_size,
            target_vol,
            target_conductance,
        )?;
        let t_formula = "(1. / target_conductance) * ln((2. * sqrt(target_vol)) / (1. - ε) + 2. * ε * target_size)";
        let t = Self::f64_is_nomal(
            (1. / target_conductance)
                * f64::ln(
                    (2. * f64::sqrt(target_vol as f64)) / (1. - eps)
                        + 2. * eps * target_size as f64,
                ),
            t_formula,
        )?;
        Ok(ApproxDirHKPR {
            graph,
            t,
            eps,
            seed,
            target_size,
            target_vol,
            target_conductance,
        })
    }

    fn random_sample_poisson(lambda: f64, n: usize) -> Result<Vec<OrderedFloat<f64>>, Error> {
        let mut rng = rng();
        let poisson = match Poisson::new(lambda) {
            Ok(dist) => dist,
            Err(e) => panic!(
                "error approx-dirchlet-hk couldn't sample poission distribution: {}",
                e
            ),
        };
        Ok(poisson
            .sample_iter(&mut rng)
            .take(n)
            .map(OrderedFloat)
            .collect())
    }

    #[inline(always)]
    fn random_neighbour(&self, deg_u: usize, u_n: NeighbourIter<EdgeType, Edge>) -> usize {
        let inv_deg_u = 1. / deg_u as f64;
        let random: f64 = rand::rng().random();
        let mut idx_plus_one = 1f64;
        for v in u_n {
            if idx_plus_one * inv_deg_u > random {
                return v.dest();
            }
            idx_plus_one += idx_plus_one;
        }
        panic!("error approx-dirchlet-hk didn't find random neighbour");
    }

    fn random_walk_seed(&self, k: usize, seed_node: usize) -> usize {
        let mut curr_node = seed_node;
        for _ in 0..k {
            let (deg_u, u_n) = match self.graph.neighbours(curr_node) {
                Ok(u_n) => (u_n.remaining_neighbours(), u_n),
                Err(e) => panic!(
                    "error approx-dirchlet-hk couldn't get neighbours for {}: {}",
                    curr_node, e
                ),
            };
            if deg_u > 0 {
                curr_node = self.random_neighbour(deg_u, u_n);
            } else {
                return curr_node;
            }
        }
        curr_node
    }

    #[allow(clippy::unreachable)]
    #[deprecated]
    pub fn compute_specify_k(
        &self,
        big_k: ApproxDirichletHeatKernelK,
    ) -> Result<Community<usize>, Error> {
        let node_count = match self.graph.size().overflowing_sub(1) {
            (_, true) => panic!("error approx-dirchlet-hk |V| + 1 == 0"),
            (0, _) => panic!("error approx-dirchlet-hk |V| == 0"),
            (i, _) => i as f64,
        };

        let r = Self::f64_is_nomal(
            (16. / self.eps.powi(2)) * f64::ln(node_count),
            "(16.0 / ε²) * ln(|V|)",
        )?;
        let one_over_r = Self::f64_is_nomal(1. / r, "1. / r")?;

        let k = match big_k {
            ApproxDirichletHeatKernelK::None => Self::f64_is_nomal(
                2. * f64::ln(1. / self.eps) / (f64::ln(f64::ln(1. / self.eps))),
                "2. * ln(1. / ε) / ln(ln(1. / ε))",
            )?,
            // optimization of SolverApproxDirHKPR
            ApproxDirichletHeatKernelK::Mean => Self::f64_is_nomal(2. * self.t, "2. * t")?,
            ApproxDirichletHeatKernelK::Unlim => f64::INFINITY,
            #[expect(unreachable_patterns)]
            _ => panic!("error approx-dirchlet-hk unknown K {:?}", big_k),
        };
        let k = OrderedFloat(k);
        println!(
            "k (ceil on sample value) computed to be {}\nr (number of samples) computed to be {}",
            k, r
        );

        let num_samples: usize = match Self::f64_to_usize_safe(r) {
            Some(s) => s,
            None => panic!("error approx-dirchlet-hk couldn't cast {} to usize", r),
        };
        let steps: Vec<OrderedFloat<f64>> = Self::random_sample_poisson(self.t, num_samples)?;
        let mut aprox_hkpr_samples: HashMap<usize, f64> = HashMap::new();

        for little_k in steps {
            let OrderedFloat(little_k) = std::cmp::min(little_k, k);
            let little_k_usize = match Self::f64_to_usize_safe(little_k) {
                Some(val) => val,
                None => panic!(
                    "error approx-dirchlet-hk couldn't cast {} to usize",
                    little_k
                ),
            };
            let v = self.random_walk_seed(little_k_usize, self.seed);
            match aprox_hkpr_samples.get_mut(&v) {
                Some(v) => *v += one_over_r,
                None => {
                    aprox_hkpr_samples.insert(v, one_over_r);
                }
            };
        }

        // FIXME: Never normalized
        let mut p: Vec<(usize, f64)> = aprox_hkpr_samples
            .keys()
            .map(|u| {
                (
                    *u,
                    *aprox_hkpr_samples.get(u).unwrap() / self.graph.node_degree(*u) as f64,
                )
            })
            .collect::<Vec<(usize, f64)>>();

        match self.graph.sweep_cut_over_diffusion_vector_by_conductance(
            p.as_mut(),
            Some(self.target_size),
            Some(self.target_vol),
        ) {
            Ok(c) => {
                println!(
                    "best community approxdirichlethkpr {{\n\tsize: {}\n\tvolume/width: {}\n\tconductance: {}\n}}",
                    c.size, c.width, c.conductance
                );
                Ok(c)
            }
            Err(e) => panic!("error: {}", e),
        }
    }

    pub fn compute(&self) -> Result<Community<usize>, Error> {
        let node_count = match self.graph.size().overflowing_sub(1) {
            (_, true) => panic!("error approx-dirchlet-hk |V| + 1 == 0"),
            (0, _) => panic!("error approx-dirchlet-hk |V| == 0"),
            (i, _) => i as f64,
        };

        let r = Self::f64_is_nomal(
            (16. / self.eps.powi(2)) * f64::ln(node_count),
            "(16.0 / ε²) * ln(|V|)",
        )?;
        let one_over_r = Self::f64_is_nomal(1. / r, "1. / r")?;

        let k = Self::f64_is_nomal(2. * self.t, "2. * t")?;
        let k = OrderedFloat(k);
        println!(
            "k (ceil on sample value) computed to be {}\nr (number of samples) computed to be {}",
            k, r
        );

        let num_samples: usize = match Self::f64_to_usize_safe(r) {
            Some(s) => s,
            None => panic!("error approx-dirchlet-hk couldn't cast {} to usize", r),
        };
        let steps: Vec<OrderedFloat<f64>> = Self::random_sample_poisson(self.t, num_samples)?;
        let mut aprox_hkpr_samples: HashMap<usize, f64> = HashMap::new();

        for little_k in steps {
            let OrderedFloat(little_k) = std::cmp::min(little_k, k);
            let little_k_usize = match Self::f64_to_usize_safe(little_k) {
                Some(val) => val,
                None => panic!(
                    "error approx-dirchlet-hk couldn't cast {} to usize",
                    little_k
                ),
            };
            let v = self.random_walk_seed(little_k_usize, self.seed);
            match aprox_hkpr_samples.get_mut(&v) {
                Some(v) => *v += one_over_r,
                None => {
                    aprox_hkpr_samples.insert(v, one_over_r);
                }
            };
        }

        // FIXME: Never normalized
        let mut p: Vec<(usize, f64)> = aprox_hkpr_samples
            .keys()
            .map(|u| {
                (
                    *u,
                    *aprox_hkpr_samples.get(u).unwrap() / self.graph.node_degree(*u) as f64,
                )
            })
            .collect::<Vec<(usize, f64)>>();

        match self.graph.sweep_cut_over_diffusion_vector_by_conductance(
            p.as_mut(),
            Some(self.target_size),
            Some(self.target_vol),
        ) {
            Ok(c) => {
                println!(
                    "best community approxdirichlethkpr {{\n\tsize: {}\n\tvolume/width: {}\n\tconductance: {}\n}}",
                    c.size, c.width, c.conductance
                );
                Ok(c)
            }
            Err(e) => panic!("error: {}", e),
        }
    }
}
