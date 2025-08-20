use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::utils::{f64_is_nomal, f64_to_usize_safe};

use ordered_float::OrderedFloat;
use rand::{Rng, rng};
use rand_distr::{Distribution, Poisson};
use std::collections::HashMap;

/// For the computation of the *ApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O. on [`GraphMemoryMap`] instances.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[derive(Clone)]
pub struct ApproxDirHKPR<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    /// Graph for which the community is computed.
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
    /// Diffusion temperature.
    pub t: f64,
    /// Diffusion error (ε).
    pub eps: f64,
    /// Diffusion seed (starting node).
    pub seed: usize,
    /// Partition's target conductance.
    pub target_conductance: f64,
    /// Partition's target node number --- defaults to half the graph (tolerance up to full graph).
    pub target_size: usize,
    /// Partition's target edge number --- defaults to quarter of the graph (tolerance up to half graph).
    pub target_vol: usize,
}

/// Type of limiter to be used for the number of steps to take for each random walk.
#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub enum ApproxDirichletHeatKernelK {
    /// * Note: too stringent of a limitation -- results will be no good.
    None,
    /// * Note: optimization described in SolverApproxDirHKPR's algorithm.
    Mean,
    /// * Note: no limit --- infeasible for eps < 0.005 in large graphs.
    Unlim,
}

#[allow(dead_code)]
impl<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> ApproxDirHKPR<'a, EdgeType, Edge> {
    /// Evaluates parameters for the *ApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O.
    ///
    /// Evaluation is successful if `|V| >= 0`, `t` is normal and bigger than zero (not equal), `ε` is normal and (exclusive) between zero and one, `target_conductance` is normal and (exclusive) between zero and one, and `seed` is a valid node id, i.e. `0 <= seed < |V|`.
    ///
    /// # Arguments
    ///
    /// * `graph` --- the [`GraphMemoryMap`] instance for which the community is computed.
    /// * `seed_node` --- seed node.
    /// * `eps` --- ε (eps) error parameter.
    /// * `target_conductance` --- target conductance parameter.
    ///
    /// # Returns
    ///
    /// `Ok(())` if successful, or `Err(_)` if not.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    fn evaluate_params(
        graph: GraphMemoryMap<EdgeType, Edge>,
        seed_node: usize,
        eps: f64,
        target_conductance: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = match graph.size().overflowing_sub(1) {
            (_, true) => {
                return Err(
                    "error hk-relax invalid parameters: |V| == 0, the graph is empty".into(),
                );
            }
            (i, _) => {
                if i == 0 {
                    return Err(
                        "error hk-relax invalid parameters: actual |V| == 0, the graph is empty"
                            .into(),
                    );
                }
                i
            }
        };
        if !eps.is_normal() || eps <= 0f64 || eps >= 1f64 {
            return Err(format!(
                "error hk-relax invalid parameters: ε == {eps} doesn't satisfy 0.0 < ε 1.0",
            )
            .into());
        }
        if !target_conductance.is_normal()
            || target_conductance <= 0f64
            || target_conductance >= 1f64
        {
            return Err(format!(
                "error hk-relax invalid parameters: target_conductance == {} doesn't satisfy 0.0 < target_conductance < 1.0",
                target_conductance
            ).into());
        }
        if seed_node > node_count - 1 {
            return Err(format!(
                "error hk-relax invalid parameters: id(seed_nodes) == {} but max_id(v in V) == {}",
                seed_node,
                node_count - 1
            )
            .into());
        }
        Ok(())
    }

    /// Initializes the *ApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O.
    ///
    /// # Arguments
    ///
    /// * `graph` --- the [`GraphMemoryMap`] instance for which the community is computed.
    /// * `eps` --- ε (eps) error parameter.
    /// * `seed` --- seed node.
    /// * `target_size` --- partition's target node number.
    /// * `target_volume` --- partition's target edge number.
    /// * `target_conductance` --- partition's target conductance.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new(
        graph: &'a GraphMemoryMap<EdgeType, Edge>,
        eps: f64,
        seed: usize,
        target_size: usize,
        target_vol: usize,
        target_conductance: f64,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let () = Self::evaluate_params(graph.clone(), seed, eps, target_conductance)?;
        let t_formula = "(1. / target_conductance) * ln((2. * sqrt(target_vol)) / (1. - ε) + 2. * ε * target_size)";
        let t = f64_is_nomal(
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

    fn random_sample_poisson(
        lambda: f64,
        n: usize,
    ) -> Result<Vec<OrderedFloat<f64>>, Box<dyn std::error::Error>> {
        let mut rng = rng();
        let poisson = match Poisson::new(lambda) {
            Ok(dist) => dist,
            Err(e) => {
                return Err(format!(
                    "error approx-dirchlet-hk couldn't sample poission distribution: {e}",
                )
                .into());
            }
        };
        Ok(poisson
            .sample_iter(&mut rng)
            .take(n)
            .map(OrderedFloat)
            .collect())
    }

    #[inline(always)]
    fn random_neighbour(
        &self,
        deg_u: usize,
        u_n: NeighbourIter<EdgeType, Edge>,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let inv_deg_u = 1. / deg_u as f64;
        let random: f64 = rand::rng().random();
        let mut idx_plus_one = 1f64;
        for v in u_n {
            if idx_plus_one * inv_deg_u > random {
                return Ok(v.dest());
            }
            idx_plus_one += idx_plus_one;
        }
        Err("error approx-dirchlet-hk didn't find random neighbour".into())
    }

    fn random_walk_seed(
        &self,
        k: usize,
        seed_node: usize,
    ) -> Result<usize, Box<dyn std::error::Error>> {
        let mut curr_node = seed_node;
        for _ in 0..k {
            let (deg_u, u_n) = match self.graph.neighbours(curr_node) {
                Ok(u_n) => (u_n.remaining_neighbours(), u_n),
                Err(e) => {
                    return Err(format!(
                        "error approx-dirchlet-hk couldn't get neighbours for {curr_node}: {e}"
                    )
                    .into());
                }
            };
            if deg_u > 0 {
                curr_node = self.random_neighbour(deg_u, u_n)?;
            } else {
                return Ok(curr_node);
            }
        }
        Ok(curr_node)
    }

    /// Computes the *ApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O. with user controlled optimization level.
    ///
    /// # Arguments
    ///
    /// * `big_k` --- described the type of optimization to be used in random walk sampling.
    ///
    #[allow(clippy::unreachable)]
    #[deprecated]
    fn compute_specify_k(
        &self,
        big_k: ApproxDirichletHeatKernelK,
    ) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        let node_count = match self.graph.size().overflowing_sub(1) {
            (_, true) => return Err("error approx-dirchlet-hk |V| + 1 == 0".into()),
            (0, _) => return Err("error approx-dirchlet-hk |V| == 0".into()),
            (i, _) => i as f64,
        };

        let r = f64_is_nomal(
            (16. / self.eps.powi(2)) * f64::ln(node_count),
            "(16.0 / ε²) * ln(|V|)",
        )?;
        let one_over_r = f64_is_nomal(1. / r, "1. / r")?;

        let k = match big_k {
            ApproxDirichletHeatKernelK::None => f64_is_nomal(
                2. * f64::ln(1. / self.eps) / (f64::ln(f64::ln(1. / self.eps))),
                "2. * ln(1. / ε) / ln(ln(1. / ε))",
            )?,
            // optimization of SolverApproxDirHKPR
            ApproxDirichletHeatKernelK::Mean => f64_is_nomal(2. * self.t, "2. * t")?,
            ApproxDirichletHeatKernelK::Unlim => f64::INFINITY,
            #[expect(unreachable_patterns)]
            _ => {
                return Err(format!("error approx-dirchlet-hk unknown K {:?}", big_k).into());
            }
        };
        let k = OrderedFloat(k);
        println!(
            "k (ceil on sample value) computed to be {k}\nr (number of samples) computed to be {r}",
        );

        let num_samples: usize = match f64_to_usize_safe(r) {
            Some(s) => s,
            None => {
                return Err(format!("error approx-dirchlet-hk couldn't cast {r} to usize").into());
            }
        };
        let steps: Vec<OrderedFloat<f64>> = Self::random_sample_poisson(self.t, num_samples)?;
        let mut aprox_hkpr_samples: HashMap<usize, f64> = HashMap::new();

        for little_k in steps {
            let OrderedFloat(little_k) = std::cmp::min(little_k, k);
            let little_k_usize = match f64_to_usize_safe(little_k) {
                Some(val) => val,
                None => {
                    return Err(format!(
                        "error approx-dirchlet-hk couldn't cast {little_k} to usize"
                    )
                    .into());
                }
            };
            let v = self.random_walk_seed(little_k_usize, self.seed)?;
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
            Err(e) => Err(format!("error performing sweepcut: {e}").into()),
        }
    }

    /// Computes the *SolverApproxDirHKPR Algorithm* as described in ["Solving Local Linear Systems with Boundary Conditions Using Heat Kernel Pagerank"](https://doi.org/10.48550/arXiv.1503.03157) by Chung F. and Simpson O. with the therein described optimizations.
    ///
    pub fn compute(&self) -> Result<Community<usize>, Box<dyn std::error::Error>> {
        let node_count = match self.graph.size().overflowing_sub(1) {
            (_, true) => {
                return Err("error approx-dirchlet-hk |V| + 1 == 0".into());
            }
            (0, _) => return Err("error approx-dirchlet-hk |V| == 0".into()),
            (i, _) => i as f64,
        };

        let r = f64_is_nomal(
            (16. / self.eps.powi(2)) * f64::ln(node_count),
            "(16.0 / ε²) * ln(|V|)",
        )?;
        let one_over_r = f64_is_nomal(1. / r, "1. / r")?;

        let k = f64_is_nomal(2. * self.t, "2. * t")?;
        let k = OrderedFloat(k);
        println!(
            "k (ceil on sample value) computed to be {}\nr (number of samples) computed to be {}",
            k, r
        );

        let num_samples: usize = match f64_to_usize_safe(r) {
            Some(s) => s,
            None => {
                return Err(format!("error approx-dirchlet-hk couldn't cast {r} to usize").into());
            }
        };
        let steps: Vec<OrderedFloat<f64>> = Self::random_sample_poisson(self.t, num_samples)?;
        let mut aprox_hkpr_samples: HashMap<usize, f64> = HashMap::new();

        for little_k in steps {
            let OrderedFloat(little_k) = std::cmp::min(little_k, k);
            let little_k_usize = match f64_to_usize_safe(little_k) {
                Some(val) => val,
                None => {
                    return Err(format!(
                        "error approx-dirchlet-hk couldn't cast {little_k} to usize"
                    )
                    .into());
                }
            };
            let v = self.random_walk_seed(little_k_usize, self.seed)?;
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
            Err(e) => Err(format!("error performing sweep cut: {e}").into()),
        }
    }
}
