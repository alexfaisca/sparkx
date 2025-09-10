use crate::{
    centralities::hyper_ball::HyperBall8,
    communities::{gve_louvain::AlgoGVELouvain, hk_relax::HKRelax},
    k_core::liu_et_al::AlgoLiuEtAl,
    k_truss::{burkhardt_et_al::AlgoBurkhardtEtAl, pkt::AlgoPKT},
    trails::hierholzer::AlgoHierholzer,
};

use super::GraphMemoryMap;

#[allow(dead_code)]
impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> GraphMemoryMap<N, E, Ix> {
    pub fn euler_trail(&self) -> Result<AlgoHierholzer<N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoHierholzer::new(self)
    }
    pub fn k_core(&self) -> Result<AlgoLiuEtAl<N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoLiuEtAl::new(self)
    }
    pub fn k_core_bz(&self) -> Result<AlgoBurkhardtEtAl<N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoBurkhardtEtAl::new(self)
    }
    pub fn k_core_lea(&self) -> Result<AlgoLiuEtAl<N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoLiuEtAl::new(self)
    }
    pub fn k_truss(&self) -> Result<AlgoBurkhardtEtAl<N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoBurkhardtEtAl::new(self)
    }
    pub fn k_truss_bea(&self) -> Result<AlgoBurkhardtEtAl<N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoBurkhardtEtAl::new(self)
    }
    pub fn k_truss_pkt(&self) -> Result<AlgoPKT<N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoPKT::new(self)
    }
    pub fn louvain(&self) -> Result<AlgoGVELouvain<N, E, Ix>, Box<dyn std::error::Error>> {
        AlgoGVELouvain::new(self)
    }
    pub fn hyperball(&self) -> Result<HyperBall8<N, E, Ix>, Box<dyn std::error::Error>> {
        HyperBall8::new(self)
    }
    pub fn hk_relax(
        &self,
        seed: usize,
        eps: f64,
    ) -> Result<HKRelax<N, E, Ix>, Box<dyn std::error::Error>> {
        HKRelax::new(self, 5., eps, vec![seed], None, None)
    }
}
