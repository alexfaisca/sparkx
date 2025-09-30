High-performance **Rust** library for large-scale static **graph analytics** built on a memory-mapped core. It includes fast implementations of:

- **k-core decomposition** (Batagelj–Zaversnik; Liu et al. parallel)
- **k-truss decomposition** (Burkhardt et al.; PKT)
- **Louvain** (GVE-Louvain variant)
- **HyperBall** (HLL/HLL++ distance/centrality estimation)
- **Heat Kernel PageRank** (HK-relax, single-source)
- Classic traversals/utilities (BFS/DFS, Hierholzer’s Euler trails, clustering coefficient, etc.)

Comes with CSV **benchmark writers** and **plotting scripts** to visualize runtime scaling versus graph size.

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Data formats](#data-formats)
- [Quick start](#quick-start)
- [Usage examples](#usage-examples)
  - [Load & inspect](#load--inspect)
  - [Run algorithms](#run-algorithms)
  - [Interactive metalabel lookup](#interactive-metalabel-lookup)
- [Benchmarking](#benchmarking)
  - [CSV writers](#csv-writers)
  - [End-to-end bench harness](#end-to-end-bench-harness)
- [Plotting results](#plotting-results)
- [Performance notes](#performance-notes)
- [Safety & caveats](#safety--caveats)
- [License](#license)
- [Citation](#citation)

---

## Features

- Zero-copy, **memory-mapped** graph store: `GraphMemoryMap<N, E, Ix>`
- Pluggable index types and optional node/edge labels
- Cache files + rich **metadata** (`Display` implemented)
- Algorithms expose summaries and can **drop** local caches
- Minimal, dependency-free **CSV appenders** to `results/`
- Ready-to-use **plotting** scripts (Matplotlib)

---

## Installation

Build in release mode (recommended):

```bash
cargo build --release
```

Enable non-default features:

```bash
cargo --features rayon build --release
```

# Data Formats

Input: **GGCAT Output Files** (`.lz4`), **Matrix Market** (`.mtx`) and cached internal formats (auto-created on first load).

## OOTB Datasets

A script to build datasets is provided for reproducibility and sandboxing purposes: `build_datasets.sh`. It builds all the datasets used for testing and benchmarking the library. It may be run in the following manner:

```bash
bash build_datasets.sh
```

It is possible to benchmark the library with datasets outside of those built with `build_datasets.sh`, for that an entry consisting of `<dataset_name>,<V>,<E>` should be addded to `datasets_size.csv` (`<dataset_name>` should be the dataset's filename without file extension):

```csv
dataset,V,E
kmer_V1r,222222,3333333
kmer_V2a,55042369,117217600
...
my_custom_dataset,123123,123123123
```

# Quick Start

To follow this guide it is advisable you first run the `build_datasets.sh` script. This way you will be able to follow along and use the provided code examples as they come.

## Load & inspect

```rust
use tool::graph::GraphMemoryMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // path to the dataset from which a graph is to be built
    let path = "./datasets/graphs/kmer_V2a.mtx";
    // build the graph
    let g: GraphMemoryMap<(), (), usize> =
        GraphMemoryMap::from_file(path, None, Some(32))?;
    println!("graph built (|V|={:?}, |E|={})", g.size(), g.width());
    println!("metadata:\n{}", g.metadata()?);
    // drop graph's cache entry
    g.drop_cache()?;
    Ok(())
}

```

## Run Algorithms

```rust
use tool::centralities::hyper_ball::*;
use tool::communities::gve_louvain::*;
use tool::communities::hk_relax::*;
use tool::graph;
use tool::graph::{E, GraphMemoryMap, IndexType, N, label::VoidLabel};
use tool::k_core::{batagelj_zaversnik::*, liu_et_al::*};
use tool::k_truss::{burkhardt_et_al::*, clustering_coefficient::*, pkt::*};
use tool::trails::hierholzer::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // path to the dataset from which a graph is to be built
    let path = "./datasets/graphs/kmer_V2a.mtx";
    // build the graph
    let g: GraphMemoryMap<(), (), usize> =
        GraphMemoryMap::from_file(path, None, Some(32))?;


    let mut pkt = AlgoPKT::new(&graph_mmaped)?;
    // get trussness of edge 0
    let kt = pkt.trussness(0);
    pkt.drop_cache()?;

    let mut burkhardt_et_al = AlgoBurkhardtEtAl::new(&graph_mmaped)?;
    // get trussness of all edges
    let kts = burkhardt_et_al.k_trusses();
    burkhardt_et_al.drop_cache()?;

    let mut bz = AlgoBatageljZaversnik::new(&graph_mmaped)?;
    // get coreness of node 0
    let c = bz.coreness(0);
    bz.drop_cache()?;

    let mut liu_et_al = AlgoLiuEtAl::new(&graph_mmaped)?;
    // get coreness of all nodes
    let cs = liu_et_al.k_cores();
    liu_et_al.drop_cache()?;

    let mut louvain = AlgoGVELouvain::new(&graph_mmaped)?;
    let number_communities_found = louvain.community_count();
    let partition_modularity = louvain.partition_modularity();
    // get community of node 0
    let com_0 = louvain.node_community(0);
    //get all communities
    let coms = louvain.communities();
    louvain.drop_cache()?;
    println!();

    let mut hyperball = HyperBallInner::<_, _, _, Precision6, 6>::new(&graph_mmaped)?;
    hyperball.drop_cache()?;

    // drop graph's cache entry
    g.drop_cache()?;
    Ok(())
}

```


