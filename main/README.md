High-performance **Rust** library for large-scale **static sparse graphs analytics** built on a memory-mapped core. It includes fast implementations of:

- **k-core decomposition** (Batagelj–Zaversnik sequential; Liu et al. parallel)
- **k-truss decomposition** (Burkhardt et al. sequential; PKT parallel)
- **Louvain** (GVE-Louvain variant)
- **HyperBall** (HLL++ distance/centrality estimation)
- **Heat Kernel PageRank** (HK-relax --- SSPR/MSPR)
- Classic traversals/utilities (BFS/DFS, Hierholzer’s Euler trails, clustering coefficient, transitivity, triangle counting, etc.)

Comes with CSV **benchmark writers** and **plotting scripts** to visualize runtime scaling versus graph size.

---

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Data formats](#data-formats)
- [Quick start](#quick-start)
  - [Load & inspect](#load--inspect)
  - [Run algorithms](#run-algorithms)
- [Plotting results](#plotting-results)
- [Performance notes](#performance-notes)

---

## Features

- Zero-copy, **memory-mapped** graph store: `GraphMemoryMap<N, E, Ix>`
- Pluggable index types and optional node/edge labels
- Cache files + rich **metadata** (`Display` implemented)
- Algorithms expose summaries and can **drop** local caches
- Minimal, dependency-free **CSV appenders** to `results/`
- Ready-to-use **plotting** scripts (with Matplotlib) for reproducibility

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

# Reproducibility

## Memory Benchmarks

To obtain our memory benchmarking results we made use of a custom `cargo` target which launches the process and records the memory usage frame by frame using `valgrind --tool=massif`, it may be deployed for a given dataset `graph.mtx`, in directory `datasets/graphs/` at the library's base directory, by running:
```bash
cargo cache --tool=massif --dataset=../datasets/graphs/graph.mtx -t target_profile
```


### `target_profile` Options

| Value | Profile Name                          | Main Algorithms or Actions Included                                                                 |
|-------|--------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **0** | cache_profile_0 | Runs all heap independent algorithms with `--pages-as-heap=no` to verify heap independence. |
| **1** | cache_profile_1 | Runs **Hierholzer's Algorithm** with `--pages-as-heap=no` to measure heap memory usage. |
| **2** | cache_profile_2 | Runs **HKRelax** for multiple seeds with `t = 45` and `ε = 0.01` with `--pages-as-heap=no` to measure heap memory usage. |
| **3** | pages_profile_0 | Builds a graph from the given dataset with `--pages-as-heap=yes` to measure total memory usage. |
| **4** | pages_profile_1 | Runs **Hierholzer's Algorithm** with `--pages-as-heap=yes` to measure total memory usage. |
| **5** | pages_profile_2 | Computes the **k-cores** of a graph using **Batagelj & Zaversnik's Algorithm** with `--pages-as-heap=yes` to measure total memory usage. |
| **6** | pages_profile_3 | Computes the **k-cores** of a graph using **Liu et al.'s Algorithm** with `--pages-as-heap=yes` to measure total memory usage. |
| **7** | pages_profile_4 | Computes the **k-trusses** of a graph using **Burkhardt et al.'s Algorithm** with `--pages-as-heap=yes` to measure total memory usage. |
| **8** | pages_profile_5 | Computes the **k-trusses** of a graph using **PKT** with `--pages-as-heap=yes` to measure total memory usage. |
| **9** | pages_profile_6 | Computes the **Louvain partition** of a graph using **GVELouvain** with `--pages-as-heap=yes` to measure total memory usage. |
| **10** | pages_profile_7 | Runs **HyperBall** with precision 6 with `--pages-as-heap=yes` to measure total memory usage. |



## Wall-Time Benchmarks

For our wall-time benchmarks we made use of runtime target profiles enabled with `feature = "bench"`, they may be deployed for a given dataset `graph.mtx`, in directory `datasets/graphs/` at the library's base directory by running:
```bash
cargo run --features bench --release -- -t16 -v -mf ./datasets/graphs/graph.mtx -e target_profile
```


### `target_profile` Options

| Value | Profile Name                          | Main Algorithms or Actions Included                                                                 |
|-------|--------------------------------------------------|------------------------------------------------------------------------------------------------------|
| **0** | hyperball_profile | **HyperBall** validation --- the exact closeness centrality of every node of the graph is calculated and the Mean Absolute Error, Mean Absolute Percentual Error and Spearman's Rho of HyperBall's closeness centrality approximation are measured and logged. |
| **1** | hk_relax_profile | **HK-Relax** validation --- the algorithm is run 10'000 times with variable `t` and `ε` parameters for random seed nodes in single--source mode, and the resulting communities are then filtered, parametrized and logged. |
| **2** | general_profile | Runs all other algorithms multiple times, and parametrizes and logs each run. |

### Thread Count

To bench wall-time with different thread counts the `-t16` can be altered to change the number of threads to be used in parallel algorithms, if:
    *`-t0` or `-t1` are supplied, the program runs single-threaded.
    *`-tx` is supplied, the program runs with `x` threads.

# Plotting results
Doc still in construction...

# Performance notes
Doc still in construction...

