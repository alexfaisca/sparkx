#![cfg(target_os = "linux")]

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};

use tool::{
    generic_edge::{GenericEdge, GenericEdgeType, TinyEdgeType, TinyLabelStandardEdge},
    generic_memory_map::GraphMemoryMap,
    k_core::liu_et_al::AlgoLiuEtAl,
    test_common::{DATASETS, load_graph},
};

use criterion_benches::{self, PerfRatio, emit_criterion_bench};

emit_criterion_bench!(ipc, branch_missprediction_rate, cache_miss_rate, fault_rate,);

/// Run Liu et al. on an existing graph, writing its outputs into a per-iteration temporary cache directory (so runs don’t step on each other).
fn run_once<'a, EdgeType, Edge>(g: &'a GraphMemoryMap<EdgeType, Edge>)
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    let _algo = AlgoLiuEtAl::<'a, EdgeType, Edge>::new(g).expect("algo should succeed");
}

fn bench_time_throughput<EdgeType, Edge>(c: &mut Criterion, datasets: &[(&str, &str)])
where
    EdgeType: GenericEdgeType + Send + Sync + 'static,
    Edge: GenericEdge<EdgeType> + Send + Sync + 'static,
{
    let mut group = c.benchmark_group("time_edges_throughput_liu_et_al");
    // Optional: tighten stats for paper-quality numbers
    group
        .sample_size(60)
        .confidence_level(0.99)
        .noise_threshold(0.01);
    for (label, path) in datasets {
        // Load/construct graph ONCE per input size; not in the timed body.
        let graph = load_graph::<EdgeType, Edge, &str>(path).expect("graph init should succeed");
        // Bench throughput for |E|
        group.throughput(Throughput::Elements(graph.width() as u64));
        group.bench_with_input(BenchmarkId::from_parameter(*label), path, |b, _p| {
            // Each measurement iteration gets a fresh cache dir and runs the algorithm once.
            b.iter_batched(
                || {
                    let algo = AlgoLiuEtAl::<EdgeType, Edge>::new_no_compute(&graph)
                        .expect("algo should succeed");
                    let proc_mem = algo.init_cache_mem().expect("cache init should succeed");
                    (algo, proc_mem)
                }, // setup: borrow the already-built graph
                |(a, p)| {
                    let () = a.compute_with_proc_mem(p).expect("algo should succeed");
                },
                BatchSize::PerIteration, // isolate IO/cache per sample
            );
        });
    }

    group.finish();
}

fn bench_generic<EdgeType, Edge>(
    c: &mut Criterion<PerfRatio>,
    datasets: &[(&str, &str)],
    group: &str,
) where
    EdgeType: GenericEdgeType + Send + Sync + 'static,
    Edge: GenericEdge<EdgeType> + Send + Sync + 'static,
{
    let mut group = c.benchmark_group(group.to_string() + "_liu_et_al");
    group
        .sample_size(60)
        .confidence_level(0.99)
        .noise_threshold(0.01);

    for (label, path) in datasets {
        // Load/construct graph ONCE per input size; not in the timed body.
        let graph = load_graph::<EdgeType, Edge, &str>(path).expect("graph init should succeed");
        group.bench_with_input(BenchmarkId::from_parameter(label), path, |b, _p| {
            // Each measurement iteration gets a fresh cache dir and runs the algorithm once.
            b.iter_batched(
                || &graph, // setup: borrow the already-built graph
                |g| {
                    run_once::<EdgeType, Edge>(g);
                },
                BatchSize::PerIteration, // isolate IO/cache per sample
            );
        });
    }

    group.finish()
}
