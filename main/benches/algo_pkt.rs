use criterion::{BatchSize, BenchmarkId, Criterion, Throughput};
use criterion_benches::{PerfRatio, emit_criterion_bench};

use tool::{
    generic_edge::{GenericEdge, GenericEdgeType, TinyEdgeType, TinyLabelStandardEdge},
    generic_memory_map::GraphMemoryMap,
    k_truss::pkt::AlgoPKT,
    test_common::{DATASETS, load_graph},
};

emit_criterion_bench!(
    time_throughput,
    branch_missprediction_rate,
    cache_miss_rate,
    fault_rate,
);

fn run_once<'a, EdgeType, Edge>(g: &'a GraphMemoryMap<EdgeType, Edge>)
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    let _algo = AlgoPKT::<'a, EdgeType, Edge>::new(g).expect("algo should succeed");
}

fn bench_time_throughput<EdgeType, Edge>(c: &mut Criterion, datasets: &[(&str, &str)])
where
    EdgeType: GenericEdgeType + Send + Sync + 'static,
    Edge: GenericEdge<EdgeType> + Send + Sync + 'static,
{
    let mut group = c.benchmark_group("time_and_edges_throughput_pkt");
    // Optional: tighten stats for paper-quality numbers
    group
        // .measurement_time(Duration::from_secs(1540))
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
                || &graph, // setup: borrow the already-built graph
                |g| {
                    run_once::<EdgeType, Edge>(g);
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
    let mut group = c.benchmark_group(group.to_string() + "_pkt");
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
