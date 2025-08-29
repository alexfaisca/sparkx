#[allow(unused_imports)]
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
#[allow(unused_imports)]
use std::{hint::black_box, path::Path, time::Duration};
//
// use tool::{
//     generic_edge::{GenericEdge, GenericEdgeType, TinyEdgeType, TinyLabelStandardEdge},
//     generic_memory_map::{GraphCache, GraphMemoryMap},
//     trails::hierholzer::AlgoHierholzer,
//     test_common::{DATASETS, load_graph},
// };
//
// /// Build (or open) a graph for a given dataset path. For synthetic graphs, generate here deterministically instead.
// fn load_graph<EdgeType, Edge>(
//     dataset: &Path,
// ) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>>
// where
//     EdgeType: GenericEdgeType,
//     Edge: GenericEdge<EdgeType>,
// {
//     let cache = GraphCache::<EdgeType, Edge>::from_file(dataset, None, None, None)?;
//     GraphMemoryMap::init(cache, Some(16))
// }
//
// /// Run Hierholzer on an existing graph, writing its outputs into a per-iteration temporary cache directory (so runs don’t step on each other).
// fn run_hierholzer_once<'a, EdgeType, Edge>(
//     g: &'a GraphMemoryMap<EdgeType, Edge>,
// ) -> (usize, TempDir)
// where
//     EdgeType: GenericEdgeType,
//     Edge: GenericEdge<EdgeType>,
// {
//     // FIXME: Make a unique, auto-cleaned cache directory for this run.
//     // If your GraphMemoryMap lets you control its cache root, point it here.
//     let tmp = tempfile::tempdir().expect("tempdir");
//     // construct + compute
//     let algo = AlgoHierholzer::<'a, EdgeType, Edge>::new(g).expect("algo should succeed");
//     let trails = black_box(algo.trail_number());
//     // FIXME: is touching the output slice so the compiler can't elide writes necessary???
//     (trails, tmp)
// }
//
// fn bench_hierholzer_edges<EdgeType, Edge>(c: &mut Criterion, datasets: &[(&str, &Path)])
// where
//     EdgeType: GenericEdgeType + Send + Sync + 'static,
//     Edge: GenericEdge<EdgeType> + Send + Sync + 'static,
// {
//     let mut group = c.benchmark_group("hierholzer_edges");
//     // Optional: tighten stats for paper-quality numbers
//     group
//         // .measurement_time(Duration::from_secs(1540))
//         .warm_up_time(Duration::from_secs(5))
//         .sample_size(60)
//         .confidence_level(0.99)
//         .noise_threshold(0.01);
//     for (label, path) in datasets {
//         // Load/construct graph ONCE per input size; not in the timed body.
//         let graph = load_graph::<EdgeType, Edge>(path).expect("garph init should succeed");
//         // Bench throughput for |E|
//         group.throughput(Throughput::Elements(graph.width() as u64));
//         group.bench_with_input(BenchmarkId::from_parameter(*label), path, |b, _p| {
//             // Each measurement iteration gets a fresh cache dir and runs the algorithm once.
//             b.iter_batched(
//                 || &graph, // setup: borrow the already-built graph
//                 |g| {
//                     let (trails, _tmp) = run_hierholzer_once::<EdgeType, Edge>(g);
//                     black_box(trails);
//                     // FIXME: rm `*.tmp` here: clean this iteration's on-disk cache
//                 },
//                 BatchSize::PerIteration, // isolate IO/cache per sample
//             );
//         });
//     }
//
//     group.finish();
// }
//
fn criterion_hierholzer(_c: &mut Criterion) {
    let _datasets: &[(&str, &Path)] = &[
        // (
        //     "ggcat_1_5",
        //     Path::new("../ggcat/graphs/random_graph_1_5.lz4"),
        // ),
        // (
        //     "ggcat_2_5",
        //     Path::new("../ggcat/graphs/random_graph_2_5.lz4"),
        // ),
        // (
        //     "ggcat_3_5",
        //     Path::new("../ggcat/graphs/random_graph_3_5.lz4"),
        // ),
        // (
        //     "ggcat_4_5",
        //     Path::new("../ggcat/graphs/random_graph_4_5.lz4"),
        // ),
        // (
        //     "ggcat_5_5",
        //     Path::new("../ggcat/graphs/random_graph_5_5.lz4"),
        // ),
        // (
        //     "ggcat_6_5",
        //     Path::new("../ggcat/graphs/random_graph_6_5.lz4"),
        // ),
        // (
        //     "ggcat_7_5",
        //     Path::new("../ggcat/graphs/random_graph_7_5.lz4"),
        // ),
        // (
        //     "ggcat_8_5",
        //     Path::new("../ggcat/graphs/random_graph_8_5.lz4"),
        // ),
        // (
        //     "ggcat_9_5",
        //     Path::new("../ggcat/graphs/random_graph_9_5.lz4"),
        // ),
        // (
        //     "ggcat_1_10",
        //     Path::new("../ggcat/graphs/random_graph_1_10.lz4"),
        // ),
        // (
        //     "ggcat_2_10",
        //     Path::new("../ggcat/graphs/random_graph_2_10.lz4"),
        // ),
        // (
        //     "ggcat_3_10",
        //     Path::new("../ggcat/graphs/random_graph_3_10.lz4"),
        // ),
        // (
        //     "ggcat_4_10",
        //     Path::new("../ggcat/graphs/random_graph_4_10.lz4"),
        // ),
        // (
        //     "ggcat_5_10",
        //     Path::new("../ggcat/graphs/random_graph_5_10.lz4"),
        // ),
        // (
        //     "ggcat_6_10",
        //     Path::new("../ggcat/graphs/random_graph_6_10.lz4"),
        // ),
        // (
        //     "ggcat_7_10",
        //     Path::new("../ggcat/graphs/random_graph_7_10.lz4"),
        // ),
        // (
        //     "ggcat_8_10",
        //     Path::new("../ggcat/graphs/random_graph_8_10.lz4"),
        // ),
        // (
        //     "ggcat_9_10",
        //     Path::new("../ggcat/graphs/random_graph_9_10.lz4"),
        // ),
        // (
        //     "ggcat_8_15",
        //     Path::new("../ggcat/graphs/random_graph_8_15.lz4"),
        // ),
        // (
        //     "ggcat_9_15",
        //     Path::new("../ggcat/graphs/random_graph_9_15.lz4"),
        // ),
    ];

    // bench_hierholzer_edges::<TinyEdgeType, TinyLabelStandardEdge>(c, datasets);
}

criterion_group!(benches, criterion_hierholzer);
criterion_main!(benches);
