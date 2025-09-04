use memmap2::MmapOptions;
#[allow(unused_imports)]
use tool::centralities::hyper_ball::*;
#[allow(unused_imports)]
use tool::communities::gve_louvain::AlgoGVELouvain;
#[allow(unused_imports)]
use tool::communities::{approx_dirichlet_hkpr::*, hk_relax::*};
#[allow(unused_imports)]
use tool::graph::{
    GenericEdge, GenericEdgeType, GraphMemoryMap,
    edge::{CompactLabel, StandardLabel, Test, TinyEdgeType, TinyLabelStandardEdge},
};
#[allow(unused_imports)]
use tool::k_core::{batagelj_zaversnik::*, liu_et_al::*};
#[allow(unused_imports)]
use tool::k_truss::{burkhardt_et_al::*, clustering_coefficient::*, pkt::*};
use tool::shared_slice::SharedSliceMut;
use tool::trails::bfs::BFSDists;
use tool::trails::dfs::DFS;
#[allow(unused_imports)]
use tool::trails::hierholzer::*;

use clap::{ArgAction, Parser};
#[allow(unused_imports)]
use hyperloglog_rs::prelude::*;
use static_assertions::const_assert;
use std::fmt::Display;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

// Checout matchtigs
// https://github.com/algbio/matchtigs

static BITS: usize = std::mem::size_of::<usize>();
static BITS_U64: usize = std::mem::size_of::<u64>();

const_assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u64>());

#[derive(Parser)]
#[command(
    name = "'The Tool'",
    version = "0.1",
    about = "Named 'The Tool'",
    long_about = "Very pretentious name. I know... ('~') but a man has to make an impression for his dissertation. (T.T)\n The contents of this crate should be a good start though. :)"
)]
struct ProgramArgs {
    /// Enable debugging mode
    #[arg(short, long)]
    debug: bool,

    /// Enable graph memory mapping mode
    #[arg(short, long, default_value_t = true)]
    mmap: bool,

    /// Program thread number, default is 1
    #[arg(short, long)]
    threads: Option<u8>,

    /// Enable or disable verbose output.
    ///
    /// Examples:
    ///   (not provided)   -> false
    ///   -v / --verbose   -> true
    ///   -v=true          -> true
    ///   -v=false         -> false
    #[arg(short, long, num_args(0..=1), default_value_t = false, default_missing_value = "true", action = ArgAction::Set)]
    verbose: bool,

    /// Output file(s) identifier (filenames are generated using this as the graph id)
    #[arg(short, long)]
    output_id: Option<String>,

    /// Required — input file (.txt, .lz4, .mmap are accepted)
    #[arg(short, long, required = true)]
    file: String,
}

fn main() {
    let args = ProgramArgs::parse();
    let _mmap: bool = args.mmap;
    let _verbose: bool = args.verbose;
    let _debug: bool = args.debug;

    // Set BRUIJNX_VERBOSE environment varioble (if = "1" then program operates in verbose mode
    // else program operates in non-verbose mode).
    unsafe { std::env::set_var("BRUIJNX_VERBOSE", if args.verbose { "1" } else { "0" }) };

    if BITS < BITS_U64 {
        panic!("error program can't operate on <64-bit systems");
    }

    // Check if <FILE> is of type .mmap, if so guarantee index file is provided
    // # monster match xD
    let _monster_match: () = match (
        Path::new(args.file.as_str())
            .extension()
            .and_then(|s| s.to_str()),
        Path::new(("./cache/index_".to_string() + args.file.as_str()).as_str()).try_exists(),
    ) {
        // .txt input file
        (Some("txt"), _) => {
            if args.mmap {
                mmapped_suite(
                    parse_bytes_mmaped::<CompactLabel, Test, String>(
                        args.file.clone(),
                        args.threads,
                        args.output_id,
                    )
                    .expect("error couldn't parse file"),
                );
            }
        }
        // .lz4 input file
        (Some("lz4"), _) => {
            if args.mmap {
                mmapped_suite(
                    parse_bytes_mmaped::<TinyEdgeType, TinyLabelStandardEdge, String>(
                        args.file.clone(),
                        args.threads,
                        args.output_id,
                    )
                    .expect("error couldn't parse file"),
                );
            }
        }
        // .mtx input file
        (Some("mtx"), _) => {
            if args.mmap {
                mmapped_suite(
                    parse_bytes_mmaped::<TinyEdgeType, TinyLabelStandardEdge, String>(
                        args.file.clone(),
                        args.threads,
                        args.output_id,
                    )
                    .expect("error couldn't parse file"),
                );
            }
        }
        // .nodes input file
        (Some("nodes"), _) => {
            if args.mmap {
                mmapped_suite(
                    parse_bytes_mmaped::<TinyEdgeType, TinyLabelStandardEdge, String>(
                        args.file.clone(),
                        args.threads,
                        args.output_id,
                    )
                    .expect("error couldn't parse file"),
                );
            }
        }
        // .edges input file
        (Some("edges"), _) => {
            if args.mmap {
                mmapped_suite(
                    parse_bytes_mmaped::<TinyEdgeType, TinyLabelStandardEdge, String>(
                        args.file.clone(),
                        args.threads,
                        args.output_id,
                    )
                    .expect("error couldn't parse file"),
                );
            }
        }
        // .mmap input file
        (Some("mmap"), Ok(false)) => match args.mmap {
            true => {}
            false => panic!("error input file of type .mmap requires setting the -m --mmap flag"),
        },
        (Some("mmap"), Ok(true)) => {
            panic!(
                "error input file <filename>.mmap requires a valid .mmap index file with name \"index_<filename>.mmap\": {}",
                "index_".to_string() + args.file.as_str()
            )
        }
        (Some("mmap"), Err(e)) => panic!("error couldn't find index file: {}", e),
        (a, b) => panic!("error invalid input file extension {:?} {:?}", a, b),
    };
}

fn mmapped_suite<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>(
    _graph: GraphMemoryMap<EdgeType, Edge>,
) {
}

// Experimentar simples, depois rustworkx, depois métodos mais eficientes
#[expect(dead_code)]
fn mmap_from_file<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>(
    data_path: String,
    threads: Option<u8>,
) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
    let graph_mmaped = GraphMemoryMap::<EdgeType, Edge>::open(&data_path, None, threads)?;

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */
    metalabel_search(&graph_mmaped);

    /* ********************************************************************************* */
    // End of lookup test
    /* ********************************************************************************* */

    Ok(graph_mmaped)
}

fn metalabel_search<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>(
    graph: &GraphMemoryMap<EdgeType, Edge>,
) {
    loop {
        print!("Enter something (press <ENTER> to quit): ");
        std::io::stdout().flush().unwrap(); // Ensure prompt is shown

        let mut input = String::new();
        std::io::stdin().read_line(&mut input).unwrap();

        let input = input.trim_end(); // remove newline characters

        if input.is_empty() {
            break;
        }
        match graph.node_id_from_metalabel(input) {
            Ok(val) => println!("Value for key {} is {}", input, val),
            Err(e) => println!("Key not found: {e}"),
        }
    }
}

fn parse_bytes_mmaped<
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
    P: AsRef<Path> + Clone,
>(
    path: P,
    threads: Option<u8>,
    id: Option<String>,
) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let mut graph_mmaped: GraphMemoryMap<EdgeType, Edge> =
        GraphMemoryMap::<EdgeType, Edge>::from_file(
            path.clone(),
            id,
            None,
            Some(|_| false),
            threads,
        )?;
    println!(
        "graph built (|V| = {:?}, |E| = {}) {:?}",
        graph_mmaped.size(),
        graph_mmaped.width(),
        time.elapsed()
    );
    println!();

    let time = Instant::now();
    let mut _pkt = AlgoPKT::new(&graph_mmaped)?;
    println!("k-truss pkt {:?}", time.elapsed());
    println!();
    _pkt.drop_cache()?;

    let time = Instant::now();
    let mut _burkhardt_et_al = AlgoBurkhardtEtAl::new(&graph_mmaped)?;
    println!("k-truss burkhardt et al {:?}", time.elapsed());
    println!();
    _burkhardt_et_al.drop_cache()?;

    // let time = Instant::now();
    // graph_cache.rebuild_fst_from_ggcat_file(path, None, None)?;
    // println!("cache fst rebuilt {:?}", time.elapsed());

    // let time = Instant::now();
    // let graph_mmaped: GraphMemoryMap<EdgeType, Edge> =
    //     GraphMemoryMap::<EdgeType, Edge>::init(graph_cache.clone(), threads)?;
    // println!("graph initialized {:?}", time.elapsed());
    // label_search(&graph_mmaped);

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */

    // metalabel_search(&graph_mmaped);

    /* ********************************************************************************* */
    // End of lookup test
    /* ********************************************************************************* */
    //

    // let e_fn = cache_file_name(
    //     &graph_mmaped.cache_index_filename(),
    //     FileType::ExactClosenessCentrality(H::H),
    //     None,
    // )?;
    // let node_count = graph_mmaped.size();
    // println!("compute closeness centrality {e_fn}",);
    // // let file = File::open(&e_fn)?;
    // // let vals = unsafe { MmapOptions::new().map(&file)? };
    // // let mut z = 0;
    // // for i in 0..node_count {
    // //     let val = unsafe { (vals.as_ptr() as *const f64).add(i).read() };
    // //     if val == 0. {
    // //         z += 1;
    // //     }
    // // }
    // // println!("found {z} zeroes");
    // let mut e = SharedSliceMut::<f64>::abst_mem_mut(&e_fn, node_count, true)?;
    // for u in 0..node_count {
    //     if u % 1000 == 0 {
    //         println!("reached {u} of {node_count}");
    //     }
    //     let bfs = BFSDists::new(&graph_mmaped, u)?;
    //     if bfs.recheable() <= 1 || bfs.total_distances() == 0. {
    //         println!("found isolated node at {u}");
    //         *e.get_mut(u) = 0.0;
    //     } else {
    //         *e.get_mut(u) = bfs.recheable() as f64 / bfs.total_distances();
    //     }
    // }
    // e.flush()?;
    // let mut i = 0;
    // loop {
    //     if i >= graph_mmaped.size() {
    //         break;
    //     }
    //     let time = Instant::now();
    //     let mut bfs = BFSDists::new(&graph_mmaped, i)?;
    //     println!(
    //         "source {i} -> bfs reacheable {} total_dist {}",
    //         bfs.recheable(),
    //         bfs.total_distances()
    //     );
    //     println!("bfs computed in {:?}", time.elapsed());
    //     println!();
    //     i += 10340;
    //     bfs.drop_cache()?;
    // }

    // let time = Instant::now();
    // let mut _dfs = DFS::new(&graph_mmaped, 0)?;
    // println!("dfs computed in {:?}", time.elapsed());
    //
    // let time = Instant::now();
    // let mut _euler_trail = AlgoHierholzer::new(&graph_mmaped)?;
    // println!("found {} euler trails", _euler_trail.trail_number());
    // println!("euler trail built {:?}", time.elapsed());
    // println!();
    // _euler_trail.drop_cache()?;

    // let time = Instant::now();
    // let mut hyperball = HyperBallInner::<_, _, Precision8, 6>::new(&graph_mmaped, None, None)?;
    // println!("hyperball {:?}", time.elapsed());
    // hyperball.drop_cache()?;

    // let time = Instant::now();
    // hyperball.compute_harmonic_centrality(None)?;
    // println!("harmonic centrality {:?}", time.elapsed());
    // println!();
    //
    // let mut i = 0;
    // while i < graph_mmaped.size() {
    //     let time = Instant::now();
    //     let hk_relax = HKRelax::new(&graph_mmaped, 45., 0.001, vec![i], None, None)?;
    //     let _ = hk_relax.compute()?;
    //     println!("HKRelax {:?}", time.elapsed());
    //     i += 1234600;
    // }
    //
    // let time = Instant::now();
    // let mut _louvain = AlgoGVELouvain::new(&graph_mmaped)?;
    // println!("found {} communities", _louvain.community_count());
    // println!("partition modularity {} ", _louvain.partition_modularity());
    // println!("louvain finished in {:?}", time.elapsed());
    // println!(
    //     "partition modularity according to graph_mmaped method {}",
    //     graph_mmaped.modularity(_louvain.communities(), _louvain.community_count())?
    // );
    // println!();
    // _louvain.drop_cache()?;
    //
    // let time = Instant::now();
    // let conductivity = ClusteringCoefficient::new(&graph_mmaped)?;
    // println!(
    //     "graph transitivity {:?}",
    //     conductivity.get_graph_transitivity()
    // );
    // println!(
    //     "average clustering coefficient {:?}",
    //     conductivity.get_average_clustering_coefficient()
    // );
    // println!("clustering coefficient finished in {:?}", time.elapsed());
    // println!();

    // let time = Instant::now();
    // let mut _liu_et_al = AlgoLiuEtAl::new(&graph_mmaped)?;
    // println!("k-core liu et al {:?}", time.elapsed());
    // println!();
    // _liu_et_al.drop_cache()?;

    println!("droping");
    graph_mmaped.drop_cache()?;
    println!("dropped");

    // let time = Instant::now();
    // let _bz = AlgoBatageljZaversnik::new(&graph_mmaped)?;
    // println!("k-core batagelj zaversnik {:?}", time.elapsed());

    // let mut i = 0;
    // while i < graph_mmaped.size().map_or(0, |s| s) {
    //     let time = Instant::now();
    //     let _approx_dirichlet_hkpr =
    //         ApproxDirHKPR::new(&graph_mmaped, 0.008, i, 100000, 4000000, 0.05)?;
    //     let _ = _approx_dirichlet_hkpr.compute()?;
    //     println!("ApproxDirichletHeatKernelK {:?}", time.elapsed());
    //     i += 12934600;
    // }
    //
    // let b = graph_mmaped.apply_mask_to_nodes(|u| -> bool { u % 2 == 0 }, Some("evennodes"))?;
    // println!("graph even nodes {:?}", b);

    // let time = Instant::now();
    // let a = graph_mmaped.export_petgraph_stripped()?;
    // println!("rustworkx_core export {:?} {:?}", a, time.elapsed());
    // let time = Instant::now();
    // use rustworkx_core::centrality::betweenness_centrality;
    // println!(
    //     "rustworkx_core export {:?} {:?}",
    //     betweenness_centrality(&a, false, true, 50),
    //     time.elapsed()
    // );
    Ok(graph_mmaped)
}

#[derive(Debug)]
enum ParsingError<T: std::error::Error> {
    Some(T),
}

impl<T: std::error::Error> From<T> for ParsingError<T> {
    fn from(e: T) -> Self {
        ParsingError::<T>::Some(e)
    }
}

impl<T: std::error::Error> Display for ParsingError<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParsingError::Some(e) => write!(f, "Other {{{}}}", e),
        }
    }
}
