#![feature(int_roundings)]
mod abstract_graph;
mod generic_edge;
mod generic_memory_map;
mod shared_slice;
mod utils;

#[allow(unused_imports)]
use generic_edge::{
    GenericEdge, GenericEdgeType, SubStandardColoredEdgeType, Test, TinyEdgeType,
    TinyLabelStandardEdge,
};

use clap::Parser;
use core::panic;
#[allow(unused_imports)]
use generic_memory_map::{
    ApproxDirHKPR, EulerTrail, GraphCache, GraphMemoryMap, HKRelax, HyperBallInner,
};
use static_assertions::const_assert;
use std::fmt::Display;
use std::io::Write;
use std::time::Instant;
use std::{io, path::Path};

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
    long_about = "Very pretentious name. I know... ('~') but a man has to make an impact for his dissertation. (T.T)\n The contents of this crate should be a good start though. :)"
)]
struct ProgramArgs {
    /// enable debugging mode
    #[arg(short, long)]
    debug: bool,

    /// enable graph memory mapping mode
    #[arg(short, long, default_value_t = true)]
    mmap: bool,

    /// program thread number, default is 1
    #[arg(short, long, default_value_t = 1)]
    threads: u8,

    /// enable verbose mode
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    /// output file(s) identifier (filenames are generated using this as the graph id)
    #[arg(short, long)]
    output_id: Option<String>,

    /// required — input file (.txt, .lz4, .mmap are accepted)
    #[arg(short, long, required = true)]
    file: String,
}

fn main() {
    let args = ProgramArgs::parse();
    let _mmap: bool = args.mmap;
    let _verbose: bool = args.verbose;
    let _debug: bool = args.debug;

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
                    parse_bytes_mmaped::<SubStandardColoredEdgeType, Test, String>(
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
                    parse_bytes_mmaped::<SubStandardColoredEdgeType, Test, String>(
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

fn mmapped_suite(_graph: generic_memory_map::GraphMemoryMap<SubStandardColoredEdgeType, Test>) {}

// Experimentar simples, depois rustworkx, depois métodos mais eficientes
#[expect(dead_code)]
fn mmap_from_file<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>(
    data_path: String,
    threads: u8,
) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
    let graph_cache: GraphCache<EdgeType, Edge> =
        GraphCache::<EdgeType, Edge>::open(data_path, None)?;
    let graph_mmaped = GraphMemoryMap::<EdgeType, Edge>::init(graph_cache, threads)?;

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */
    label_search(&graph_mmaped);

    /* ********************************************************************************* */
    // End of lookup test
    /* ********************************************************************************* */

    Ok(graph_mmaped)
}

fn label_search<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>(
    graph: &GraphMemoryMap<EdgeType, Edge>,
) {
    loop {
        print!("Enter something (press <ENTER> to quit): ");
        io::stdout().flush().unwrap(); // Ensure prompt is shown

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        let input = input.trim_end(); // remove newline characters

        if input.is_empty() {
            break;
        }
        match graph.node_id_from_kmer(input) {
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
    threads: u8,
    id: Option<String>,
) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let time = Instant::now();
    let graph_cache = generic_memory_map::GraphCache::<EdgeType, Edge>::from_ggcat_file(
        path.clone(),
        id,
        None,
        None,
    )?;
    println!("cache no fst built {:?}", time.elapsed());
    let time = Instant::now();
    let graph_mmaped: GraphMemoryMap<EdgeType, Edge> =
        GraphMemoryMap::<EdgeType, Edge>::init(graph_cache.clone(), threads)?;
    println!(
        "graph initialized (|V| = {}, |E| = {}) {:?}",
        graph_mmaped.size() - 1,
        graph_mmaped.width(),
        time.elapsed()
    );
    // label_search(&graph_mmaped);
    // let time = Instant::now();
    // graph_cache.rebuild_fst_from_ggcat_file(path, None, None)?;
    // println!("cache fst built {:?}", time.elapsed());
    // let time = Instant::now();
    // let graph_mmaped: GraphMemoryMap<EdgeType, Edge> =
    //     GraphMemoryMap::<EdgeType, Edge>::init(graph_cache.clone(), threads)?;
    // println!("graph initialized {:?}", time.elapsed());

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */
    let time = Instant::now();
    let mut hyper = HyperBallInner::new(graph_mmaped.clone(), Some(6), Some(70))?;
    println!("hyperball {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_harmonic_centrality(None)?;
    println!("centrality harmonic None {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_harmonic_centrality(Some(false))?;
    println!("centrality harmonic false {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_harmonic_centrality(Some(true))?;
    println!("centrality harmonic true {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_closeness_centrality(None)?;
    println!("centrality closeness None {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_closeness_centrality(Some(false))?;
    println!("centrality closeness false {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_closeness_centrality(Some(true))?;
    println!("centrality closeness true {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_lins_centrality()?;
    println!("centrality lin {:?}", time.elapsed());

    // label_search(&graph_mmaped);

    /* ********************************************************************************* */
    // End of lookup test
    /* ********************************************************************************* */

    // let time = Instant::now();
    // let euler_trail = EulerTrail::new(graph_mmaped.clone())?;
    // euler_trail.find_eulerian_cycle(2)?;
    // println!("euler trail built {:?}", time.elapsed());
    let time = Instant::now();
    graph_mmaped.compute_k_core_bz(5)?;
    println!("k-core bz {:?}", time.elapsed());
    let time = Instant::now();
    graph_mmaped.compute_k_core_liu_et_al(5)?;
    println!("k-core liu et al {:?}", time.elapsed());
    let time = Instant::now();
    graph_mmaped.k_truss_decomposition(7)?;
    println!("k-truss decomposition {:?}", time.elapsed());
    let time = Instant::now();
    graph_mmaped.pkt(7)?;
    println!("pkt {:?}", time.elapsed());
    let time = Instant::now();
    let hk_relax = HKRelax::new(
        graph_mmaped.clone(),
        20.,
        0.0001,
        vec![64256],
        Some(10_000),
        Some(50_000),
    )?;
    let _ = hk_relax.compute()?;
    println!("HKRelax {:?}", time.elapsed());
    let time = Instant::now();
    let _approx_dirichlet_hkpr =
        ApproxDirHKPR::new(graph_mmaped.clone(), 0.008, 8, 100000, 4000000, 0.05)?;
    let _ = _approx_dirichlet_hkpr.compute(generic_memory_map::ApproxDirichletHeatKernelK::Mean)?;
    println!("ApproxDirichletHeatKernelK {:?}", time.elapsed());
    // let time = Instant::now();
    // let _ = graph_mmaped.cleanup_cache();
    // let b = graph_mmaped
    //     .apply_mask_to_nodes(|u| -> bool { u % 2 == 0 }, Some("evennodes".to_string()))?;
    // println!("graph even nodes {:?}", b);
    //
    // println!("graph {:?}", b.edges());
    // println!("node 0: {:?}", b.neighbours(0)?.collect::<Vec<Test>>());
    // println!("node 1: {:?}", b.neighbours(1)?.collect::<Vec<Test>>());
    // println!("node 2: {:?}", b.neighbours(2)?.collect::<Vec<Test>>());
    // println!("node 3: {:?}", b.neighbours(3)?.collect::<Vec<Test>>());
    // println!("node 4: {:?}", b.neighbours(4)?.collect::<Vec<Test>>());
    // println!("node 5: {:?}", b.neighbours(5)?.collect::<Vec<Test>>());

    // let a = graph_mmaped.export_petgraph()?;
    // println!("rustworkx_core export {:?}", a);
    // use rustworkx_core::centrality::betweenness_centrality;
    // println!(
    //     "rustworkx_core export {:?}",
    //     betweenness_centrality(&a, false, true, 50)
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
