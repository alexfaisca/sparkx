#![feature(int_roundings)]
mod abstract_graph;
mod generic_edge;
mod generic_memory_map;
mod shared_slice;

use clap::Parser;
use core::panic;
use generic_edge::{
    GenericEdge, GenericEdgeType, SubStandardColoredEdgeType, Test, TinyEdgeType,
    TinyLabelStandardEdge,
};
use generic_memory_map::{
    ApproxDirHKPR, EulerTrail, GraphCache, GraphMemoryMap, HKRelax, HyperBall,
};
use static_assertions::const_assert;
use std::fmt::Display;
use std::io::Write;
use std::time::Instant;
use std::{
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

// Checout matchtigs
// https://github.com/algbio/matchtigs

static BITS: usize = std::mem::size_of::<usize>();
static BITS_U64: usize = std::mem::size_of::<u64>();

const_assert!(std::mem::size_of::<usize>() >= std::mem::size_of::<u64>());

enum InputType {
    Txt,
    Lz4,
}

#[derive(Parser)]
#[command(name = "'The Tool'", version = "0.1", about = "Named 'The Tool'", long_about = None)]
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
                    parse_bytes_mmaped::<SubStandardColoredEdgeType, Test>(
                        read_file(args.file.clone(), InputType::Txt)
                            .expect("error couldn't read file")
                            .as_ref(),
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
                    parse_bytes_mmaped(
                        read_file(args.file.clone(), InputType::Lz4)
                            .expect("error couldn't read file")
                            .as_ref(),
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

fn read_file<P: AsRef<Path>>(path: P, mode: InputType) -> io::Result<Vec<u8>> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut contents = Vec::new();

    match mode {
        InputType::Lz4 => {
            // Decompress .lz4 using lz4 crate
            let mut decoder = lz4::Decoder::new(reader)?;
            decoder.read_to_end(&mut contents)?;
        }
        InputType::Txt => {
            reader.read_to_end(&mut contents)?;
        }
    };

    Ok(contents)
}

fn parse_direction(orig: &str, dest: &str) -> Result<TinyEdgeType, Box<dyn std::error::Error>> {
    match (orig, dest) {
        ("+", "+") => Ok(TinyEdgeType::FF),
        ("+", "-") => Ok(TinyEdgeType::FR),
        ("-", "+") => Ok(TinyEdgeType::RF),
        ("-", "-") => Ok(TinyEdgeType::RR),
        _ => panic!("error invalid direction: \"{}:{}\"", orig, dest),
    }
}

fn label_search<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>(
    graph: &GraphMemoryMap<EdgeType, Edge>,
) {
    loop {
        print!("Enter something (empty to quit): ");
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

fn parse_bytes_mmaped<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>>(
    input: &[u8],
    threads: u8,
    id: Option<String>,
) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
    // This assumes UTF-8 but avoids full conversion
    let mut lines = input.split(|&b| b == b'\n');
    let mut graph_cache = match id {
        Some(id) => {
            generic_memory_map::GraphCache::<EdgeType, Edge>::init_with_id(id, Some(20_000))?
        }
        None => generic_memory_map::GraphCache::<EdgeType, Edge>::init(Some(20_000))?,
    };

    //debug
    // println!("graph cache: {:?}", graph_cache);

    let time = Instant::now();
    while let Some(line) = lines.next() {
        if line.is_empty() {
            continue;
        }

        // Convert each line to str temporarily -> cut off ">" char
        let line_str = std::str::from_utf8(&line[1..])?;

        let sequence_line = match lines.next() {
            None => panic!("error no k-mer sequence for node {}", line_str),
            Some(i) => i,
        };
        let k_mer = std::str::from_utf8(&sequence_line[0..])?;

        // Convert each line to str temporarily -> cut off ">" char
        let line_str = std::str::from_utf8(&line[1..line.len()])?;
        let node = line_str.split_whitespace().collect::<Vec<&str>>();

        let mut node = node.iter().peekable();
        let id: usize = node.next().unwrap().parse()?;

        let _node_lengh = node.next(); // length
        let _node_color = node.next(); // color value

        let mut edges = vec![];

        for link in node {
            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
            edges.push(Edge::new(
                link_slice[1].parse()?,
                parse_direction(link_slice[0], link_slice[2])?.label() as u64,
            ));
        }

        edges.sort_unstable_by_key(|e| e.dest());

        if id == 0 {
            graph_cache.write_node(id, edges.as_slice(), k_mer)?;
        } else {
            graph_cache.write_unlabeled_node(id, edges.as_slice())?;
        }
    }

    println!("Graph built {:?}", time.elapsed());
    let time = Instant::now();
    graph_cache.make_readonly()?;
    println!("Fst built {:?}", time.elapsed());

    //debug
    // println!("graph cache: {:?}", graph_cache);

    let graph_mmaped: GraphMemoryMap<EdgeType, Edge> =
        GraphMemoryMap::<EdgeType, Edge>::init(graph_cache, threads)?;

    println!("graph {:?}", graph_mmaped.edges());
    println!("node 5: {:?}", graph_mmaped.neighbours(5)?);

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */

    label_search(&graph_mmaped);
    loop {
        print!("Enter something (empty to quit): ");
        io::stdout().flush().unwrap(); // Ensure prompt is shown

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        let input = input.trim_end(); // remove newline characters

        if input.is_empty() {
            break;
        }
        match graph_mmaped.node_id_from_kmer(input) {
            Ok(val) => println!("Value for key {} is {}", input, val),
            Err(e) => println!("Key not found: {e}"),
        }
    }

    /* ********************************************************************************* */
    // End of lookup test
    /* ********************************************************************************* */

    let time = Instant::now();
    let euler_trail = EulerTrail::new(graph_mmaped.clone())?;
    euler_trail.find_eulerian_cycle(2)?;
    println!("euler trail built {:?}", time.elapsed());
    let time = Instant::now();
    graph_mmaped.compute_k_core_bz(5)?;
    println!("k-core bz {:?}", time.elapsed());
    let time = Instant::now();
    graph_mmaped.compute_k_core_liu_et_al(5)?;
    println!("k-core liu et al {:?}", time.elapsed());
    let time = Instant::now();
    graph_mmaped.pkt(7)?;
    println!("pkt {:?}", time.elapsed());
    let time = Instant::now();
    graph_mmaped.k_truss_decomposition(7)?;
    println!("k-truss decomposition {:?}", time.elapsed());
    let time = Instant::now();
    let hk_relax = HKRelax::new(graph_mmaped.clone(), 20., 0.0001, vec![64256])?;
    let _ = hk_relax.compute()?;
    println!("HKRelax {:?}", time.elapsed());
    let time = Instant::now();
    let _approx_dirichlet_hkpr =
        ApproxDirHKPR::new(graph_mmaped.clone(), 0.008, 8, 100000, 4000000, 0.05)?;
    let _ = _approx_dirichlet_hkpr.compute(generic_memory_map::ApproxDirichletHeatKernelK::Mean)?;
    println!("ApproxDirichletHeatKernelK {:?}", time.elapsed());
    let time = Instant::now();
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
    let mut hyper = HyperBall::new(graph_mmaped.clone(), Some(6), Some(70))?;
    println!("hyperball {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_harmonic_centrality(Some(true))?;
    println!("centrality harmonic {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_closeness_centrality(Some(true))?;
    println!("centrality closeness {:?}", time.elapsed());
    let time = Instant::now();
    hyper.compute_lins_centrality()?;
    println!("centrality lin {:?}", time.elapsed());
    // hyper.compute_harmonic_centrality(Some(true))?;
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
