mod abstract_graph;
mod generic_memory_map;
mod node;
mod shared_slice;

use clap::Parser;
use core::panic;
use generic_memory_map::{
    DirectedEdge, EdgeOutOf, EulerTrail, GraphCache, GraphMemoryMap, OutEdgeRecord,
};
use node::EdgeType;
use petgraph::graphmap::DiGraphMap;
use rustworkx_core::petgraph::{self};
use rustworkx_core::{self};
use static_assertions::const_assert;
use std::fmt::Display;
use std::io::Write;
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
        (Some("txt"), _) => match args.mmap {
            true => mmapped_suite(
                match parse_bytes_mmaped(
                    read_file(args.file.clone(), InputType::Txt)
                        .expect("error couldn't read file")
                        .as_ref(),
                    args.threads,
                    args.output_id,
                ) {
                    Ok(i) => i,
                    Err(e) => panic!("error couldn't parse file, {:?}", e),
                },
            ),
            false => petgraph_suite(
                parse_bytes_petgraph(
                    read_file(args.file, InputType::Txt)
                        .expect("error couldn't read file")
                        .as_ref(),
                )
                .expect("error couldn't parse file"),
            ),
        },
        // .lz4 input file
        (Some("lz4"), _) => match args.mmap {
            true => mmapped_suite(
                parse_bytes_mmaped(
                    read_file(args.file.clone(), InputType::Lz4)
                        .expect("error couldn't read file")
                        .as_ref(),
                    args.threads,
                    args.output_id,
                )
                .expect("error couldn't parse file"),
            ),
            false => petgraph_suite(
                parse_bytes_petgraph(
                    read_file(args.file, InputType::Lz4)
                        .expect("error couldn't read file")
                        .as_ref(),
                )
                .expect("error couldn't parse file"),
            ),
        },
        // .mmap input file
        (Some("mmap"), Ok(false)) => match args.mmap {
            true => mmapped_suite(
                mmap_from_file(args.file.clone(), args.threads).expect("error couldn't read file"),
            ),
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

fn petgraph_suite(_graph: DiGraphMap<u64, EdgeType>) {}

fn mmapped_suite(_graph: generic_memory_map::GraphMemoryMap<OutEdgeRecord, DirectedEdge>) {}

// Experimentar simples, depois rustworkx, depois métodos mais eficientes

fn mmap_from_file(
    data_path: String,
    threads: u8,
) -> Result<GraphMemoryMap<OutEdgeRecord, DirectedEdge>, ParsingError> {
    let graph_cache: GraphCache<OutEdgeRecord> = GraphCache::<OutEdgeRecord>::open(data_path)?;
    let graph_mmaped = GraphMemoryMap::<OutEdgeRecord, DirectedEdge>::init(graph_cache, threads)?;

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */

    loop {
        print!("Enter something (empty to quit): ");
        io::stdout().flush().unwrap(); // Ensure prompt is shown

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        let input = input.trim_end(); // remove newline characters

        if input.is_empty() {
            break;
        }
        if let Ok(val) = graph_mmaped.node_id_from_kmer(input) {
            println!("Value for key {} is {}", input, val);
        } else {
            println!("Key {} not found", input);
        }
        println!("You entered: {}", input);
    }
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

fn parse_direction(orig: &str, dest: &str) -> Result<EdgeType, ParsingError> {
    match (orig, dest) {
        ("+", "+") => Ok(EdgeType::FF),
        ("+", "-") => Ok(EdgeType::FR),
        ("-", "+") => Ok(EdgeType::RF),
        ("-", "-") => Ok(EdgeType::RR),
        _ => panic!("error invalid direction: \"{}:{}\"", orig, dest),
    }
}

fn parse_bytes_petgraph(input: &[u8]) -> Result<DiGraphMap<u64, EdgeType>, ParsingError> {
    // let mut result: Vec<(u64, Node)> = vec![];
    let mut graph = DiGraphMap::<u64, EdgeType>::new();
    let mut lines = input.split(|&b| b == b'\n');

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
        let _k_mer = std::str::from_utf8(&sequence_line[0..])?;

        let node = line_str.split_whitespace().collect::<Vec<&str>>();

        let mut node = node.iter().peekable();
        let id: u64 = node.next().unwrap().parse()?;

        let _node_lengh = node.next(); // length
        let _node_color = node.next(); // color value

        for link in node {
            let link = &link.split(':').collect::<Vec<&str>>()[1..];
            graph.add_edge(id, link[1].parse()?, parse_direction(link[0], link[2])?);
        }
    }

    println!("{:?}", graph);

    Ok(graph)
}

fn parse_bytes_mmaped(
    input: &[u8],
    threads: u8,
    id: Option<String>,
) -> Result<GraphMemoryMap<OutEdgeRecord, DirectedEdge>, ParsingError> {
    // This assumes UTF-8 but avoids full conversion
    let mut lines = input.split(|&b| b == b'\n');
    let mut graph_cache = match id {
        Some(id) => generic_memory_map::GraphCache::<OutEdgeRecord>::init_with_id(id)?,
        None => generic_memory_map::GraphCache::<OutEdgeRecord>::init()?,
    };

    //debug
    // println!("graph cache: {:?}", graph_cache);

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
            edges.push(OutEdgeRecord::new(
                parse_direction(link_slice[0], link_slice[2])?,
                link_slice[1].parse()?,
            ));
        }

        edges.sort_unstable_by_key(|e| e.dest());

        graph_cache.write_node(id, edges.as_slice(), k_mer)?;
    }

    graph_cache.make_readonly()?;

    //debug
    // println!("graph cache: {:?}", graph_cache);

    let graph_mmaped: GraphMemoryMap<OutEdgeRecord, DirectedEdge> =
        GraphMemoryMap::<OutEdgeRecord, DirectedEdge>::init(graph_cache, threads)?;

    println!("graph {:?}", graph_mmaped.edges());
    println!("node 5: {:?}", graph_mmaped.neighbours(5)?);

    /* ********************************************************************************* */
    // Lookup test
    /* ********************************************************************************* */

    loop {
        print!("Enter something (empty to quit): ");
        io::stdout().flush().unwrap(); // Ensure prompt is shown

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        let input = input.trim_end(); // remove newline characters

        if input.is_empty() {
            break;
        }
        if let Ok(val) = graph_mmaped.node_id_from_kmer(input) {
            println!("Value for key {} is {}", input, val);
        } else {
            println!("Key {} not found", input);
        }
        println!("You entered: {}", input);
    }

    /* ********************************************************************************* */
    // End of lookup test
    /* ********************************************************************************* */

    let euler_trail = EulerTrail::new(graph_mmaped.clone())?;
    euler_trail.find_eulerian_cycle(2)?;
    graph_mmaped.compute_k_core_liu_et_al(5)?;
    graph_mmaped.k_truss_decomposition(5)?;
    Ok(graph_mmaped)
}

#[derive(Debug)]
enum ParsingError {
    Format(std::fmt::Error),
    Io(std::io::Error),
    Utf8(std::str::Utf8Error),
    ParseInt(std::num::ParseIntError),
}

impl From<std::fmt::Error> for ParsingError {
    fn from(e: std::fmt::Error) -> Self {
        ParsingError::Format(e)
    }
}

impl From<std::io::Error> for ParsingError {
    fn from(e: std::io::Error) -> Self {
        ParsingError::Io(e)
    }
}

impl From<std::str::Utf8Error> for ParsingError {
    fn from(e: std::str::Utf8Error) -> Self {
        ParsingError::Utf8(e)
    }
}

impl From<std::num::ParseIntError> for ParsingError {
    fn from(e: std::num::ParseIntError) -> Self {
        ParsingError::ParseInt(e)
    }
}

impl Display for ParsingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParsingError::Format(e) => write!(f, "FormatError {{{}}}", e),
            ParsingError::Io(e) => write!(f, "IoError {{{}}}", e),
            ParsingError::Utf8(e) => write!(f, "Utf8Error {{{}}}", e),
            ParsingError::ParseInt(e) => write!(f, "ParseIntError {{{}}}", e),
        }
    }
}
