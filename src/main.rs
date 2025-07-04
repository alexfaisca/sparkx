mod abstract_graph;
mod generic_memory_map;
mod node;

use clap::Parser;
use core::panic;
use generic_memory_map::{DirectedEdge, GraphCache, GraphMemoryMap, OutEdgeRecord};
use node::EdgeType;
use petgraph::graphmap::DiGraphMap;
use rustworkx_core::petgraph;
use rustworkx_core::{self};
use static_assertions::const_assert;
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

#[derive(Debug)]
enum ParsingError {
    FormatError,
}

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
    #[arg(short, long)]
    mmap: bool,

    /// enable verbose mode
    #[arg(short, long)]
    verbose: bool,

    /// required — input file (.txt, .lz4, .mmap are accepted)
    #[arg(short, long)]
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
        Path::new(("index_".to_string() + args.file.as_str()).as_str()).try_exists(), // needs to
                                                                                      // be in
                                                                                      // cache dir
    ) {
        // .txt input file
        (Some("txt"), _) => match args.mmap {
            true => mmapped_suite(
                parse_bytes_mmaped(
                    read_file(args.file.clone(), InputType::Txt)
                        .expect("error couldn't read file")
                        .as_ref(),
                )
                .expect("error couldn't parse file"),
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
        (Some("mmap"), Ok(true)) => match args.mmap {
            true => {
                mmapped_suite(mmap_from_file(args.file.clone()).expect("error couldn't read file"))
            }
            false => panic!("error input file of type .mmap requires setting the -m --mmap flag"),
        },
        (Some("mmap"), Ok(false)) => {
            panic!(
                "error input file <filename>.mmap requires a valid .mmap index file with name \"index_<filename>.mmap\""
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
) -> Result<GraphMemoryMap<OutEdgeRecord, DirectedEdge>, ParsingError> {
    let graph_cache: GraphCache<OutEdgeRecord> = GraphCache::<OutEdgeRecord>::open(data_path)
        .expect("error couldn't build cache from memory");

    match GraphMemoryMap::<OutEdgeRecord, DirectedEdge>::init(graph_cache) {
        Ok(i) => Ok(i),
        Err(e) => panic!("error coudn't build memmapped graph from cache {}", e),
    }
}

fn read_file<P: AsRef<Path>>(path: P, mode: InputType) -> io::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

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

fn parse_bytes_petgraph(input: &[u8]) -> Result<DiGraphMap<u64, EdgeType>, ParsingError> {
    // let mut result: Vec<(u64, Node)> = vec![];
    let mut graph = DiGraphMap::<u64, EdgeType>::new();
    let mut lines = input.split(|&b| b == b'\n');

    while let Some(line) = lines.next() {
        if line.is_empty() {
            continue;
        }

        let _sequence_line = lines.next(); // ignore sequence value

        // Convert each line to str temporarily -> cut off ">" char
        let line_str =
            std::str::from_utf8(&line[1..line.len()]).map_err(|_| ParsingError::FormatError)?;
        let node = line_str.split_whitespace().collect::<Vec<&str>>();

        let mut node = node.iter().peekable();
        let id: u64 = match node.next().unwrap().parse() {
            Ok(i) => i,
            Err(i) => panic!("parse error invalid node id: {} \"{:?}\"", i, node.peek()),
        };

        let _node_lengh = node.next(); // length
        let _node_color = node.next(); // color value

        // Test 1
        // graph_index_mmaped
        //     .write_u64(byte)
        //     .expect("error couldn't write to index mmap");

        // parse and store links

        for link in node {
            let mut link1 = link.split(':').collect::<Vec<&str>>();
            link1.remove(0);
            let destiny_node: u64 = match link1[1].parse() {
                Ok(i) => match i > id {
                    true => i,
                    false => i,
                },
                Err(i) => panic!(
                    "parse error invalid destination node: {} \"{}\"",
                    i, link1[1]
                ),
            };
            let weight = match link1[0] {
                "+" => match link1[2] {
                    "+" => EdgeType::FF,
                    "-" => EdgeType::FR,
                    _ => panic!(
                        "parse error invalid destination direction: \"{}\"",
                        link1[2]
                    ),
                },
                "-" => match link1[2] {
                    "+" => EdgeType::RF,
                    "-" => EdgeType::RR,
                    _ => panic!(
                        "parse error invalid destination direction: \"{}\"",
                        link1[2]
                    ),
                },
                _ => panic!("parse error invalid origin direction: \"{}\"", link1[0]),
            };
            graph.add_edge(id, destiny_node, weight);
        }
        // debug
        // println!("node {}: {{\n\t{:?}\n}}", id, node_links);
    }

    println!("{:?}", graph);

    Ok(graph)
}

fn parse_bytes_mmaped(
    input: &[u8],
) -> Result<GraphMemoryMap<OutEdgeRecord, DirectedEdge>, ParsingError> {
    // This assumes UTF-8 but avoids full conversion
    let mut lines = input.split(|&b| b == b'\n');
    let mut graph_cache = generic_memory_map::GraphCache::<OutEdgeRecord>::init()
        .expect("error couldn't initialize memapped graph");

    //debug
    println!("graph cache: {:?}", graph_cache);

    while let Some(line) = lines.next() {
        if line.is_empty() {
            continue;
        }
        // Convert each line to str temporarily -> cut off ">" char
        let line_str =
            std::str::from_utf8(&line[1..line.len()]).map_err(|_| ParsingError::FormatError)?;

        let sequence_line = match lines.next() {
            None => panic!("error no k-mer sequence for node {}", line_str),
            Some(i) => i,
        };
        let k_mer = std::str::from_utf8(&sequence_line[0..sequence_line.len()])
            .map_err(|_| ParsingError::FormatError)?;
        // Convert each line to str temporarily -> cut off ">" char
        let line_str =
            std::str::from_utf8(&line[1..line.len()]).map_err(|_| ParsingError::FormatError)?;
        let node = line_str.split_whitespace().collect::<Vec<&str>>();

        let mut node = node.iter().peekable();
        let id: u64 = match node.next().unwrap().parse() {
            Ok(i) => i,
            Err(i) => panic!("parse error invalid node id: {} \"{:?}\"", i, node.peek()),
        };

        let _node_lengh = node.next(); // length
        let _node_color = node.next(); // color value

        let mut edges = vec![];
        for link in node {
            let mut link1 = link.split(':').collect::<Vec<&str>>();
            link1.remove(0);
            let destiny_node: u64 = match link1[1].parse() {
                Ok(i) => match i > id {
                    true => i,
                    false => i,
                },
                Err(i) => panic!(
                    "parse error invalid destination node: {} \"{}\"",
                    i, link1[1]
                ),
            };
            let weight = match link1[0] {
                "+" => match link1[2] {
                    "+" => EdgeType::FF,
                    "-" => EdgeType::FR,
                    _ => panic!(
                        "parse error invalid destination direction: \"{}\"",
                        link1[2]
                    ),
                },
                "-" => match link1[2] {
                    "+" => EdgeType::RF,
                    "-" => EdgeType::RR,
                    _ => panic!(
                        "parse error invalid destination direction: \"{}\"",
                        link1[2]
                    ),
                },
                _ => panic!("parse error invalid origin direction: \"{}\"", link1[0]),
            };
            edges.push(OutEdgeRecord::new(weight, destiny_node));
        }
        graph_cache
            .write_node(id, edges.as_slice(), k_mer)
            .expect("error couldn't write edge");
    }

    // debug
    graph_cache
        .make_readonly()
        .expect("error making cache readonly");

    //debug
    println!("graph cache: {:?}", graph_cache);

    let graph_mmaped: GraphMemoryMap<OutEdgeRecord, DirectedEdge> =
        GraphMemoryMap::<OutEdgeRecord, DirectedEdge>::init(graph_cache)
            .expect("error memmapping graph");
    println!("graph {:?}", graph_mmaped.edges());
    println!(
        "node 5: {:?}",
        graph_mmaped
            .neighbours(5)
            .expect("error couldn't read node")
    );

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
