use std::{path::{Path, PathBuf}, sync::Arc};

use crossbeam::thread;
use num_cpus::get_physical;
use portable_atomic::{AtomicUsize, Ordering};

use crate::{graph::{GenericEdge, GenericEdgeType}, shared_slice::{AbstractedProceduralMemoryMut, SharedSlice}};

use super::{GraphCache, MultithreadedParserIndexBounds};

#[allow(dead_code)]
impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> GraphCache<EdgeType, Edge> {
    /// Parses a [`GGCAT`](https://github.com/algbio/ggcat) output file input into a [`GraphCache`] instance.
    ///
    /// Input file is assumed to have file extension .lz4, if provided in compressed form using LZ4, or .txt, if provided in plaintext form. Furthermore, the file contents must follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `path` --- input file[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    /// * `batch`--- size of input chunking for fst rebuild[^3][^4].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^5].
    ///
    /// [^1]: for example, a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if [`None`] is provided defaults to **NOT** storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    pub fn from_ggcat_file_impl<P: AsRef<Path>>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<GraphCache<EdgeType, Edge>, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;

        let path_str = path.as_ref().to_str().ok_or_else(|| -> Box<dyn std::error::Error> {format!("error getting path str from {:?}", path.as_ref()).into()})?;
        // parse optional inputs && fallback to defaults for the Nones found
        let id = id.map_or(path_str.to_string(), |id| id);
        let batching = Some(batch.map_or(Self::DEFAULT_BATCHING_SIZE, |b| b));
        let in_fst = in_fst.unwrap_or(|_id: usize| -> bool { false });
        let (input, tmp_path) = Self::read_input_file(path)?;

        // init cache
        let mut cache = Self::init_with_id(&id, batching)?;

        // parse and cache input
        cache.parallel_parse_ggcat_bytes_mmap(&input[..], in_fst)?;

        // if a tmp file was created delete it
        if let Some(p) = tmp_path {
            std::fs::remove_file(p)?;
        }

        // make cache readonly (for now only serves to allow clone() on instances)
        cache.make_readonly()?;

        Ok(cache)
    }

    /// Creates a [`GraphCache`] instance from an existing cached graph.
    ///
    /// # Arguments
    ///
    /// * `filename` --- filename of one of the graph's cache files[^1].
    /// * `batch` --- size of input chunking for fst rebuild[^2][^3].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^4].
    ///
    /// [^1]: may be from any of the graph's cached files, the graph's cache id is then extracted from the filename and all necessary filename's necessary for the [`GraphCache`] instance are inferred from the cache id.
    /// [^2]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^3]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^4]: if [`None`] is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    pub(super) fn rebuild_fst_from_ggcat_file_impl<P: AsRef<Path>>(
        &mut self,
        path: P,
        batch: Option<usize>,
        in_fst: Option<fn(usize) -> bool>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let batching = batch.map_or(Self::DEFAULT_BATCHING_SIZE, |b| b);
        let in_fst = in_fst.unwrap_or(|_id: usize| -> bool { true });

        // self.parallel_fst_from_ggcat_with_reader(path, batching, in_fst, get_physical())?;
        let (input, tmp) = Self::read_input_file(path)?;

        self.parallel_fst_from_ggcat_bytes(
            &input[..],
            batching,
            in_fst,
            num_cpus::get_physical(),
        )?;

        if let Some(p) = tmp {
            std::fs::remove_file(p)?;
        }
        // self.fst_from_ggcat_bytes(input.as_slice(), batching, in_fst)?;

        self.finish()?;

        Ok(())
    }

    /// Parses a ggcat output file input into a [`GraphCache`] instance.
    ///
    /// Input is assumed to be of type UTF-8 and to follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `input` - Input bytes.
    /// * `max_edges` - Ascribes a maximum number of edges for a node. If None is provided, defaults to `16`.
    /// * `in_fst` - A function that receives a usize as input and returns a bool as output. For every node id it should return false, if its kmer is not to be included in the graph's metalabel fst or true, vice-versa.
    ///
    /// # Returns
    ///
    /// Empty Ok().
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * In the input slice no k-mer / label sequence is found for some node.
    /// * An error occurs parsing a line of the input from UTF-8.
    /// * An error occurs parsing a node's identifier.
    /// * An error occurs parsing an edge's destiny node identifier or direction annototation.
    /// * An error occurs writing the node into the memmapped cache files.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    fn parse_ggcat_bytes_mmap(
        &mut self,
        input: &[u8],
        in_fst: fn(usize) -> bool,
    ) -> Result<(), Box<dyn std::error::Error>> {

        // this assumes UTF-8 but avoids full conversion
        let mut lines = input.split(|&b| b == b'\n');
        let mut edges = vec![];
        while let Some(line) = lines.next() {
            if line.is_empty() {
                continue;
            }
            // convert each line to str temporarily && cut off ">" char
            let line_str = std::str::from_utf8(&line[1..])?;
            let sequence_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("error no k-mer sequence for node {line_str}").into()
            })?;

            // convert each line to str temporarily -> cut off ">" char
            let line_str = std::str::from_utf8(&line[1..line.len()])?;
            let node = line_str.split_whitespace().collect::<Vec<&str>>();
            let mut node = node.iter().peekable();

            let id: usize = node.next().unwrap().parse()?;
            let _node_lengh = node.next(); // length 
            let _node_color = node.next(); // color value
            for link in node {
                let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                let dest: u64 = link_slice[1].parse()?;
                edges.push(Edge::new(
                    dest,
                    Self::parse_ggcat_direction(link_slice[0], link_slice[2])?,
                ));
            }
            edges.sort_unstable_by_key(|e| e.dest());

            if in_fst(id) {
                self.write_node(
                    id,
                    edges.as_slice(),
                    std::str::from_utf8(&sequence_line[0..])?,
                )?;
            } else {
                self.write_unlabeled_node(id, edges.as_slice())?;
            }
            edges.clear();
        }

        Ok(())
    }

    /// Parses a ggcat output edge direction
    ///
    /// # Arguments
    ///
    /// * `orig` - The edge's origin endpoint annotation. Must be either '+' or '-'.
    /// * `destiny` - The edge's destiny endpoint annotation. Must be either '+' or '-'.
    ///
    /// # Returns
    ///
    /// A `usize` that encodes the edge direction:
    /// * `0usize` --- for a forward-forward edge.
    /// * `1usize` --- for a forward-reverse edge.
    /// * `2usize` --- for a reverse-forward edge.
    /// * `3usize` --- for a reverse-reverse edge.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * An unknown annotation is given as input.
    ///
    pub(super) fn parse_ggcat_direction(
        orig: &str,
        dest: &str,
    ) -> Result<u64, Box<dyn std::error::Error>> {
        match (orig, dest) {
            ("+", "+") => Ok(0u64),
            ("+", "-") => Ok(1u64),
            ("-", "+") => Ok(2u64),
            ("-", "-") => Ok(3u64),
            _ => Err(format!("error ubknown edge direction annotations (supported are '+' && '-'): EdgeAnnotations {{orig: '{orig}', dest: '{dest}'}}").into()),
        }
    }

    fn parse_ggcat_builder_thread_bounds(input: &[u8], threads: usize) -> Result<MultithreadedParserIndexBounds, Box<dyn std::error::Error>> {
        let thread_load = input.len().div_ceil(threads);

        // figure out thread bounds 
        let mut bounds = vec![(0usize, 0usize); threads];
        let mut previous_end = 0usize;
        (0..threads).try_for_each(|tid| -> Result<(), Box<dyn std::error::Error>> {
            let begin = previous_end;
            if begin != input.len() && input[begin] != b'>' {
                return Err(format!("error getting threads bounds for thread {tid}: input[{{thread_begin: {begin}}}] = {} (should equal '>')", input[begin]).into());
            }
            let mut end = std::cmp::min((tid + 1) * thread_load, input.len());

            // find beginning of next node entry after end of slice (marked by '>')
            while end < input.len() && input[end] != b'>' {
                end += 1;
            }
            previous_end = end;
            bounds[tid] = (begin, end);
            Ok(())
        })?;

        Ok(bounds.into_boxed_slice())
    }

    fn parse_ggcat_max_node_id(input: &[u8]) -> Result<Option<usize>, Box<dyn std::error::Error>> {
        let mut begin = input.len().saturating_sub(1);
        let mut last_space = begin;

        while begin > 0 && input[begin] != b'>' {
            if input[begin] == b' ' {
                last_space = begin;
            }
            begin -= 1;
        }

        // only possible if input.len() == 0 or no '>' markers exist on the file 
        // in either case, file is considered empty
        if begin == input.len() || begin == 0 && input[begin] != b'>' {
            return Ok(None);
        }

        let it = input[begin + 1..last_space].iter().copied();
        let mut acc: usize = 0;
        let mut saw_digit = false;
        for b in it {
            if !b.is_ascii_digit() { return Err(format!("error parsing ggcat input file's last node index: {b} is not a valid digit").into()); }
            saw_digit = true;
            let d = (b - b'0') as usize;
            acc = acc.checked_mul(10).ok_or_else(|| -> Box<dyn std::error::Error> {format!("error parsing ggcat input file's last node index: {acc} * 10 overflowed").into()})?;
            acc = acc.checked_add(d).ok_or_else(|| -> Box<dyn std::error::Error> {format!("error parsing ggcat input file's last node index: {acc} + {d} overflowed").into()})?;
        }
        // get max node id
        if !saw_digit { return Err("error parsing ggcat input file's last node index: not one valid ascii digit was found".into()); }

        Ok(Some(acc))

    }

    /// Parses a ggcat output file input into a [`GraphCache`] instance.
    ///
    /// Input is assumed to be of type UTF-8 and to follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `input` - Input bytes.
    /// * `max_edges` - Ascribes a maximum number of edges for a node. If None is provided, defaults to `16`.
    /// * `in_fst` - A function that receives a usize as input and returns a bool as output. For every node id it should return false, if its kmer is not to be included in the graph's metalabel fst or true, vice-versa.
    ///
    /// # Returns
    ///
    /// Empty Ok().
    ///
    /// # Errors
    ///
    /// Returns an error if:
    ///
    /// * In the input slice no k-mer sequence is found for some node.
    /// * An error occurs parsing a line of the input from UTF-8.
    /// * An error occurs parsing a node's identifier.
    /// * An error occurs parsing an edge's destiny node identifier or direction annototation.
    /// * An error occurs writing the node into the memmapped cache files.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    fn parallel_parse_ggcat_bytes_mmap(
        &mut self,
        input: &[u8],
        in_fst: fn(usize) -> bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let threads = (get_physical() * 2).max(1);

        let bounds = Self::parse_ggcat_builder_thread_bounds(input, threads)?;
        let max_id = match Self::parse_ggcat_max_node_id(input)? {
            Some(id) => id,
            // not one id was found --- input file is empty, so graph cache has empty files
            None => return Ok(()),
        };

        let node_count = match max_id.overflowing_add(1) {
            (_, true) => return Err(format!("error getting ggcat input file's node count: {max_id} + 1 overflowed").into()),
            (r, false) => r
        };
        let offsets_size = match node_count.overflowing_add(1) {
            (_, true) => return Err(format!("error getting ggcat input file's offset size: {node_count} + 1 overflowed").into()),
            (r, false) => r
        };

        let mut offsets = AbstractedProceduralMemoryMut::<usize>::from_file(&self.index_file, offsets_size)?;

        // get node degrees
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let mut offsets = offsets.shared_slice();

                handles.push(s.spawn(move |_|  -> Result<(), Box<dyn std::error::Error + Send + Sync>> {

                    // this assumes UTF-8 but avoids full conversion
                    let mut lines = input.split(|&b| b == b'\n');
                    while let Some(line) = lines.next() {
                        // skip k-mer line
                        lines.next();
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        *offsets.get_mut(id) = node.len() - 2;
                    }
                    Ok(())
                }));
            });
            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting ggcat input file's node degrees (thread {tid}): {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        // prefix sum degrees to get offsets
        let mut sum = 0;
        for u in 0..node_count {
            let degree = *offsets.get(u);
            *offsets.get_mut(u) = sum;
            sum += degree;
        }
        *offsets.get_mut(node_count) = sum;
        offsets.flush()?;

        let edges = AbstractedProceduralMemoryMut::<Edge>::from_file(&self.graph_file, sum)?;
        let batch_num = Arc::new(AtomicUsize::new(0));
        let batch_size = self.batch.map_or(Self::DEFAULT_BATCHING_SIZE, |s| s);


        // write edges
        let batches = thread::scope(|s| -> Result<Box<[PathBuf]>, Box<dyn std::error::Error>> {
            let mut handles = Vec::with_capacity(threads);
            (0..threads).for_each(|tid| {
                let thread_bounds = bounds[tid];
                let input = &input[thread_bounds.0..thread_bounds.1];
                let offsets = offsets.shared_slice();
                let mut edges = edges.shared_slice();

                let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);
                let batch_num = batch_num.clone();
                let mut batches = Vec::new();
                let cache = &self;

                handles.push(s.spawn(move |_|  -> Result<Box<[PathBuf]>, Box<dyn std::error::Error + Send + Sync>> {
                    // this assumes UTF-8 but avoids full conversion 
                    let mut lines = input.split(|&b| b == b'\n');
                    let mut node_edges = Vec::with_capacity(u8::MAX as usize);
                    while let Some(line) = lines.next() {
                        if line.is_empty() {
                            continue;
                        }

                        // convert each line to str temporarily -> cut off ">" char
                        let line_str = std::str::from_utf8(&line[1..line.len()])?;
                        let label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error + Send + Sync> {
                            format!("error in thread {tid}: no k-mer sequence for node {line_str}").into()
                        })?;

                        let node = line_str.split_whitespace().collect::<Vec<&str>>();
                        let mut node = node.iter().peekable();

                        let id: usize = node.next().unwrap().parse()?;
                        let _node_lengh = node.next(); // length 
                        let _node_color = node.next(); // color value

                        for link in node {
                            let link_slice = &link.split(':').collect::<Vec<&str>>()[1..];
                            let dest: u64 = link_slice[1].parse()?;
                            node_edges.push(Edge::new(
                                    dest,
                                    Self::parse_ggcat_direction(
                                        link_slice[0], 
                                        link_slice[2])
                                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error in thread {tid}: {e}").into()})?,
                                    )
                                );
                        }
                        node_edges.sort_unstable_by_key(|e| e.dest());

                        if in_fst(id) {
                            current_batch.push((label_line, id as u64));
                            if current_batch.len() >= batch_size {
                                // println!("wrote {}", batch_num.load(Ordering::Relaxed));
                                let tmp_fst = cache.build_batch_fst(
                                    &mut current_batch,
                                    batch_num.fetch_add(1, Ordering::Relaxed),
                                    ).map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                        format!("error in thread {tid}: {e}").into()})?;
                                batches.push(tmp_fst);
                                current_batch.clear();
                            }
                        }
                        edges.write_slice(*offsets.get(id), node_edges.as_slice());

                        node_edges.clear();
                    }

                    // process the last batch if not empty
                    if !current_batch.is_empty() {
                        let tmp_fst = cache.build_batch_fst(
                            &mut current_batch,
                            batch_num.fetch_add(1, Ordering::Relaxed),
                            ).map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                format!("error in thread {tid}: {e}").into()}
                            )?;
                        batches.push(tmp_fst);
                    }

                    // return slice with all the tmp fst filenames
                    Ok(batches.into_boxed_slice())
                }));
            });
            let mut all_batches = Vec::new();

            // check for errors
            for (tid, r) in handles.into_iter().enumerate() {
                let thread_batches = r.join()
                    .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error getting node degrees from ggcat file (thread {tid}): {:?}", e).into()
                    })?;
                all_batches.extend(thread_batches);
            }
            Ok(all_batches.into_boxed_slice())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        edges.flush_async()?;
        drop(offsets);
        drop(edges);

        // if graph cache was used to build a graph then the graph's fst holds meatlabel_filename 
        // open, hence, it must be removed so that the new fst may be built
        std::fs::remove_file(&self.metalabel_filename)?;

        // merge all batch FSTs into option
        self.merge_fsts(&batches)?;

        // Cleanup temp batch files (not necessary because that is done in method finish())
        // for batch_file in batches {
        //     let _ = std::fs::remove_file(batch_file);
        // }

        self.graph_bytes = sum;
        self.index_bytes = offsets_size;

        self.finish()
    }

    fn process_chunk(
        &mut self,
        input: &[u8],
        end: usize,
        batch_num: Arc<AtomicUsize>,
        batch_size: usize,
        in_fst: fn(usize) -> bool,
    ) -> Result<Box<[PathBuf]>, Box<dyn std::error::Error>> {
        let mut begin_pos = 0;
        let mut end_pos = 0;

        let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);
        let mut batches = Vec::new();

        // find beginning of next node entry (marked by '>')
        while begin_pos < input.len() && input[begin_pos] != b'>' {
            begin_pos += 1;
        }

        // find beginning of next node entry after end of slice (marked by '\n>')
        while end + end_pos + 1 < input.len() && input[end + end_pos + 1] != b'>' {
            end_pos += 1;
        }

        // truncate input
        let input = &input[begin_pos..=std::cmp::min(input.len() - 1, end + end_pos)];

        let mut lines = input.split(|&b| b == b'\n');

        while let Some(line) = lines.next() {
            if line.is_empty() {
                continue;
            }

            // convert each line to str temporarily && cut off ">" char
            let line_str = std::str::from_utf8(&line[1..])?;
            let label_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("error no label for node {line_str}").into()
            })?;

            let line_str = std::str::from_utf8(&line[1..line.len()])?;
            let node = line_str.split_whitespace().collect::<Vec<&str>>();
            let id: usize = node.first().unwrap().parse()?;

            if in_fst(id) {
                current_batch.push((label_line, id as u64));

                if current_batch.len() >= batch_size {
                    // println!("wrote {}", batch_num.load(Ordering::Relaxed));
                    let tmp_fst = self.build_batch_fst(
                        &mut current_batch,
                        batch_num.fetch_add(1, Ordering::Relaxed),
                    )?;
                    batches.push(tmp_fst);
                    current_batch.clear();
                }
            }
        }

        // Process the last batch if not empty
        if !current_batch.is_empty() {
            let tmp_fst = self.build_batch_fst(
                &mut current_batch,
                batch_num.fetch_add(1, Ordering::Relaxed),
            )?;
            batches.push(tmp_fst);
        }

        Ok(batches.into_boxed_slice())
    }

    fn parallel_fst_from_ggcat_bytes(
        &mut self,
        input: &[u8],
        batch_size: usize,
        in_fst: fn(usize) -> bool,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if !self.readonly {
            return Err("error cache must be readonly to build fst in parallel".into());
        }

        let input_size = input.len();
        let chunk_size = input_size / threads;
        let batch_num = Arc::new(AtomicUsize::new(0));

        let shared_slice = SharedSlice::from_slice(input);

        let batches = thread::scope(|s| -> Result<Box<[PathBuf]>, Box<dyn std::error::Error>> {
            let mut thread_res = Vec::new();
            for i in 0..threads {
                let mut cache = self.clone();

                let batch = batch_num.clone();

                let start = std::cmp::min(i * chunk_size, input_size);
                let end = std::cmp::min(
                    start + chunk_size + Self::DEFAULT_CORRECTION_BUFFER_SIZE,
                    input_size,
                );

                thread_res.push(s.spawn(
                    move |_| -> Result<Box<[PathBuf]>, Box<dyn std::error::Error + Send + Sync>> {
                        let input = shared_slice.slice(start, end).ok_or_else(
                            || -> Box<dyn std::error::Error + Send + Sync> {
                                "error occured while chunking input in slices".into()
                            },
                        )?;

                        cache.process_chunk(input, chunk_size, batch, batch_size, in_fst)
                            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                                format!("error occured while batching fst build in parallel: {e}")
                                    .into()
                            })
                    },
                ));
            }

            let mut all_batches = Vec::new();
            for (idx, handle) in thread_res.into_iter().enumerate() {
                all_batches.extend(
                    handle
                        .join()
                        .map_err(|e| -> Box<dyn std::error::Error> {
                            format!("error joining thread {idx}: {:?}", e).into()
                        })?
                        .map_err(|e| -> Box<dyn std::error::Error> {
                            format!("error in thread {idx}: {:?}", e).into()
                        })?,
                );
            }
            Ok(all_batches.into_boxed_slice())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        // if graph cache was used to build a graph then the graph's fst
        //  holds meatlabel_filename open, hence, it must be removed so that the new fst may
        //  be built
        std::fs::remove_file(&self.metalabel_filename)?;

        // Now merge all batch FSTs into option
        self.merge_fsts(&batches)?;

        // Cleanup temp batch files
        for batch_file in batches {
            std::fs::remove_file(batch_file)?;
        }

        Ok(())
    }

    /// Parses a [`GGCAT`](https://github.com/algbio/ggcat) output file given as input and builds an fst for the graph according to the input.
    /// parameters.
    ///
    /// Input is assumed to be of type UTF-8 and to follow the format of [`GGCAT`](https://github.com/algbio/ggcat)'s output.
    ///
    /// # Arguments
    ///
    /// * `input` - Input bytes.
    /// * `in_fst` - A function that receives a usize as input and returns a bool as output. For every node id it should return false, if its kmer is not to be included in the graph's metalabel fst or true, vice-versa.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * In the input slice no k-mer / label sequence is found for some node.
    /// * An error occurs parsing a line of the input from UTF-8.
    /// * An error occurs parsing a node's identifier.
    /// * An error occurs parsing an edge's destiny node identifier or direction annototation.
    /// * An error occurs writing the node into the memmapped cache files.
    ///
    #[deprecated]
    fn fst_from_ggcat_bytes(
        &mut self,
        input: &[u8],
        batch_size: usize,
        in_fst: fn(usize) -> bool,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // this assumes UTF-8 but avoids full conversion

        let mut batch_num: usize = 0usize;

        let mut current_batch: Vec<(&[u8], u64)> = Vec::with_capacity(batch_size);
        let mut batches = Vec::new();

        let mut lines = input.split(|&b| b == b'\n');

        while let Some(line) = lines.next() {
            if line.is_empty() {
                continue;
            }
            // convert each line to str temporarily && cut off ">" char
            let line_str = std::str::from_utf8(&line[1..])?;

            let sequence_line = lines.next().ok_or_else(|| -> Box<dyn std::error::Error> {
                format!("error no k-mer sequence for node {line_str}").into()
            })?;

            let line = std::str::from_utf8(&line[1..])?;
            let node = line.split_whitespace().collect::<Vec<&str>>();
            let id: usize = node.first().unwrap().parse()?;

            if in_fst(id) {
                current_batch.push((sequence_line, id as u64));
                if current_batch.len() >= batch_size {
                    let tmp_fst = self.build_batch_fst(&mut current_batch, batch_num)?;
                    batches.push(tmp_fst);
                    current_batch.clear();
                    batch_num += 1;
                }
            }
        }

        // Process the last batch if not empty
        if !current_batch.is_empty() {
            let tmp_fst = self.build_batch_fst(&mut current_batch, batch_num)?;
            batches.push(tmp_fst);
        }

        // if graph cache was used to build a graph then the graph's fst
        // holds metalabel_filename open, hence, it must be removed so that the new fst may
        // be built
        std::fs::remove_file(&self.metalabel_filename)?;
        // Now merge all batch FSTs into one
        self.merge_fsts(&batches)?;

        // Cleanup temp batch files
        for batch_file in batches {
            std::fs::remove_file(batch_file)?;
        }

        Ok(())
    }
}
