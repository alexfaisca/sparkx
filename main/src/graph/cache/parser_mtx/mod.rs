use crate::{
    graph::cache::utils::{FileType, H, cache_file_name},
    shared_slice::SharedSliceMut,
};

use crossbeam::thread;
use num_cpus::get_physical;
use portable_atomic::{AtomicUsize, Ordering};
use std::{
    fs::File,
    io::{BufRead, BufReader, Seek, SeekFrom},
    path::Path,
    sync::Arc,
};

use super::GraphCache;

#[allow(dead_code)]
impl<N: super::N, E: super::E, Ix: super::IndexType> GraphCache<N, E, Ix> {
    /// Parses a [`MatrixMarket`](https://math.nist.gov/MatrixMarket/formats.html) file input into a [`GraphCache`] instance.
    ///
    /// Input file is assumed have file extension .mtx and must be of type matrix coordinate pattern/integer symmetric/skew-symmetric/general.
    ///
    /// # Arguments
    ///
    /// * `path` - input file[^1].
    /// * `id` --- graph cache id for the [`GraphCache`] instance[^2].
    /// * `batch` --- size of input chunking for fst rebuild[^3][^4].
    /// * `in_fst` --- closure to be applied on each entry's node id to determine if the entry's metalabel-to-node pair is stored in the fst[^5].
    ///
    /// [^1]: for example, as a [`String`].
    /// [^2]: if [`None`] is provided defaults to a random generated cache id, which may later be retrieved trhough the provided getter method.
    /// [^3]: if [`None`] is provided defaults to [`DEFAULT_BATCHING_SIZE`].
    /// [^4]: given a `batch_size` `n`, finite state transducers of size up to `n` are succeedingly built until no more unprocessed entries remain, at which point all the partial fsts are merged into the final resulting general fst.
    /// [^5]: if [`None`] is provided defaults to storing every node's label.
    ///
    /// [`GraphCache`]: ./struct.GraphCache.html#
    /// [`DEFAULT_BATCHING_SIZE`]: ./struct.GraphCache.html#associatedconstant.DEFAULT_BATCHING_SIZE
    pub(super) fn from_mtx_file_impl<P: AsRef<Path>>(
        path: P,
        id: Option<String>,
        batch: Option<usize>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        Self::guarantee_caching_dir()?;
        let threads = (get_physical() * 2).max(1);

        let id = id.unwrap_or(
            path.as_ref()
                .to_str()
                .ok_or_else(|| -> Box<dyn std::error::Error> {
                    "error getting path as string".into()
                })?
                .to_string(),
        );

        let (node_count, _, _) = Self::parse_mtx_header(path.as_ref())?;

        let n_fn = Self::init_cache_file_from_id_or_random(&id, FileType::Edges(H::H), None)?;
        let o_fn = Self::init_cache_file_from_id_or_random(&id, FileType::Index(H::H), None)?;
        let h_fn = cache_file_name(&n_fn, FileType::Helper(H::H), Some(0))?;
        let offset_size = match node_count.overflowing_add(1) {
            (_, true) => {
                return Err(
                    format!("error calculating offsets size: {node_count} + 1 overflowed").into(),
                );
            }
            (r, false) => r,
        };
        let offsets = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&o_fn, offset_size, true)?;
        let counters = SharedSliceMut::<AtomicUsize>::abst_mem_mut(&h_fn, node_count, true)?;

        // accumulate node degrees on index
        let edge_count = Arc::new(AtomicUsize::new(0));
        {
            Self::parallel_parse_mtx_with(
                path.as_ref(),
                {
                    let edge_count = edge_count.clone();
                    let offsets = offsets.shared_slice().clone();
                    move |u, _v, _w| {
                        edge_count.add(1, Ordering::Relaxed);
                        offsets.get(u).add(1, Ordering::Relaxed);
                    }
                },
                |_, _| {},
                threads,
            )?;
        }
        let mut neighbors =
            SharedSliceMut::<usize>::abst_mem_mut(&n_fn, edge_count.load(Ordering::Relaxed), true)?;
        // build offset vector from degrees
        let mut sum = 0;
        let mut max_degree = 0;
        // this works because after aloc memmaped files are zeroed (so index[nr] = 0)
        for u in 0..offset_size {
            let deg_u = offsets.get(u).load(Ordering::Relaxed);
            if deg_u > max_degree {
                max_degree = deg_u;
            }
            offsets.get(u).store(sum, Ordering::Relaxed);
            sum += deg_u;
        }
        // println!(
        //     "|V| == {}, |E| == {}",
        //     node_count,
        //     index.get(node_count).load(Ordering::Relaxed)
        // );

        if max_degree >= u8::MAX as usize {
            return Err(format!("Error graph has a max_degree of {max_degree} which, unforturnately, is bigger than {}, our current maximum supported size. If you feel a mistake has been made or really need this feature, please contact the developer team. We sincerely apologize.", u8::MAX).into());
        }

        // let firsts_atomic: Arc<Vec<AtomicUsize>> =
        //     Arc::new((0..threads).map(|_| AtomicUsize::new(usize::MAX)).collect());

        Self::parallel_parse_mtx_with(
            path.as_ref(),
            {
                let mut neighbors = neighbors.shared_slice();
                let offsets = offsets.shared_slice();
                let counters = counters.shared_slice();
                // move |u, v, w| {
                //     *edges.get_mut(
                //         index.get(u).load(Ordering::Relaxed)
                //             + offsets.get(u).fetch_add(1, Ordering::Relaxed),
                //     ) = Edge::new(v, w);
                // }
                move |u, v, _w| {
                    *neighbors.get_mut(
                        offsets.get(u).load(Ordering::Relaxed)
                            + counters.get(u).fetch_add(1, Ordering::Relaxed),
                    ) = v as usize;
                }
            },
            {
                // let firsts_atomic = Arc::clone(&firsts_atomic);
                // move |tid, first| {
                //     // each thread writes its slot exactly once
                //     firsts_atomic[tid].store(first, Ordering::Relaxed);
                // }
                |_, _| {}
            },
            threads,
        )?;

        // // materialize the result Vec<(tid, first)>
        // let mut firsts = HashSet::new();
        // for (_tid, a) in firsts_atomic.iter().enumerate() {
        //     let v = a.load(Ordering::Relaxed);
        //     if v != usize::MAX {
        //         firsts.insert(v);
        //     }
        // }
        //
        // println!("firsts {:?}", firsts);

        // order edges by dest node id
        for node in 0..node_count {
            let begin = offsets.get(node).load(Ordering::Relaxed);
            let end = offsets.get(node + 1).load(Ordering::Relaxed);
            let node_edges =
                neighbors
                    .mut_slice(begin, end)
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        format!("error getting node {node}'s edges as a mut slice for ordering")
                            .into()
                    })?;
            // println!("edges before {:?}", node_edges);
            node_edges.sort();
            // println!("edges after {:?}", node_edges);
        }

        neighbors.flush()?;
        offsets.flush()?;

        std::fs::remove_file(&h_fn)?;

        Self::open(&n_fn, batch)
    }

    fn parse_mtx_header<P: AsRef<Path>>(path: P) -> Result<MTXHeader, Box<dyn std::error::Error>> {
        let f = File::open(path.as_ref())?;
        let mut rdr = BufReader::new(f);

        // read and parse the header line
        let mut header = String::new();
        rdr.read_line(&mut header)?;
        if !header.starts_with("%%MatrixMarket") {
            return Err("Invalid MatrixMarket header".into());
        }
        // tokens: 0=%%MatrixMarket, 1=matrix, 2=coordinate, 3=field, 4=symmetry
        let toks: Vec<_> = header.split_whitespace().collect();
        if toks.len() < 5 {
            return Err("Malformed MatrixMarket header".into());
        }
        if !toks[1].eq_ignore_ascii_case("matrix")
            || !toks[2].eq_ignore_ascii_case("coordinate")
            || !(toks[3].eq_ignore_ascii_case("pattern") || toks[3].eq_ignore_ascii_case("integer"))
            || !(toks[4].eq_ignore_ascii_case("symmetric")
                || toks[4].eq_ignore_ascii_case("skew-symmetric")
                || toks[4].eq_ignore_ascii_case("general"))
        {
            return Err(
                "Only 'matrix coordinate pattern/integer symmetric' format is supported".into(),
            );
        }

        // skip comment lines (%) to find the size line: "nrows ncols nnz"
        let (nrows, ncols, nnz_declared) = {
            let mut line = String::new();
            loop {
                line.clear();
                let n = rdr.read_line(&mut line)?;
                if n == 0 {
                    return Err("Unexpected EOF before reading size line".into());
                }
                // ignore comment lines
                let line_trim = line.trim();
                if line_trim.is_empty() || line_trim.starts_with('%') {
                    continue;
                }
                // parse sizes
                let mut it = line_trim.split_whitespace();
                let nr: usize = it.next().ok_or("Missing nrows")?.parse()?;
                let nc: usize = it.next().ok_or("Missing ncols")?.parse()?;
                let nnz: usize = it.next().ok_or("Missing nnz")?.parse()?;
                if nr != nc {
                    return Err(format!(
                        "Only symmetric matrices are supported but {{nr = {nr}}} != {{nc = {nc}}}"
                    )
                    .into());
                }
                break (nr, nc, nnz);
            }
        };
        Ok((nrows, ncols, nnz_declared))
    }

    fn parallel_parse_mtx_with<F, FF, P: AsRef<Path>>(
        path: P,
        emit: F,
        emit_first: FF,
        threads: usize,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, u64, u64) + Send + Sync + Clone,
        FF: FnMut(usize, usize) + Send + Sync + Clone,
    {
        let f = File::open(path.as_ref())?;
        let file_len = f.metadata()?.len();
        let mut rdr = BufReader::new(f);

        // read and parse the header line
        let mut header = String::new();
        rdr.read_line(&mut header)?;
        if !header.starts_with("%%MatrixMarket") {
            return Err("Invalid MatrixMarket header".into());
        }
        // tokens: 0=%%MatrixMarket, 1=matrix, 2=coordinate, 3=field, 4=symmetry
        let toks: Vec<_> = header.split_whitespace().collect();
        if toks.len() < 5 {
            return Err("Malformed MatrixMarket header".into());
        }
        if !toks[1].eq_ignore_ascii_case("matrix")
            || !toks[2].eq_ignore_ascii_case("coordinate")
            || !(toks[3].eq_ignore_ascii_case("pattern") || toks[3].eq_ignore_ascii_case("integer"))
            || !(toks[4].eq_ignore_ascii_case("symmetric")
                || toks[4].eq_ignore_ascii_case("skew-symmetric")
                || toks[4].eq_ignore_ascii_case("general"))
        {
            return Err(
                "Only 'matrix coordinate pattern/integer symmetric' format is supported".into(),
            );
        }
        let pattern = toks[3].eq_ignore_ascii_case("pattern");
        let symmetric = !toks[4].eq_ignore_ascii_case("general");

        // skip comment lines (%) to find the size line: "nrows ncols nnz"
        let (nrows, ncols, nnz_declared) = {
            let mut line = String::new();
            loop {
                line.clear();
                let n = rdr.read_line(&mut line)?;
                if n == 0 {
                    return Err("Unexpected EOF before reading size line".into());
                }
                // ignore comment lines
                let line_trim = line.trim();
                if line_trim.is_empty() || line_trim.starts_with('%') {
                    continue;
                }
                // parse sizes
                let mut it = line_trim.split_whitespace();
                let nr: usize = it.next().ok_or("Missing nrows")?.parse()?;
                let nc: usize = it.next().ok_or("Missing ncols")?.parse()?;
                let nnz: usize = it.next().ok_or("Missing nnz")?.parse()?;
                if nr != nc {
                    return Err(format!(
                        "Only symmetric matrices are supported but {{nr = {nr}}} != {{nc = {nc}}}"
                    )
                    .into());
                }
                break (nr, nc, nnz);
            }
        };

        let header_offset = rdr.stream_position()?;
        let thread_load = file_len
            .saturating_sub(header_offset)
            .div_ceil(threads as u64);

        let seen = Arc::new(AtomicUsize::new(0));
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::new();

            for tid in 0..threads {
                let begin_pos = std::cmp::min(header_offset + tid as u64 * thread_load, file_len);
                let end_pos =
                    std::cmp::min(header_offset + (tid + 1) as u64 * thread_load, file_len)
                        as usize;
                let p = path.as_ref();

                let seen = seen.clone();
                let mut emit_first = emit_first.clone();
                let mut emit = emit.clone();

                let handle = s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        // open the file separately for each thread
                        let file = File::open(p)?;
                        let mut cur_pos = begin_pos as usize;

                        // line up bufreader to next line start
                        let mut rdr = BufReader::new(file);
                        if tid == 0 {
                            rdr.seek(SeekFrom::Start(begin_pos))?;
                        } else {
                            // this way we avoid the case where thread's bufreader startsoff
                            // aligned and the first edge is skipped
                            rdr.seek(SeekFrom::Start(begin_pos.saturating_sub(1)))?;
                            cur_pos += rdr.skip_until(b'\n')?.saturating_sub(1);
                        }

                        let mut line = String::new();
                        if seen.load(Ordering::Relaxed) < nnz_declared && end_pos > cur_pos {
                            line.clear();

                            let n = rdr.read_line(&mut line)?;
                            if n == 0 {
                                // Some files under-report nnz in the header; we stop if EOF reached
                                return Ok(());
                            } else {
                                cur_pos += n;
                            }

                            let mut t = line.trim();
                            while t.is_empty() || t.starts_with('%') {
                                let n = rdr.read_line(&mut line)?;
                                if n == 0 {
                                    // Some files under-report nnz in the header; we stop if EOF reached
                                    return Ok(());
                                } else {
                                    cur_pos += n;
                                }
                                t = line.trim()
                            }

                            let mut it = t.split_whitespace();

                            // 1st two tokens must be i j (1-based)
                            let i1: usize = match it.next() {
                                Some(s) => s.parse()?,
                                None => {
                                    return Err(format!(
                                        "error in thread {tid}: error getting first node's id"
                                    )
                                    .into());
                                } // malformed line
                            };

                            let j1: u64 = match it.next() {
                                Some(s) => s.parse()?,
                                None => {
                                    return Err(format!(
                                        "error in thread {tid}: error getting first edge's dest id"
                                    )
                                    .into());
                                } // malformed line
                            };

                            // Convert to 0-based, also ensure in-bounds
                            if i1 == 0 || j1 == 0 {
                                return Err(format!(
                                    "MatrixMarket indices are 1-based; found 0 ({line})"
                                )
                                .into());
                            }

                            let i = i1 - 1;
                            let j = j1 - 1;

                            let w: u64 = if pattern {
                                0
                            } else {
                                match it.next() {
                                    Some(s) => s.parse()?,
                                    None => 0, // malformed line; skip
                                }
                            };

                            if i >= nrows || j as usize >= ncols {
                                return Err(format!(
                                    "Entry out of bounds: ({i},{j}) not in [{nrows}x{ncols}]"
                                )
                                .into());
                            }

                            // Emit edges:
                            // first directed edge i --(w)--> j
                            emit_first(tid, i);
                            emit(i, j, w);

                            // second directed edge (if symmetrical) j --(w)--> i
                            if symmetric && i != j as usize {
                                emit(j as usize, i as u64, w);
                            }

                            seen.add(1, Ordering::Relaxed);
                        }

                        while seen.load(Ordering::Relaxed) < nnz_declared && end_pos > cur_pos {
                            line.clear();

                            let n = rdr.read_line(&mut line)?;
                            if n == 0 {
                                // Some files under-report nnz in the header; we stop if EOF reached
                                break;
                            } else {
                                cur_pos += n;
                            }

                            let t = line.trim();
                            if t.is_empty() || t.starts_with('%') {
                                continue;
                            }

                            let mut it = t.split_whitespace();

                            // 1st two tokens must be i j (1-based)
                            let i1: usize = match it.next() {
                                Some(s) => s.parse()?,
                                None => continue, // malformed line; skip
                            };

                            let j1: u64 = match it.next() {
                                Some(s) => s.parse()?,
                                None => continue, // malformed line; skip
                            };

                            // Convert to 0-based, also ensure in-bounds
                            if i1 == 0 || j1 == 0 {
                                return Err(format!(
                                    "MatrixMarket indices are 1-based; found 0 ({line})"
                                )
                                .into());
                            }

                            let i = i1 - 1;
                            let j = j1 - 1;

                            let w: u64 = if pattern {
                                0
                            } else {
                                match it.next() {
                                    Some(s) => s.parse()?,
                                    None => 0, // malformed line; skip
                                }
                            };

                            if i >= nrows || j as usize >= ncols {
                                return Err(format!(
                                    "Entry out of bounds: ({i},{j}) not in [{nrows}x{ncols}]"
                                )
                                .into());
                            }

                            // Emit edges:
                            // first directed edge i --(w)--> j
                            emit(i, j, w);

                            // second directed edge (if symmetrical) j --(w)--> i
                            if symmetric && i != j as usize {
                                emit(j as usize, i as u64, w);
                            }

                            seen.add(1, Ordering::Relaxed);
                        }
                        Ok(())
                    },
                );

                handles.push(handle);
            }

            for (idx, handle) in handles.into_iter().enumerate() {
                handle
                    .join()
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error joining thread {idx}: {:?}", e).into()
                    })?
                    .map_err(|e| -> Box<dyn std::error::Error> {
                        format!("error in thread {idx}: {:?}", e).into()
                    })?;
            }
            Ok(())
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })??;

        Ok(())
    }

    fn parse_mtx_with<F, P: AsRef<Path>>(
        path: P,
        mut emit: F,
    ) -> Result<(), Box<dyn std::error::Error>>
    where
        F: FnMut(usize, u64, u64),
    {
        let f = File::open(path.as_ref())?;
        let mut rdr = BufReader::new(f);

        // read and parse the header line
        let mut header = String::new();
        rdr.read_line(&mut header)?;
        if !header.starts_with("%%MatrixMarket") {
            return Err("Invalid MatrixMarket header".into());
        }
        // tokens: 0=%%MatrixMarket, 1=matrix, 2=coordinate, 3=field, 4=symmetry
        let toks: Vec<_> = header.split_whitespace().collect();
        if toks.len() < 5 {
            return Err("Malformed MatrixMarket header".into());
        }
        if !toks[1].eq_ignore_ascii_case("matrix")
            || !toks[2].eq_ignore_ascii_case("coordinate")
            || !(toks[3].eq_ignore_ascii_case("pattern") || toks[3].eq_ignore_ascii_case("integer"))
            || !(toks[4].eq_ignore_ascii_case("symmetric")
                || toks[4].eq_ignore_ascii_case("skew-symmetric")
                || toks[4].eq_ignore_ascii_case("general"))
        {
            return Err(
                "Only 'matrix coordinate pattern/integer symmetric' format is supported".into(),
            );
        }
        let pattern = toks[3].eq_ignore_ascii_case("pattern");
        let symmetric = !toks[4].eq_ignore_ascii_case("general");

        // skip comment lines (%) to find the size line: "nrows ncols nnz"
        let (nrows, ncols, nnz_declared) = {
            let mut line = String::new();
            loop {
                line.clear();
                let n = rdr.read_line(&mut line)?;
                if n == 0 {
                    return Err("Unexpected EOF before reading size line".into());
                }
                // ignore comment lines
                let line_trim = line.trim();
                if line_trim.is_empty() || line_trim.starts_with('%') {
                    continue;
                }
                // parse sizes
                let mut it = line_trim.split_whitespace();
                let nr: usize = it.next().ok_or("Missing nrows")?.parse()?;
                let nc: usize = it.next().ok_or("Missing ncols")?.parse()?;
                let nnz: usize = it.next().ok_or("Missing nnz")?.parse()?;
                if nr != nc {
                    return Err(format!(
                        "Only symmetric matrices are supported but {{nr = {nr}}} != {{nc = {nc}}}"
                    )
                    .into());
                }
                break (nr, nc, nnz);
            }
        };

        // stream the nnz entries
        // For coordinate:
        // - pattern:          i j
        // - integer/real:     i j val
        // - complex/hermitian i j real imag  (we’ll ignore values and just read i j)
        //
        // MatrixMarket is 1-based; convert to 0-based.
        // We’ll be forgiving: we keep reading non-comment non-empty lines until we see nnz_declared entries.
        let mut seen = 0usize;
        let mut line = String::new();
        while seen < nnz_declared {
            line.clear();
            let n = rdr.read_line(&mut line)?;
            if n == 0 {
                // Some files under-report nnz in the header; we stop if EOF reached
                break;
            }
            let t = line.trim();
            if t.is_empty() || t.starts_with('%') {
                continue;
            }
            let mut it = t.split_whitespace();

            // 1st two tokens must be i j (1-based)
            let i1: usize = match it.next() {
                Some(s) => s.parse()?,
                None => continue, // malformed line; skip
            };
            let j1: u64 = match it.next() {
                Some(s) => s.parse()?,
                None => continue, // malformed line; skip
            };
            // Convert to 0-based, also ensure in-bounds
            if i1 == 0 || j1 == 0 {
                return Err("MatrixMarket indices are 1-based; found 0".into());
            }
            let i = i1 - 1;
            let j = j1 - 1;

            let w: u64 = if pattern {
                0
            } else {
                match it.next() {
                    Some(s) => s.parse()?,
                    None => 0, // malformed line; skip
                }
            };
            if i >= nrows || j as usize >= ncols {
                return Err(
                    format!("Entry out of bounds: ({i},{j}) not in [{nrows}x{ncols}]").into(),
                );
            }

            // Emit edges:
            // one directed edge i --(w)--> j
            emit(i, j, w);
            if symmetric && i != j as usize {
                emit(j as usize, i as u64, w);
            }

            seen += 1;
        }

        Ok(())
    }
}

type MTXHeader = (usize, usize, usize);
