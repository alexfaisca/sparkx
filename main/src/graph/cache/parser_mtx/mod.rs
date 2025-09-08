mod producers;

use crate::{
    graph::cache::utils::{FileType, H, cache_file_name},
    shared_slice::SharedSliceMut,
};

use num_cpus::get_physical;
use portable_atomic::{AtomicUsize, Ordering};
use std::path::Path;

use super::{GraphCache, utils::apply_permutation_in_place};

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
        let h_fn = cache_file_name(&n_fn, &FileType::Helper(H::H), Some(0))?;
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
        {
            Self::parallel_no_labels_parse_mtx_with(
                path.as_ref(),
                {
                    let offsets = offsets.shared_slice().clone();
                    move |u, _| {
                        offsets.get(u).add(1, Ordering::Relaxed);
                    }
                },
                threads,
            )?;
        }
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

        let mut neighbors = SharedSliceMut::<usize>::abst_mem_mut(&n_fn, sum, true)?;
        let el_fn = Self::init_cache_file_from_id_or_random(&id, FileType::EdgeLabel(H::H), None)?;
        let mut edgelabels = SharedSliceMut::<E>::abst_mem_mut(&el_fn, sum, true)?;

        if E::is_labeled() {
            Self::parallel_edge_labels_parse_mtx_with(
                path.as_ref(),
                {
                    let mut neighbors = neighbors.shared_slice();
                    let mut edgelabels = edgelabels.shared_slice();
                    let offsets = offsets.shared_slice();
                    let counters = counters.shared_slice();
                    move |u, v, w| {
                        *neighbors.get_mut(
                            offsets.get(u).load(Ordering::Relaxed)
                                + counters.get(u).load(Ordering::Relaxed),
                        ) = v;
                        *edgelabels.get_mut(
                            offsets.get(u).load(Ordering::Relaxed)
                                + counters.get(u).fetch_add(1, Ordering::Relaxed),
                        ) = E::new(w);
                    }
                },
                threads,
            )?;
            let mut idx = Vec::with_capacity(u16::MAX as usize);
            for node in 0..node_count {
                let begin = offsets.get(node).load(Ordering::Relaxed);
                let end = offsets.get(node + 1).load(Ordering::Relaxed);
                (0..end - begin).for_each(|v| {
                    idx.push(v);
                });
                let node_edges = neighbors.mut_slice(begin, end).ok_or_else(
                    || -> Box<dyn std::error::Error> {
                        format!("error getting node {node}'s edges as a mut slice for ordering")
                            .into()
                    },
                )?;
                let edge_labels = edgelabels.mut_slice(begin, end).ok_or_else(
                    || -> Box<dyn std::error::Error> {
                        format!("error getting node {node}'s edges' edge labels as a mut slice for ordering")
                            .into()
                    },
                )?;
                idx.sort_by_key(|&i| node_edges[i]);
                apply_permutation_in_place(idx.as_mut_slice(), node_edges, edge_labels);
            }
            edgelabels.flush()?;
        } else {
            Self::parallel_no_labels_parse_mtx_with(
                path.as_ref(),
                {
                    let mut neighbors = neighbors.shared_slice();
                    let offsets = offsets.shared_slice();
                    let counters = counters.shared_slice();
                    move |u, v| {
                        *neighbors.get_mut(
                            offsets.get(u).load(Ordering::Relaxed)
                                + counters.get(u).fetch_add(1, Ordering::Relaxed),
                        ) = v;
                    }
                },
                threads,
            )?;
            for node in 0..node_count {
                let begin = offsets.get(node).load(Ordering::Relaxed);
                let end = offsets.get(node + 1).load(Ordering::Relaxed);
                let node_edges = neighbors.mut_slice(begin, end).ok_or_else(
                    || -> Box<dyn std::error::Error> {
                        format!("error getting node {node}'s edges as a mut slice for ordering")
                            .into()
                    },
                )?;
                node_edges.sort();
            }
        }

        neighbors.flush()?;
        offsets.flush()?;

        std::fs::remove_file(&h_fn)?;

        Self::open(&n_fn, batch)
    }
}

type MTXHeader = (usize, usize, usize);
