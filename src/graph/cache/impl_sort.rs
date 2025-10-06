use crate::shared_slice::SharedSliceMut;

use super::{utils::apply_permutation_in_place, GraphCache};

use crossbeam::thread;

#[allow(dead_code)]
impl<N: super::N, E: super::E, Ix: super::IndexType> GraphCache<N, E, Ix> {
    #[inline(always)]
    pub(super) fn sort_edges(offsets: SharedSliceMut<usize>, neighbors: SharedSliceMut<usize>, threads: usize) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = offsets.len().saturating_sub(1);
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::new();
            for tid in 0..threads {
                let node_load = node_count.div_ceil(threads);
                let begin = std::cmp::min(node_load * tid, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                let mut neighbors = neighbors;
    
                let handle = s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        for node in begin..end {
                            let begin = *offsets.get(node);
                            let end = *offsets.get(node + 1);
                            let node_edges = neighbors.mut_slice(begin, end).ok_or_else(
                                || -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error getting node {node}'s edges as a mut slice for ordering")
                                        .into()
                                },
                            )?;
                            node_edges.sort();
                        }
                        Ok(())
                });
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
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
    }

    #[inline(always)]
    pub(super) fn sort_edges_with_labels(offsets: SharedSliceMut<usize>, neighbors: SharedSliceMut<usize>, edge_labels: SharedSliceMut<E>, threads: usize) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = offsets.len().saturating_sub(1);
        thread::scope(|s| -> Result<(), Box<dyn std::error::Error>> {
            let mut handles = Vec::new();
            for tid in 0..threads {
                let node_load = node_count.div_ceil(threads);
                let begin = std::cmp::min(node_load * tid, node_count);
                let end = std::cmp::min(begin + node_load, node_count);
                let mut neighbors = neighbors;
                let mut edge_labels = edge_labels;
    
                let handle = s.spawn(
                    move |_| -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
                        let mut idx = Vec::with_capacity(u16::MAX as usize);
                        for node in begin..end {
                            let begin = *offsets.get(node);
                            let end = *offsets.get(node + 1);
                            (0..end - begin).for_each(|v| {
                                idx.push(v);
                            });
                            let node_edges = neighbors.mut_slice(begin, end).ok_or_else(
                                || -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error getting node {node}'s edges as a mut slice for ordering")
                                        .into()
                                },
                            )?;
                            let edge_labels = edge_labels.mut_slice(begin, end).ok_or_else(
                                || -> Box<dyn std::error::Error + Send + Sync> {
                                    format!("error getting node {node}'s edges' edge labels as a mut slice for ordering")
                                        .into()
                                },
                            )?;
                            idx.sort_by_key(|&i| node_edges[i]);
                            apply_permutation_in_place(idx.as_mut_slice(), node_edges, edge_labels);
                        }
                        Ok(())
                });
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
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?
    }
}
