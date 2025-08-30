use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;

use crossbeam::thread;
use memmap2::{Mmap, MmapMut};
use num_cpus::get_physical;
use std::mem::ManuallyDrop;
use std::{collections::HashMap, fs::OpenOptions, time::Instant};

/// For the computation of the euler trail(s) of a [`GraphMemoryMap`] instance using *Hierholzer's Algorithm*.
///
/// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
#[allow(dead_code)]
pub struct AlgoHierholzer<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    /// Graph for which the euler trail(s) is(are) computed.
    g: &'a GraphMemoryMap<EdgeType, Edge>,
    /// Memmapped slice containing the euler trails.
    pub euler_trails: AbstractedProceduralMemoryMut<usize>,
    /// Array containing the starting position of each euler trail.
    pub trail_index: Vec<usize>,
}

#[allow(dead_code)]
impl<'a, EdgeType, Edge> AlgoHierholzer<'a, EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    /// Performs graph traversal using *Hierholzer's Algorithm*.
    ///
    /// The resulting Euler trail(s) is(are) stored in memory (in a memmapped file) nodewise[^1].
    ///
    /// * Note *1*: by definition, isolated nodes won't be present in the euler trail unless they have *self-loops*[^2].
    /// * Note *2*: in the case of a non-eulerian graph, trails are stored sequentially in memory, with their respective starting indexes stored in the aboce mentioned `trail_index` array.
    /// * Note *3*: the last edge of each euler trail, connecting the trail into an euler cycle is intentionally left out of the resulting euler trail(s).
    ///
    /// [^1]: for each two consecutive node entries, u and v, the(an) edge (u,v) between them is traversed.
    /// [^2]: edges of the type (u, u), for a given node u.
    ///
    /// # Arguments
    ///
    /// * `g` --- the [`GraphMemoryMap`] instance for which k-core decomposition is to be performed in.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn new(g: &'a GraphMemoryMap<EdgeType, Edge>) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::EulerTrail, None)?;
        let mut euler = AlgoHierholzer {
            g,
            euler_trails: SharedSliceMut::<usize>::abst_mem_mut(&out_fn, g.width(), true)?,
            trail_index: Vec::new(),
        };
        euler.compute(10)?;
        Ok(euler)
    }

    pub fn drop_cache(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        let this = ManuallyDrop::new(self);
        let out_fn = this.g.build_cache_filename(CacheFile::EulerTrail, None)?;
        std::fs::remove_file(out_fn)?;
        Ok(())
    }

    fn initialize_hierholzers_procedural_memory(
        &self,
        mmap: u8,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<usize>,
            AbstractedProceduralMemoryMut<u8>,
        ),
        Box<dyn std::error::Error>,
    > {
        let node_count = self.g.size();

        let index_ptr = SharedSlice::<usize>::new(self.g.index_ptr(), self.g.offsets_size());

        let e_fn = self
            .g
            .build_cache_filename(CacheFile::EulerTrail, Some(0))?;
        let c_fn = self
            .g
            .build_cache_filename(CacheFile::EulerTrail, Some(1))?;

        let edges = SharedSliceMut::<usize>::abst_mem_mut(&e_fn, node_count, mmap > 0)?;
        let count = SharedSliceMut::<u8>::abst_mem_mut(&c_fn, node_count, mmap > 1)?;

        thread::scope(|scope| {
            // initializations always uses at least two threads per core
            let threads = self.g.thread_num().max(get_physical() * 2);
            let node_load = node_count.div_ceil(threads);

            for i in 0..threads {
                let mut edges = edges.shared_slice();
                let mut count = count.shared_slice();
                let begin = std::cmp::min(i * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                scope.spawn(move |_| {
                    for k in begin..end {
                        *edges.get_mut(k) = *index_ptr.get(k);
                        *count.get_mut(k) = (*index_ptr.get(k + 1) - *index_ptr.get(k)) as u8;
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        edges.flush()?;
        count.flush()?;

        Ok((edges, count))
    }

    /// Computes the euler trail(s) of a graph using *Hierholzer's Algorithm*.
    ///
    /// # Arguments
    ///
    /// * `mmap`: `u8` --- the level of memmapping to be used during the computation (*experimental feature*).
    ///
    pub fn compute(&mut self, mmap: u8) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = match self.g.size() {
            0 => return Ok(()),
            i => i,
        };
        let graph_ptr = SharedSlice::<Edge>::new(self.g.edges_ptr(), self.g.width());

        let (mut edges, mut edge_count) = self.initialize_hierholzers_procedural_memory(mmap)?;

        let mut start_vertex_counter = 0usize;
        let mut cycles = self.euler_trails.shared_slice();
        let mut write_idx = 0usize;
        // find cycles until no unused edges remain
        loop {
            let start_v = loop {
                let idx = if start_vertex_counter >= node_count {
                    break None;
                } else {
                    start_vertex_counter
                };
                if idx >= node_count {
                    break None;
                }
                if *edge_count.get(idx) > 0 {
                    break Some(idx);
                }
                start_vertex_counter += 1;
            };
            let start_v = match start_v {
                Some(v) => v,
                None => {
                    break;
                }
            };

            // Hierholzer's DFS
            let mut stack: Vec<usize> = Vec::new();
            let mut cycle: Vec<usize> = Vec::new();
            stack.push(start_v);
            while let Some(&v) = stack.last() {
                if *edge_count.get(v) > 0 {
                    *edge_count.get_mut(v) -= 1;
                    stack.push(graph_ptr.get(*edges.get(v)).dest());
                    *edges.get_mut(v) += 1;
                } else {
                    stack.pop();
                    cycle.push(v);
                }
            }
            if !cycle.is_empty() {
                cycle.reverse();

                let cycle_slice = cycle.as_slice();
                let end = cycle_slice.len() - 1;

                write_idx = cycles
                    .write_slice(write_idx, &cycle_slice[..end])
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "error couldn't slice mmap to write cycle".into()
                    })?;

                self.trail_index.push(write_idx);
                cycle.clear();
            }
        }

        self.euler_trails.flush()?;

        // cleanup cache
        self.g.cleanup_cache(CacheFile::EulerTrail)?;

        Ok(())
    }

    /// Returns the number of euler trails found upon performing *Hierholzer's* graph traversal algorithm.
    pub fn trail_number(&self) -> usize {
        self.trail_index.len()
    }

    /// Returns the number of (strongly[^1]) connected components on the [`GraphMemoryMap`] instance.
    ///
    /// [^1]: Given the topologic properties of de Bruijn graphs, the number of strongly connected
    /// components always equals the number of weakly connected components.
    ///
    /// [`GraphMemoryMap`]: ../../generic_memory_map/struct.GraphMemoryMap.html#
    pub fn count_connected_components(&self) -> usize {
        self.trail_number()
    }

    /// Returns the starting index of a given euler trail.
    ///
    /// # Arguments
    ///
    /// * `idx`: `usize` --- the euler trail for which the in-memory starting index is to be returned.
    ///
    /// * Returns
    ///
    /// This method will return `Some` if `idx` is smaller than the number of euler trails found upon performing *Hierholzer's* graph traversal algorithm and `None` otherwise, i.e. iff `idx < trail_index.len()`.
    ///
    pub fn get_trail(&self, idx: usize) -> Option<SharedSliceMut<usize>> {
        if idx < self.trail_index.len() {
            Some(self.euler_trails.shared_slice())
        } else {
            None
        }
    }

    #[deprecated]
    fn create_memmapped_mut_slice_from_tmp_file<V>(
        filename: String,
        len: usize,
    ) -> Result<(SharedSliceMut<V>, MmapMut), Box<dyn std::error::Error>> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&filename)?;
        file.set_len((len * std::mem::size_of::<V>()) as u64)?;
        SharedSliceMut::<V>::from_file(&file)
    }

    #[deprecated]
    fn create_memmapped_slice_from_tmp_file<V>(
        filename: String,
    ) -> Result<(SharedSlice<V>, Mmap), Box<dyn std::error::Error>> {
        SharedSlice::<V>::from_file(&OpenOptions::new().read(true).open(filename)?)
    }

    #[deprecated]
    #[allow(deprecated)]
    fn merge_euler_trails(
        &mut self,
        cycle_offsets: Vec<(usize, usize, usize)>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut trail_heads: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
        let mmap_fn = self.g.build_cache_filename(CacheFile::EulerTrail, None)?;
        let (cycles, _mmap) = Self::create_memmapped_slice_from_tmp_file::<usize>(mmap_fn)?;

        cycle_offsets.iter().for_each(|(idx, begin, _)| {
            trail_heads
                .entry(*cycles.get(*begin))
                .or_default()
                .push((*idx, *idx, 0));
        });

        let euler_time = Instant::now();
        // generate writing sets
        let chunk = 4096; // read 8 pages of 4KB at a time
        for (t_idx, t_begin, t_end) in cycle_offsets.iter() {
            let trail_ptr = match cycles.slice(*t_begin, *t_end) {
                Some(i) => i.as_ptr(),
                None => panic!("error getting memmapped slice of trail {}", t_idx),
            };
            let trail_slice = SharedSlice::<usize>::new(trail_ptr, *t_end - *t_begin);
            let mut pos = 0; // pos in u64 terms
            while let Some(next_slice) = trail_slice.slice(pos, pos + chunk) {
                for (pos_idx, node) in next_slice.iter().enumerate() {
                    if let Some(head_v) = trail_heads.get_mut(node) {
                        let p_idx = pos_idx + pos + 1;
                        for (vec_idx, (in_cyc, _, _)) in head_v.clone().iter().enumerate() {
                            if *in_cyc == *t_idx {
                                continue;
                            }
                            head_v[vec_idx] = (*in_cyc, *t_idx, p_idx);
                        }
                    }
                }
                pos += next_slice.len();
            }
        }
        println!("euler writing set {:?}", euler_time.elapsed());

        // break cycles
        let mut v: Vec<_> = trail_heads
            .values_mut()
            .flat_map(|vec| vec.drain(..))
            .collect();

        let euler_time = Instant::now();
        let mut euler_trail_sets = FindDisjointSetsEulerTrails::new(v.as_mut());
        euler_trail_sets.cycle_check();

        // Union find and write
        let mut trail_sets: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();

        for ((trail, parent_trail, pos), grand_parent) in euler_trail_sets.trails {
            trail_sets
                .entry(grand_parent)
                .or_default()
                .push((trail, parent_trail, pos));
        }
        println!("euler union {:?}", euler_time.elapsed());

        let mut output_len: HashMap<usize, (usize, usize)> = HashMap::new();
        // in nodes usize
        let cycles_length: Vec<usize> = cycle_offsets.iter().map(|(_, b, e)| e - b).collect();
        for head_trail in trail_sets.keys() {
            output_len.insert(*head_trail, (*head_trail, 1));
            for (trail, _, _) in trail_sets.get(head_trail).unwrap() {
                output_len.get_mut(head_trail).unwrap().1 += cycles_length[*trail] - 1;
            }
        }

        let mut keys_by_trail_size: Vec<(usize, usize)> =
            output_len.values().map(|&(a, b)| (a, b)).collect();
        keys_by_trail_size.sort_unstable_by_key(|(_, s)| std::cmp::Reverse(*s));

        let mut total = 0;
        self.trail_index.reserve_exact(keys_by_trail_size.len());

        for (head_trail, output_len) in keys_by_trail_size.iter() {
            // may be a loop onto self
            if *output_len <= 1 {
                continue;
            }
            self.trail_index.push(total);
            // Initialize writing guide
            let trail_guide = trail_sets.get(head_trail).unwrap();
            let mut pos_map: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
            for (trail, parent_trail, pos) in trail_guide {
                pos_map
                    .entry(*parent_trail)
                    .or_default()
                    .push((*pos, *trail));
            }

            // sort positions in ascending order for each trail and push head trail to stack
            pos_map
                .values_mut()
                .for_each(|v| v.sort_unstable_by_key(|(pos, _)| std::cmp::Reverse(*pos)));

            let mut stack: Vec<(usize, usize, usize)> = vec![];
            let mut remaining: Vec<(usize, usize)> = vec![];
            let mut expand = Some((*head_trail, 0));
            while let Some((current_trail, pos)) = expand {
                if let Some(nested_trails) = pos_map.get_mut(&current_trail) {
                    if let Some((insert_pos, trail)) = nested_trails.pop() {
                        if trail == current_trail || cycles_length[trail] == 0 || insert_pos < pos {
                            continue;
                        }
                        stack.push((current_trail, pos, insert_pos));
                        remaining.push((current_trail, insert_pos));
                        remaining.push((trail, 1)); // elipse repeated node
                    } else {
                        stack.push((current_trail, pos, cycles_length[current_trail]));
                    }
                } else {
                    stack.push((current_trail, pos, cycles_length[current_trail]));
                }
                expand = remaining.pop()
            }
            let out_len = *output_len - 1;

            let mut idx = 0;

            for (cycle, from, to) in stack {
                if to <= from {
                    continue;
                }
                let (_, t_begin, _) = cycle_offsets[cycle];
                let begin = t_begin + from;
                // if at the last element of trail, elipse it to break loop
                let end = if idx + to - from > out_len {
                    t_begin + to - 1
                } else {
                    t_begin + to
                };

                self.euler_trails.shared_slice().write_shared_slice(
                    cycles,
                    total + idx,
                    begin,
                    end - begin,
                );
                idx += end - begin;
                total += end - begin;
            }
            self.euler_trails.flush_async()?;
        }
        // println!(
        //     "euler write {:?} (written {:?} num trails {:?}) {:?} (written - num trails {:?})",
        //     euler_time.elapsed(),
        //     total,
        //     keys_by_trail_size.len(),
        //     self.graph.width(),
        //     total - keys_by_trail_size.len()
        // );
        // println!("euler trails merge {:?}", time.elapsed());

        Ok(())
    }
}

#[cfg(feature = "bench")]
#[allow(dead_code)]
impl<'a, EdgeType, Edge> AlgoHierholzer<'a, EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    pub fn new_no_compute(
        g: &'a GraphMemoryMap<EdgeType, Edge>,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let out_fn = g.build_cache_filename(CacheFile::EulerTrail, None)?;
        Ok(Self {
            g,
            euler_trails: SharedSliceMut::<usize>::abst_mem_mut(&out_fn, g.width(), true)?,
            trail_index: Vec::new(),
        })
    }

    pub fn init_cache_mem(
        &self,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<usize>,
            AbstractedProceduralMemoryMut<u8>,
        ),
        Box<dyn std::error::Error>,
    > {
        let node_count = self.g.size();

        let index_ptr = SharedSlice::<usize>::new(self.g.index_ptr(), self.g.offsets_size());

        let e_fn = self
            .g
            .build_cache_filename(CacheFile::EulerTrail, Some(0))?;
        let c_fn = self
            .g
            .build_cache_filename(CacheFile::EulerTrail, Some(1))?;

        let edges = SharedSliceMut::<usize>::abst_mem_mut(&e_fn, node_count, true)?;
        let count = SharedSliceMut::<u8>::abst_mem_mut(&c_fn, node_count, true)?;

        thread::scope(|scope| {
            // initializations always uses at least two threads per core
            let threads = self.g.thread_num().max(get_physical() * 2);
            let node_load = node_count.div_ceil(threads);

            for i in 0..threads {
                let mut edges = edges.shared_slice();
                let mut count = count.shared_slice();
                let begin = std::cmp::min(i * node_load, node_count);
                let end = std::cmp::min(begin + node_load, node_count);

                scope.spawn(move |_| {
                    for k in begin..end {
                        *edges.get_mut(k) = *index_ptr.get(k);
                        *count.get_mut(k) = (*index_ptr.get(k + 1) - *index_ptr.get(k)) as u8;
                    }
                });
            }
        })
        .map_err(|e| -> Box<dyn std::error::Error> { format!("{:?}", e).into() })?;

        edges.flush()?;
        count.flush()?;

        Ok((edges, count))
    }

    pub fn compute_with_proc_mem(
        &mut self,
        proc_mem: (
            AbstractedProceduralMemoryMut<usize>,
            AbstractedProceduralMemoryMut<u8>,
        ),
    ) -> Result<(), Box<dyn std::error::Error>> {
        let node_count = match self.g.size() {
            0 => return Ok(()),
            i => i,
        };
        let graph_ptr = SharedSlice::<Edge>::new(self.g.edges_ptr(), self.g.width());

        let (mut edges, mut edge_count) = proc_mem;

        let mut start_vertex_counter = 0usize;
        let mut cycles = self.euler_trails.shared_slice();
        let mut write_idx = 0usize;
        // find cycles until no unused edges remain
        loop {
            let start_v = loop {
                let idx = if start_vertex_counter >= node_count {
                    break None;
                } else {
                    start_vertex_counter
                };
                if idx >= node_count {
                    break None;
                }
                if *edge_count.get(idx) > 0 {
                    break Some(idx);
                }
                start_vertex_counter += 1;
            };
            let start_v = match start_v {
                Some(v) => v,
                None => {
                    break;
                }
            };

            // Hierholzer's DFS
            let mut stack: Vec<usize> = Vec::new();
            let mut cycle: Vec<usize> = Vec::new();
            stack.push(start_v);
            while let Some(&v) = stack.last() {
                if *edge_count.get(v) > 0 {
                    *edge_count.get_mut(v) -= 1;
                    stack.push(graph_ptr.get(*edges.get(v)).dest());
                    *edges.get_mut(v) += 1;
                } else {
                    stack.pop();
                    cycle.push(v);
                }
            }
            if !cycle.is_empty() {
                cycle.reverse();

                let cycle_slice = cycle.as_slice();
                let end = cycle_slice.len() - 1;

                write_idx = cycles
                    .write_slice(write_idx, &cycle_slice[..end])
                    .ok_or_else(|| -> Box<dyn std::error::Error> {
                        "error couldn't slice mmap to write cycle".into()
                    })?;

                self.trail_index.push(write_idx);
                cycle.clear();
            }
        }

        self.euler_trails.flush()?;

        // cleanup cache
        self.g.cleanup_cache(CacheFile::EulerTrail)?;

        Ok(())
    }

    #[cfg(feature = "bench")]
    pub fn get_throughput_factor(&self) -> usize {
        self.g.width()
    }
}

#[derive(Clone)]
struct FindDisjointSetsEulerTrails {
    trails: Vec<((usize, usize, usize), usize)>,
    cycle_check: bool,
}

impl FindDisjointSetsEulerTrails {
    pub fn new(trails: &mut [(usize, usize, usize)]) -> Self {
        trails.sort_unstable_by_key(|(trail_id, _, _)| *trail_id);
        FindDisjointSetsEulerTrails {
            trails: trails
                .iter()
                .map(|(trail, parent_trail, pos)| ((*trail, *parent_trail, *pos), *parent_trail))
                .collect::<Vec<((usize, usize, usize), usize)>>(),
            cycle_check: false,
        }
    }

    fn cycle_b(t: &mut [((usize, usize, usize), usize)], visited: &mut [bool], i: usize) -> usize {
        if t[i].0.0 == t[i].1 {
            return t[i].1;
        }

        if visited[t[i].0.0] {
            t[i].1 = i;
            t[i].0.1 = i;
            t[i].0.2 = 0;
            i
        } else {
            visited[t[i].0.0] = true;
            t[i].1 = Self::cycle_b(t, visited, t[i].1);
            t[i].1
        }
    }

    fn cycle_break(&mut self, i: usize) -> usize {
        let mut visited = vec![false; self.trails.len()];
        let t = &mut self.trails;
        Self::cycle_b(t.as_mut_slice(), visited.as_mut_slice(), i)
    }

    pub fn cycle_check(&mut self) {
        if self.cycle_check {
            return;
        }

        // build tree
        for (id, _) in self.clone().trails.iter().enumerate() {
            self.cycle_break(id);
        }
        // functions as an union find in tree
        for (id, _) in self.clone().trails.iter().enumerate() {
            self.cycle_break(id);
        }

        self.cycle_check = true;
    }
}

#[cfg(test)]
mod test {
    use crate::test_common::get_or_init_dataset_cache_entry;
    use crate::trails::verify_trails;

    use super::*;
    use paste::paste;
    use std::path::Path;

    macro_rules! graph_tests {
        ($($name:ident => $path:expr ,)*) => {
            $(
                paste! {
                    #[test]
                    fn [<hierholzers_euler_trails_ $name>]() -> Result<(), Box<dyn std::error::Error>> {
                        generic_test($path)
                    }
                }
            )*
        }
    }

    fn generic_test<P: AsRef<Path>>(path: P) -> Result<(), Box<dyn std::error::Error>> {
        let graph =
            GraphMemoryMap::init(get_or_init_dataset_cache_entry(path.as_ref())?, Some(16))?;
        let het = AlgoHierholzer::new(&graph)?;

        verify_trails(&graph, het.euler_trails, het.trail_index)
    }

    // generate test cases from dataset
    graph_tests! {
        ggcat_1_5 => "../ggcat/graphs/random_graph_1_5.lz4",
        ggcat_2_5 => "../ggcat/graphs/random_graph_2_5.lz4",
        ggcat_3_5 => "../ggcat/graphs/random_graph_3_5.lz4",
        ggcat_4_5 => "../ggcat/graphs/random_graph_4_5.lz4",
        ggcat_5_5 => "../ggcat/graphs/random_graph_5_5.lz4",
        ggcat_6_5 => "../ggcat/graphs/random_graph_6_5.lz4",
        ggcat_7_5 => "../ggcat/graphs/random_graph_7_5.lz4",
        ggcat_8_5 => "../ggcat/graphs/random_graph_8_5.lz4",
        ggcat_9_5 => "../ggcat/graphs/random_graph_9_5.lz4",
        ggcat_1_10 => "../ggcat/graphs/random_graph_1_10.lz4",
        // ggcat_2_10 => "../ggcat/graphs/random_graph_2_10.lz4",
        ggcat_3_10 => "../ggcat/graphs/random_graph_3_10.lz4",
        ggcat_4_10 => "../ggcat/graphs/random_graph_4_10.lz4",
        ggcat_5_10 => "../ggcat/graphs/random_graph_5_10.lz4",
        // ggcat_6_10 => "../ggcat/graphs/random_graph_6_10.lz4",
        // ggcat_7_10 => "../ggcat/graphs/random_graph_7_10.lz4",
        // ggcat_8_10 => "../ggcat/graphs/random_graph_8_10.lz4",
        // ggcat_9_10 => "../ggcat/graphs/random_graph_9_10.lz4",
        // ggcat_8_15 => "../ggcat/graphs/random_graph_8_15.lz4",
        // ggcat_9_15 => "../ggcat/graphs/random_graph_9_15.lz4",
        // … add the rest
    }
}
