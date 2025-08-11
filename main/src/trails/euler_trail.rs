use crate::generic_edge::*;
use crate::generic_memory_map::*;
use crate::shared_slice::*;
use crate::utils::*;

use crossbeam::thread;
use memmap2::{Mmap, MmapMut};
use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::Error,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU8, AtomicUsize, Ordering},
    },
    time::Instant,
};

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

        // FIXME: Ugly!

        // Break cycles in graph
        for (id, _i) in self.clone().trails.iter().enumerate() {
            self.cycle_break(id);
        }
        // Works as an union find in tree
        for (id, _i) in self.clone().trails.iter().enumerate() {
            self.cycle_break(id);
        }

        self.cycle_check = true;
    }
}

#[allow(dead_code)]
pub struct EulerTrail<'a, EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> {
    graph: &'a GraphMemoryMap<EdgeType, Edge>,
}

#[allow(dead_code)]
impl<'a, EdgeType, Edge> EulerTrail<'a, EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    pub fn new(graph: &'a GraphMemoryMap<EdgeType, Edge>) -> Result<Self, Error> {
        Ok(EulerTrail { graph })
    }

    fn create_memmapped_mut_slice_from_tmp_file<V>(
        filename: String,
        len: usize,
    ) -> Result<(SharedSliceMut<V>, MmapMut), Error> {
        let file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .read(true)
            .open(&filename)?;
        file.set_len((len * std::mem::size_of::<V>()) as u64)?;
        SharedSliceMut::<V>::from_file(&file)
    }

    fn create_memmapped_slice_from_tmp_file<V>(
        filename: String,
    ) -> Result<(SharedSlice<V>, Mmap), Error> {
        let file = OpenOptions::new().read(true).open(filename)?;
        SharedSlice::<V>::from_file(&file)
    }

    fn merge_euler_trails(
        &self,
        cycle_offsets: Vec<(usize, usize, usize)>,
    ) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
        let time = Instant::now();
        let mut trail_heads: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
        let mmap_fn = cache_file_name(self.graph.cache_index_filename(), FileType::EulerTmp, None)?;
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

        let euler_time = Instant::now();
        let mut keys_by_trail_size: Vec<(usize, usize)> =
            output_len.values().map(|&(a, b)| (a, b)).collect();
        keys_by_trail_size.sort_unstable_by_key(|(_, s)| std::cmp::Reverse(*s));

        for (idx, (head_trail, output_len)) in keys_by_trail_size.iter().enumerate() {
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

            let output_filename = cache_file_name(
                self.graph.cache_index_filename(),
                FileType::EulerPath,
                Some(idx),
            )?;
            let out =
                SharedSliceMut::<usize>::abst_mem_mut(output_filename.clone(), *output_len, true)?;

            let mut idx = 0;

            for (cycle, from, to) in stack.iter() {
                if *to <= *from {
                    continue;
                }
                let (_, t_begin, _) = cycle_offsets[*cycle];
                let begin = t_begin + *from;
                let end = t_begin + *to;

                out.shared_slice()
                    .write_shared_slice(cycles, idx, begin, end - begin);
                idx += end - begin;
            }

            out.flush_async()?;
        }
        println!("euler write {:?}", euler_time.elapsed());
        println!("euler trails merge {:?}", time.elapsed());

        cleanup_cache()?;

        Ok(keys_by_trail_size
            .iter()
            .enumerate()
            .map(|(idx, (_, size))| (idx as u64, *size as u64))
            .collect())
    }

    fn initialize_hierholzers_procedural_memory(
        &self,
        mmap: u8,
    ) -> Result<
        (
            AbstractedProceduralMemoryMut<AtomicUsize>,
            AbstractedProceduralMemoryMut<AtomicU8>,
        ),
        Box<dyn std::error::Error>,
    > {
        let node_count = self.graph.size() - 1;
        let threads = self.graph.thread_num();
        let thread_load = node_count.div_ceil(threads);

        let index_ptr = Arc::new(SharedSlice::<usize>::new(
            self.graph.index_ptr(),
            self.graph.size(),
        ));

        let template_fn = self.graph.cache_edges_filename();
        let e_fn = cache_file_name(template_fn.clone(), FileType::EulerTmp, Some(1))?;
        let c_fn = cache_file_name(template_fn.clone(), FileType::EulerTmp, Some(2))?;

        let edges = SharedSliceMut::<AtomicUsize>::abst_mem_mut(e_fn, node_count, mmap > 0)?;
        let count = SharedSliceMut::<AtomicU8>::abst_mem_mut(c_fn, node_count, mmap > 1)?;

        thread::scope(|scope| {
            for i in 0..threads {
                let index = Arc::clone(&index_ptr);
                let edges = &edges;
                let count = &count;
                let begin = std::cmp::min(thread_load * i, node_count);
                let end = std::cmp::min(begin + thread_load, node_count);

                scope.spawn(move |_| {
                    for k in begin..end {
                        edges.get(k).store(*index.get(k), Ordering::Relaxed);
                        count
                            .get(k)
                            .store((*index.get(k + 1) - *index.get(k)) as u8, Ordering::Relaxed);
                    }
                });
            }
        })
        .unwrap();
        edges.flush()?;
        count.flush()?;
        Ok((edges, count))
    }

    /// find Eulerian cycle and write sequence of node ids to memory-mapped file.
    /// num_threads controls parallelism level (defaults to 1, single-threaded).
    /// returns vec of (euler path file sequence number, file size(vytes)).
    pub fn find_eulerian_cycle(
        &self,
        mmap: u8,
    ) -> Result<Vec<(u64, u64)>, Box<dyn std::error::Error>> {
        let time = Instant::now();
        let node_count = match self.graph.size() - 1 {
            0 => panic!("Graph has no vertices"),
            i => i,
        };
        let graph_ptr = Arc::new(SharedSlice::<Edge>::new(
            self.graph.edges_ptr(),
            self.graph.width(),
        ));

        // The Vec<_> and MmapMut refs need to be in scope for the structures not to be deallocated
        let (edges, edge_count) = self.initialize_hierholzers_procedural_memory(mmap)?;

        // Atomic counter to pick next starting vertex for a new cycle
        let start_vertex_counter = Arc::new(AtomicUsize::new(0));
        // mmap to store disjoined trails for subsequent merging
        let filename = cache_file_name(self.graph.cache_fst_filename(), FileType::EulerTmp, None)?;
        let (mmap_slice, mmap) = Self::create_memmapped_mut_slice_from_tmp_file::<usize>(
            filename,
            self.graph.width() * 2,
        )?;
        let mmap_mutex = Mutex::new(mmap_slice);

        let cycle_offsets: Mutex<Vec<(usize, usize, usize)>> = std::sync::Mutex::new(vec![]);
        let mmap_offset: Mutex<usize> = Mutex::new(0);

        thread::scope(|scope| {
            for _ in 0..self.graph.thread_num() {
                let graph = Arc::clone(&graph_ptr);
                let next_edge = &edges;
                let edge_count = &edge_count;
                let start_vertex_counter = Arc::clone(&start_vertex_counter);
                let cycle_offsets = &cycle_offsets;
                let mmap_offset = &mmap_offset;
                let mmap = &mmap_mutex;

                // Spawn a thread
                scope.spawn(move |_| {
                    // find cycles until no unused edges remain
                    loop {
                        let start_v = loop {
                            let idx = start_vertex_counter
                                .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |x| {
                                    if x >= node_count {
                                        Some(0)
                                    } else {
                                        Some(x + 1)
                                    }
                                })
                                .unwrap_or(0);
                            if idx >= node_count {
                                break None;
                            }
                            if edge_count.get(idx).load(Ordering::Relaxed) > 0 {
                                break Some(idx);
                            }
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
                            if edge_count
                                .get(v)
                                .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |x| {
                                    if x > 0 {
                                        Some(x - 1)
                                    } else {
                                        None // don't update
                                    }
                                })
                                .is_ok()
                            {
                                let edge_idx = next_edge.get(v).fetch_add(1, Ordering::Relaxed);
                                stack.push(graph.get(edge_idx).dest());
                            } else {
                                stack.pop();
                                if cycle.is_empty() {
                                    // check if error of atomic
                                    if stack.is_empty() {
                                        continue;
                                    }
                                }
                                cycle.push(v);
                            }
                        }
                        if !cycle.is_empty() {
                            cycle.reverse();
                            let cycle = cycle.as_slice();
                            let mut cycle_stack = cycle_offsets.lock().unwrap();
                            let mut offset = mmap_offset.lock().unwrap();
                            let begin = *offset;
                            *offset += cycle.len();
                            let end = *offset;
                            let cycle_id = cycle_stack.len();
                            cycle_stack.push((cycle_id, begin, end));

                            // Write to mmap, this may not done concurrently
                            let mut mmap_guard = match mmap.lock() {
                                Ok(i) => i,
                                Err(e) => panic!("error mutex 1: {:?}", e),
                            };
                            let () = match mmap_guard.mut_slice(begin, end) {
                                Some(i) => i.copy_from_slice(cycle),
                                None => panic!("error couldn't slice mmap to write cycle"),
                            };
                        }
                    }
                });
            }
        })
        .unwrap();
        mmap.flush()?;
        let disjoint_cycles = cycle_offsets.into_inner().unwrap();
        // Euler trails are in the mem_mapped file
        println!("euler trails hierholzer's {:?}", time.elapsed());
        self.merge_euler_trails(disjoint_cycles)
    }
}
