use super::{GenericEdge, GenericEdgeType, GraphFile, GraphMemoryMap, cache::GraphCache};
use crate::shared_slice::SharedSliceMut;

use fst::Streamer;
use std::fs::OpenOptions;

#[allow(dead_code)]
impl<EdgeType, Edge> GraphMemoryMap<EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    /// Applies the mask: fn(usize) -> bool function to each node id and returns the resulting subgraph.
    ///
    /// The resulting subgraph is that wherein only the set of nodes, `S ⊂ V`, of the nodes for whose the output of mask is true, as well as, only the set of edges coming from and going to nodes in `S`[^1].
    ///
    /// [^1]: the node ids of the subgraph may not, and probably will not, correspond to the original node identifiers, efffectively, it will be a whole new graph.
    pub(super) fn apply_mask_to_nodes(
        &mut self,
        mask: fn(usize) -> bool,
        identifier: Option<&str>,
    ) -> Result<GraphMemoryMap<EdgeType, Edge>, Box<dyn std::error::Error>> {
        let node_count = self.size();

        let id = match identifier.map(|id| id.to_string()) {
            Some(id) => id.to_string(),
            None => {
                let seq = self.exports_fetch_increment()?;
                self.graph_cache.build_subgraph_cache_id(seq)?
            }
        };

        // build subgraph's cache entries' filenames
        let e_fn = GraphCache::<EdgeType, Edge>::build_graph_filename(&id, GraphFile::Edges, None)?;
        let i_fn = GraphCache::<EdgeType, Edge>::build_graph_filename(&id, GraphFile::Index, None)?;
        let ml_fn =
            GraphCache::<EdgeType, Edge>::build_graph_filename(&id, GraphFile::Metalabel, None)?;

        if node_count > 1 {
            // helper counter
            let hc_fn = self.build_helper_filename(0)?;
            // helper indexer
            let hi_fn = self.build_helper_filename(1)?;

            // allocate |V| + 1 usize's to store the beginning and end offsets for each node's edges
            let mut edge_count =
                SharedSliceMut::<usize>::abst_mem_mut(&hc_fn, self.offsets_size(), true)?;
            // allocate |V| usize's to store each node's new id if present in the subgraph
            let mut node_index = SharedSliceMut::<usize>::abst_mem_mut(&hi_fn, self.size(), true)?;

            let mut curr_node_index: usize = 0;
            let mut curr_edge_count: usize = 0;
            *edge_count.get_mut(0) = curr_edge_count;
            // iterate over |V|
            for u in 0..self.size() {
                if mask(u) {
                    *node_index.get_mut(u) = curr_node_index;
                    curr_node_index += 1;
                    let neighbours = self.neighbours(u)?.filter(|x| mask(x.dest())).count();
                    curr_edge_count += neighbours;
                    *edge_count.get_mut(u + 1) = curr_edge_count;
                } else {
                    *node_index.get_mut(u) = usize::MAX;
                    *edge_count.get_mut(u + 1) = curr_edge_count;
                }
            }

            let mut metalabel_stream = self.metalabels.stream();

            let mut edges = SharedSliceMut::<Edge>::abst_mem_mut(&e_fn, curr_edge_count, true)?;
            let mut index =
                SharedSliceMut::<usize>::abst_mem_mut(&i_fn, curr_node_index + 1, true)?;
            let metalabel_file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&ml_fn)?;

            let mut build = fst::MapBuilder::new(&metalabel_file)?;

            // write nodes in order of lexicographically ordered metalabels to avoid sorting metaabels
            // FIXME: what is more costly random page accesses or sorting build merging metalabel fst?
            *index.get_mut(0) = 0;
            while let Some((metalabel, node_id)) = metalabel_stream.next() {
                let id = node_id as usize;
                if mask(id) {
                    // write index file for next node (id + 1)
                    let new_id = *node_index.get(id);
                    *index.get_mut(new_id + 1) = *edge_count.get(id + 1);
                    // write edge file node
                    edges
                        .write_slice(
                            *edge_count.get(id),
                            self.neighbours(id)?
                                .filter(|x| mask(x.dest()))
                                .collect::<Vec<Edge>>()
                                .as_slice(),
                        )
                        .ok_or("error writing edges for node {id}")?;
                    // write fst for node
                    build.insert(metalabel, new_id as u64)?;
                }
            }
            build.finish()?;
        } else {
            // if graph is empty allocate empty for its empty subgraph
            SharedSliceMut::<Edge>::abst_mem_mut(&e_fn, 0, true)?;
            let mut i = SharedSliceMut::<usize>::abst_mem_mut(&i_fn, 1, true)?;
            // store end of offsets in index entry at |V| (empty graph --- offsets end at 0)
            *i.get_mut(0) = 0;
            let metalabel_file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&ml_fn)?;

            let build = fst::MapBuilder::new(&metalabel_file)?;
            build.finish()?;
        }

        // finalize by initialozing a GraphCache instance for the subgraph and building it
        let cache: GraphCache<EdgeType, Edge> = GraphCache::open(&ml_fn, None)?;
        self.cleanup_helpers()?;
        GraphMemoryMap::init_from_cache(cache, Some(self.thread_count))
    }
}
