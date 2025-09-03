use super::*;

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> crate::trails::bfs_visit::Graph
    for GraphMemoryMap<EdgeType, Edge>
{
    fn s(&self) -> usize {
        self.size()
    }

    fn w(&self) -> usize {
        self.width()
    }

    fn neigh(&self, u: usize) -> Box<[usize]> {
        self[u..u + 1]
            .iter()
            .map(|e| e.dest())
            .collect::<Vec<usize>>()
            .into_boxed_slice()
    }

    fn cache_file_name(
        &self,
        target: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.build_cache_filename(target, seq)
    }

    fn cleanup(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>> {
        self.cleanup_cache(target)
    }
}

impl<EdgeType: GenericEdgeType, Edge: GenericEdge<EdgeType>> crate::trails::dfs_visit::Graph
    for GraphMemoryMap<EdgeType, Edge>
{
    fn s(&self) -> usize {
        self.size()
    }

    fn w(&self) -> usize {
        self.width()
    }

    fn neigh(&self, u: usize) -> Box<[usize]> {
        self[u..u + 1]
            .iter()
            .map(|e| e.dest())
            .collect::<Vec<usize>>()
            .into_boxed_slice()
    }

    fn cache_file_name(
        &self,
        target: CacheFile,
        seq: Option<usize>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        self.build_cache_filename(target, seq)
    }

    fn cleanup(&self, target: CacheFile) -> Result<(), Box<dyn std::error::Error>> {
        self.cleanup_cache(target)
    }
}
