use crate::trails::{bfs_visit, dfs_visit};

use super::*;

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> dfs_visit::Graph
    for GraphMemoryMap<N, E, Ix>
{
    fn s(&self) -> usize {
        self.size()
    }

    fn w(&self) -> usize {
        self.width()
    }

    fn neigh(&self, u: usize) -> Box<[usize]> {
        self[u..u + 1].to_vec().into_boxed_slice()
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

impl<N: crate::graph::N, E: crate::graph::E, Ix: crate::graph::IndexType> bfs_visit::Graph
    for GraphMemoryMap<N, E, Ix>
{
    fn s(&self) -> usize {
        self.size()
    }

    fn w(&self) -> usize {
        self.width()
    }

    fn neigh(&self, u: usize) -> Box<[usize]> {
        self[u..u + 1].to_vec().into_boxed_slice()
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
