use super::{GenericEdge, GenericEdgeType, GraphMemoryMap};

#[allow(dead_code)]
impl<EdgeType, Edge> GraphMemoryMap<EdgeType, Edge>
where
    EdgeType: GenericEdgeType,
    Edge: GenericEdge<EdgeType>,
{
    #[inline(always)]
    pub(super) fn is_neighbour_impl(&self, u: usize, v: usize) -> Option<usize> {
        assert!(
            u < self.size(),
            "{} is not smaller than max node id |V| = {} --- node doesn't exist",
            u,
            self.size()
        );

        let mut floor = unsafe { (self.index.as_ptr() as *const usize).add(u).read() };
        let mut ceil = unsafe { (self.index.as_ptr() as *const usize).add(u + 1).read() };
        // binary search on neighbours w, where w < v
        loop {
            // may happen if u + 1 overflows
            if floor > ceil {
                return None;
            }
            let m = floor + (ceil - floor) / 2;
            let dest = unsafe { (self.graph.as_ptr() as *const Edge).add(m).read().dest() };
            match dest.cmp(&v) {
                std::cmp::Ordering::Equal => break Some(m),
                std::cmp::Ordering::Greater => ceil = m - 1,
                std::cmp::Ordering::Less => floor = m + 1,
            }
        }
    }

    pub(super) fn is_triangle_impl(&self, u: usize, v: usize, w: usize) -> Option<(usize, usize)> {
        let mut index_a = None;
        let mut index_b = None;
        let switch = v < u;

        if let Ok(mut iter) = self.neighbours(w) {
            loop {
                if let Some((index, n)) = iter._next_with_offset() {
                    if index_a.is_none() {
                        match (if switch { v } else { u }).cmp(&n.dest()) {
                            std::cmp::Ordering::Less => {
                                return None;
                            }
                            std::cmp::Ordering::Equal => {
                                if let Some(b) = index_b {
                                    return Some((index, b));
                                }
                                index_a = Some(index);
                            }
                            _ => {}
                        };
                    } else {
                        match (if switch { u } else { v }).cmp(&n.dest()) {
                            std::cmp::Ordering::Less => {
                                return None;
                            }
                            std::cmp::Ordering::Equal => {
                                if let Some(a) = index_a {
                                    return Some(if switch { (index, a) } else { (a, index) });
                                }
                            }
                            _ => {}
                        };
                    }
                } else {
                    return None;
                }
                if let Some((index, n)) = iter._next_back_with_offset() {
                    if index_b.is_none() {
                        match (if switch { u } else { v }).cmp(&n.dest()) {
                            std::cmp::Ordering::Greater => return None,
                            std::cmp::Ordering::Equal => {
                                if let Some(a) = index_a {
                                    return Some((a, index));
                                }
                                index_b = Some(index);
                            }
                            _ => {}
                        };
                    } else {
                        match (if switch { v } else { u }).cmp(&n.dest()) {
                            std::cmp::Ordering::Greater => return None,
                            std::cmp::Ordering::Equal => {
                                if let Some(b) = index_b {
                                    return Some(if switch { (b, index) } else { (index, b) });
                                }
                            }
                            _ => {}
                        };
                    }
                } else {
                    return None;
                }
            }
        }
        None
    }
}
