use super::{E, N, sparkx_label};

#[sparkx_label]
pub struct VoidLabel {}
impl N for VoidLabel {
    fn new(_v: usize) -> Self {
        Self {}
    }
}
impl E for VoidLabel {
    fn new(_v: usize) -> Self {
        Self {}
    }
}

impl N for () {
    fn new(_v: usize) {}
}

impl E for () {
    fn new(_v: usize) {}
}

#[sparkx_label]
#[repr(C)]
pub struct NodeLabel {
    label: usize,
}

#[sparkx_label]
#[repr(C)]
pub struct EdgeLabel {
    label: usize,
}

impl N for NodeLabel {
    fn new(v: usize) -> Self {
        NodeLabel { label: v }
    }
}

impl E for EdgeLabel {
    fn new(v: usize) -> Self {
        EdgeLabel { label: v }
    }
}
