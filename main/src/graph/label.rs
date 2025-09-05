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

#[sparkx_label]
#[repr(C)]
pub struct NodeLabel {
    label: usize,
}
impl N for NodeLabel {
    fn new(v: usize) -> Self {
        NodeLabel { label: v }
    }
}
impl N for () {
    fn new(_v: usize) {}
}
