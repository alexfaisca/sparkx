#![feature(int_roundings)]
#![feature(once_cell_try)]

mod shared_slice;
#[cfg(any(test, bench))]
mod test_common;
mod utils;

pub mod centralities;
pub mod communities;
pub mod generic_edge;
pub mod generic_memory_map;
pub mod k_core;
pub mod k_truss;
pub mod trails;
