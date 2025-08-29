#![feature(int_roundings)]
#![feature(once_cell_try)]
// This library can only be used in linux
#![cfg(target_os = "linux")]

mod shared_slice;
#[cfg(any(test, feature = "bench"))]
pub mod test_common;
pub mod utils;

pub mod centralities;
pub mod communities;
pub mod generic_edge;
pub mod generic_memory_map;
pub mod k_core;
pub mod k_truss;
pub mod trails;
