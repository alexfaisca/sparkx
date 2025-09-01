#![feature(int_roundings)]
#![feature(once_cell_try)]
// This library can only be used in linux
#![cfg(target_os = "linux")]

mod shared_slice;

#[cfg(any(test, feature = "bench"))]
pub mod test_common;

pub mod centralities;
pub mod communities;
pub mod graph;
pub mod k_core;
pub mod k_truss;
pub mod trails;
pub mod utils;
