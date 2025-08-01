pub use graph_derive::{GenericEdge, GenericEdgeType};

use bitfield::bitfield;
use bytemuck::{Pod, Zeroable};
use std::{
    cmp::{Eq, Ord, PartialEq, PartialOrd},
    convert,
    fmt::{Debug, Display},
};

// procedural macro debug
const _: () = {
    const fn _static_checker<
        U: GenericEdgeType,
        T: GenericEdge<U>
            + Copy
            + Clone
            + PartialEq
            + Eq
            + std::fmt::Debug
            + std::fmt::Display
            + PartialOrd
            + Ord
            + Pod
            + Zeroable,
    >() {
    }
    _static_checker::<TinyEdgeType, TinyEdge>();
    _static_checker::<TinyEdgeType, SubStandardEdge>();
    _static_checker::<TinyEdgeType, StandardEdge>();
    _static_checker::<SubStandardColoredEdgeType, ColoredSubStandardEdge>();
    _static_checker::<ColoredEdgeType, ColouredStandardEdge>();
};

/// describes the behavior edge types must exhibit to be used by the tool
pub trait GenericEdgeType:
    Copy + Clone + Debug + Display + PartialEq + Eq + Zeroable + From<u64> + From<usize>
{
    /// edge label
    fn label(&self) -> usize;
}

/// describes the behavior edges must exhibit to be used by the tool
pub trait GenericEdge<T: GenericEdgeType>:
    Copy + Clone + Debug + Display + PartialEq + Eq + PartialOrd + Ord + Pod + Zeroable
{
    /// destiny node id
    fn dest(&self) -> usize;
    /// edge type
    fn e_type(&self) -> T;
}

bitfield! {
#[derive(GenericEdge)]
#[edge_type(getter = "edge_type", real_type = "TinyEdgeType")]
#[edge_dest(getter = "dest_node", real_type = "u16")]
#[repr(C)]
    pub struct TinyEdge(u16);
    impl BitAnd;
    impl BitOr;
    impl BitXor;
    impl new;
    u16;
    u64, edge_type, set_edge_type: 1, 0;
    u64, dest_node, set_dest_node: 15, 2;
}

bitfield! {
#[derive(GenericEdge)]
#[edge_type(getter = "edge_type", real_type = "TinyEdgeType")]
#[edge_dest(getter = "dest_node", real_type = "u32")]
#[repr(C)]
    pub struct SubStandardEdge(u32);
    impl BitAnd;
    impl BitOr;
    impl BitXor;
    impl new;
    u32;
    u64, edge_type, set_edge_type: 1, 0;
    u64, dest_node, set_dest_node: 31, 2;
}

bitfield! {
#[derive(GenericEdge)]
#[edge_type(getter = "edge_type", real_type = "TinyEdgeType")]
#[edge_dest(getter = "dest_node", real_type = "u64")]
#[repr(C)]
    pub struct StandardEdge(u64);
    impl BitAnd;
    impl BitOr;
    impl BitXor;
    impl new;
    u64;
    u64, edge_type, set_edge_type: 1, 0;
    u64, dest_node, set_dest_node: 63, 2;
}

bitfield! {
#[derive(GenericEdge)]
#[edge_type(getter = "edge_type", real_type = "SubStandardColoredEdgeType")]
#[edge_dest(getter = "dest_node", real_type = "u32")]
#[repr(C)]
    pub struct ColoredSubStandardEdge(u64);
    impl BitAnd;
    impl BitOr;
    impl BitXor;
    impl new;
    u64;
    u64, edge_type, set_edge_type: 31, 0;
    u64, dest_node, set_dest_node: 63, 32;
}

bitfield! {
#[derive(GenericEdge)]
#[edge_type(getter = "edge_type", real_type = "ColoredEdgeType")]
#[edge_dest(getter = "dest_node", real_type = "u64")]
#[repr(C)]
    pub struct ColouredStandardEdge(u128);
    impl BitAnd;
    impl BitOr;
    impl BitXor;
    impl new;
    u128;
    u64, edge_type, set_edge_type: 63, 0;
    u64, dest_node, set_dest_node: 127, 64;
}
#[derive(GenericEdgeType)]
#[repr(C)]
pub enum TinyEdgeType {
    FF,
    FR,
    RF,
    RR,
}

#[derive(GenericEdgeType)]
#[repr(C)]
pub struct SubStandardColoredEdgeType {
    color: usize,
}

#[derive(GenericEdgeType)]
#[repr(C)]
pub struct ColoredEdgeType {
    color: usize,
}

impl convert::From<u64> for TinyEdgeType {
    fn from(v: u64) -> Self {
        match v {
            0 => TinyEdgeType::FF,
            1 => TinyEdgeType::FR,
            2 => TinyEdgeType::RF,
            3 => TinyEdgeType::RR,
            _ => panic!("Invalid edge type"),
        }
    }
}

impl convert::From<TinyEdgeType> for u64 {
    fn from(v: TinyEdgeType) -> u64 {
        match v {
            TinyEdgeType::FF => 0,
            TinyEdgeType::FR => 1,
            TinyEdgeType::RF => 2,
            TinyEdgeType::RR => 3,
        }
    }
}

impl convert::From<usize> for TinyEdgeType {
    fn from(v: usize) -> Self {
        match v {
            0 => TinyEdgeType::FF,
            1 => TinyEdgeType::FR,
            2 => TinyEdgeType::RF,
            3 => TinyEdgeType::RR,
            _ => panic!("Invalid edge type"),
        }
    }
}

impl convert::From<TinyEdgeType> for usize {
    fn from(v: TinyEdgeType) -> usize {
        match v {
            TinyEdgeType::FF => 0,
            TinyEdgeType::FR => 1,
            TinyEdgeType::RF => 2,
            TinyEdgeType::RR => 3,
        }
    }
}

impl convert::From<u64> for SubStandardColoredEdgeType {
    fn from(v: u64) -> Self {
        SubStandardColoredEdgeType { color: v as usize }
    }
}

impl convert::From<SubStandardColoredEdgeType> for u64 {
    fn from(v: SubStandardColoredEdgeType) -> u64 {
        v.color as u64
    }
}

impl convert::From<usize> for SubStandardColoredEdgeType {
    fn from(v: usize) -> Self {
        SubStandardColoredEdgeType { color: v }
    }
}

impl convert::From<SubStandardColoredEdgeType> for usize {
    fn from(v: SubStandardColoredEdgeType) -> usize {
        v.color
    }
}

impl convert::From<u64> for ColoredEdgeType {
    fn from(v: u64) -> Self {
        ColoredEdgeType { color: v as usize }
    }
}

impl convert::From<ColoredEdgeType> for u64 {
    fn from(v: ColoredEdgeType) -> u64 {
        v.color as u64
    }
}

impl convert::From<usize> for ColoredEdgeType {
    fn from(v: usize) -> Self {
        ColoredEdgeType { color: v }
    }
}

impl convert::From<ColoredEdgeType> for usize {
    fn from(v: ColoredEdgeType) -> usize {
        v.color
    }
}
