use std::{
    convert,
    fmt::{Debug, Display},
};

use bytemuck::Zeroable;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum EdgeType {
    FF,
    FR,
    RF,
    RR,
}

unsafe impl Zeroable for EdgeType {}

impl Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl convert::From<u8> for EdgeType {
    fn from(v: u8) -> Self {
        match v {
            0 => EdgeType::FF,
            1 => EdgeType::FR,
            2 => EdgeType::RF,
            3 => EdgeType::RR,
            _ => panic!("Invalid edge type"),
        }
    }
}

impl convert::From<EdgeType> for u8 {
    fn from(v: EdgeType) -> u8 {
        match v {
            EdgeType::FF => 0,
            EdgeType::FR => 1,
            EdgeType::RF => 2,
            EdgeType::RR => 3,
        }
    }
}

impl convert::From<u16> for EdgeType {
    fn from(v: u16) -> Self {
        match v {
            0 => EdgeType::FF,
            1 => EdgeType::FR,
            2 => EdgeType::RF,
            3 => EdgeType::RR,
            _ => panic!("Invalid edge type"),
        }
    }
}

impl convert::From<EdgeType> for u16 {
    fn from(v: EdgeType) -> u16 {
        match v {
            EdgeType::FF => 0,
            EdgeType::FR => 1,
            EdgeType::RF => 2,
            EdgeType::RR => 3,
        }
    }
}

impl convert::From<u32> for EdgeType {
    fn from(v: u32) -> Self {
        match v {
            0 => EdgeType::FF,
            1 => EdgeType::FR,
            2 => EdgeType::RF,
            3 => EdgeType::RR,
            _ => panic!("Invalid edge type"),
        }
    }
}

impl convert::From<EdgeType> for u32 {
    fn from(v: EdgeType) -> u32 {
        match v {
            EdgeType::FF => 0,
            EdgeType::FR => 1,
            EdgeType::RF => 2,
            EdgeType::RR => 3,
        }
    }
}

impl convert::From<u64> for EdgeType {
    fn from(v: u64) -> Self {
        match v {
            0 => EdgeType::FF,
            1 => EdgeType::FR,
            2 => EdgeType::RF,
            3 => EdgeType::RR,
            _ => panic!("Invalid edge type"),
        }
    }
}

impl convert::From<EdgeType> for u64 {
    fn from(v: EdgeType) -> u64 {
        match v {
            EdgeType::FF => 0,
            EdgeType::FR => 1,
            EdgeType::RF => 2,
            EdgeType::RR => 3,
        }
    }
}
