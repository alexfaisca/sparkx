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
    Copy
    + Clone
    + Default
    + Debug
    + Display
    + PartialEq
    + Eq
    + Zeroable
    + From<u64>
    + From<usize>
    + Send
    + Sync
{
    /// edge_label getter
    fn label(&self) -> usize;
}

/// describes the behavior edges must exhibit to be used by the tool
pub trait GenericEdge<T: GenericEdgeType>:
    Copy
    + Clone
    + Default
    + Debug
    + Display
    + PartialEq
    + Eq
    + PartialOrd
    + Ord
    + Pod
    + Zeroable
    + Send
    + Sync
{
    /// constructor from an <<edge_dest: u64>> and an <<edge_type: u64>>
    fn new(edge_dest: u64, edge_type: u64) -> Self;
    /// edge_dest setter from a <<new_edge_dest: u64>>
    fn set_edge_dest(&mut self, new_edge_dest: u64) -> &mut Self;
    /// edge_type setter from a <<new_edge_type: u64>>
    fn set_edge_type(&mut self, new_edge_type: u64) -> &mut Self;
    /// edge_dest getter
    fn dest(&self) -> usize;
    /// edge_type getter
    fn e_type(&self) -> T;
}

pub fn _test_proc_macro_capabilities() {
    #[repr(C)]
    #[derive(GenericEdge)]
    struct NamedTest1 {
        pub dest_node: i32,
        pub edge_type: SubStandardColoredEdgeType,
    }

    let t1_test = NamedTest1::new(142351, 124141);
    let t1_expect = NamedTest1 {
        dest_node: 142351,
        edge_type: SubStandardColoredEdgeType { color: 124141usize },
    };
    assert!(
        t1_test == t1_expect,
        "named struct (NamedTest1) proc macro not recognizing implicit fields"
    );
    let mut t1_test = NamedTest1::default();
    assert!(
        t1_test
            == NamedTest1 {
                dest_node: 0,
                edge_type: SubStandardColoredEdgeType { color: 0 }
            },
        "named struct (NamedTest1) proc macro implicit fields default malfunction"
    );
    assert!(
        t1_test.dest() == 0,
        "named struct (NamedTest1) proc macro implicit fields dest() malfunction"
    );
    assert!(
        t1_test.e_type() == SubStandardColoredEdgeType::from(0usize),
        "named struct (NamedTest1) proc macro implicit fields e_type() malfunction"
    );
    t1_test.set_edge_dest(1);
    assert!(
        t1_test.dest() == 1,
        "named struct (NamedTest1) proc macro implicit fields  set_edge_dest() malfunction"
    );
    t1_test.set_edge_type(2352);
    assert!(
        t1_test.e_type() == SubStandardColoredEdgeType::from(2352u64),
        "named struct (NamedTest1) proc macro implicit fields set_edge_type() malfunction"
    );
    #[repr(C)]
    #[derive(GenericEdge)]
    struct NamedTest2 {
        #[edge_dest]
        pub edge_type: i32,
        #[edge_type]
        pub node_dest: SubStandardColoredEdgeType,
    }
    let t1_test = NamedTest2::new(142351, 124141);
    let t1_expect = NamedTest2 {
        edge_type: 142351,
        node_dest: SubStandardColoredEdgeType { color: 124141usize },
    };
    assert!(
        t1_test == t1_expect,
        "named struct (NamedTest1) proc macro not recognizing annotated fields"
    );
    #[repr(C)]
    #[derive(GenericEdge)]
    struct NamedTest3 {
        pub dest_node: i32,
        #[edge_type]
        pub color: SubStandardColoredEdgeType,
    }
    let t1_test = NamedTest3::new(142351, 124141);
    let t1_expect = NamedTest3 {
        dest_node: 142351,
        color: SubStandardColoredEdgeType { color: 124141usize },
    };
    assert!(
        t1_test == t1_expect,
        "named struct (NamedTest1) proc macro not recognizing single implicit fields new"
    );
    let mut t1_test = NamedTest3::default();
    let mut t1_expect = NamedTest3 {
        dest_node: 0,
        color: SubStandardColoredEdgeType { color: 0usize },
    };
    assert!(
        t1_test == t1_expect,
        "named struct (NamedTest1) proc macro not recognizing single implicit fields default"
    );
    t1_test.set_edge_type(956);
    t1_expect.color.color = 956;
    assert!(
        t1_test == t1_expect,
        "named struct (NamedTest1) proc macro not recognizing single implicit fields auto setter"
    );
    #[repr(C)]
    #[derive(GenericEdge)]
    struct NamedTest4 {
        #[edge_dest]
        pub dest: i32,
        #[edge_type(getter = "c", setter = "random_setter_name_12242")]
        pub color: SubStandardColoredEdgeType,
    }

    impl NamedTest4 {
        pub fn c(&self) -> SubStandardColoredEdgeType {
            SubStandardColoredEdgeType { color: 3 }
        }
        pub fn random_setter_name_12242(&mut self, _blah: SubStandardColoredEdgeType) -> &mut Self {
            self.color = SubStandardColoredEdgeType { color: 555555 };
            self
        }
    }

    let mut t1_test = NamedTest4::default();
    let t1_expect = NamedTest4::new(0, 0);
    assert!(
        t1_test == t1_expect,
        "named struct (NamedTest1) proc macro not recognizing annotated fields automatic & default constructors"
    );
    t1_test.set_edge_type(1);
    assert!(
        t1_test.e_type() == SubStandardColoredEdgeType { color: 3 }
            && t1_test.color.color == 555555,
        "named struct (NamedTest1) proc macro not recognizing annotated fields manual getters & setters"
    );

    #[repr(C)]
    #[derive(GenericEdge)]
    struct NamedTest5 {
        #[edge_dest(getter = "banana", setter = "savannah")]
        pub pi: i32,
        #[edge_type(setter = "random_setter_name_12242")]
        pub euler: SubStandardColoredEdgeType,
    }

    impl NamedTest5 {
        pub fn banana(&self) -> usize {
            usize::try_from(self.pi).unwrap()
        }
        pub fn savannah(&mut self, baboon: i32) -> &mut Self {
            self.pi = baboon;
            self
        }
        pub fn random_setter_name_12242(&mut self, blah: SubStandardColoredEdgeType) -> &mut Self {
            self.euler = blah;
            self
        }
    }

    let mut t1_test = NamedTest5::default();
    let t1_expect = NamedTest5::new(0, 0);
    assert!(
        t1_test == t1_expect,
        "named struct (NamedTest1) proc macro not recognizing annotated fields automatic & default constructors"
    );
    t1_test.set_edge_type(1);
    assert!(
        t1_test.e_type() == SubStandardColoredEdgeType { color: 1 } && t1_test.euler.color == 1,
        "named struct (NamedTest1) proc macro not recognizing annotated fields manual getters & setters"
    );
    t1_test.set_edge_dest(1);
    assert!(
        t1_test.dest() == 1usize && t1_test.pi == 1i32,
        "named struct (NamedTest1) proc macro not recognizing annotated fields manual getters & setters"
    );
}

#[repr(C)]
#[derive(GenericEdge)]
pub struct Test {
    #[edge_dest(getter = "_get_d", t = "u32")]
    pub dest_node: u32,
    #[edge_type(getter = "_get_t", t = "SubStandardColoredEdgeType")]
    pub edge_type: SubStandardColoredEdgeType,
}

impl Test {
    fn _get_d(&self) -> u32 {
        self.dest_node
    }
    fn _get_t(&self) -> SubStandardColoredEdgeType {
        self.edge_type
    }
    fn _set_d(&mut self, d: u32) -> &mut Self {
        self.dest_node = d;
        self
    }
    fn _set_t(mut self, t: SubStandardColoredEdgeType) -> Self {
        self.edge_type = t;
        self
    }
}

bitfield! {
#[derive(GenericEdge)]
#[edge_type(setter = "set_edge_type", getter = "edge_type", real_type = "TinyEdgeType")]
#[edge_dest(setter = "set_dest_node", getter = "dest_node", real_type = "u64")]
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
#[edge_type(setter = "set_edge_type", getter = "edge_type", real_type = "TinyEdgeType")]
#[edge_dest(setter = "set_dest_node", getter = "dest_node", real_type = "u64")]
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
#[edge_type(setter = "set_edge_type", getter = "edge_type", real_type = "TinyEdgeType")]
#[edge_dest(setter = "set_dest_node", getter = "dest_node", real_type = "u64")]
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
#[edge_type(setter = "set_edge_type", getter = "edge_type", real_type = "SubStandardColoredEdgeType")]
#[edge_dest(setter = "set_dest_node", getter = "dest_node", real_type = "u32")]
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
#[edge_type(setter = "set_edge_type", getter = "edge_type", real_type = "ColoredEdgeType")]
#[edge_dest(setter = "set_dest_node", getter = "dest_node", real_type = "u64")]
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
