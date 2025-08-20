pub use graph_derive::{GenericEdge, GenericEdgeType};

use bitfield::bitfield;
use bytemuck::{Pod, Zeroable};
use rustworkx_core::petgraph::EdgeType;
use std::{
    cmp::{Eq, Ord, PartialEq, PartialOrd},
    convert,
    fmt::{Debug, Display},
};

/// Describes the behavior edge types must exhibit to be used by the tool.
///
/// Given an struct this trait may be automatically implemented using the provided procedural macro `GenericEdgeType`.
#[allow(dead_code)]
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
    + EdgeType
    + Send
    + Sync
{
    /// Edge label getter.
    fn label(&self) -> usize;
    /// Edge label setter.
    fn set_label(&mut self, label: u64);
}

/// Describes the behavior edges must exhibit to be used by the tool.
///
/// Given an implicitly packed struct this trait may be automatically implemented using the provided procedural macro `GenericEdge`.
#[allow(dead_code)]
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
    /// Constructor from an <<edge_dest: u64>> and an <<edge_type: u64>>
    fn new(edge_dest: u64, edge_type: u64) -> Self;
    /// Edge destiny node setter from a <<new_edge_dest: u64>>.
    fn set_edge_dest(&mut self, new_edge_dest: u64) -> &mut Self;
    /// Edge type setter from a <<new_edge_type: u64>>.
    fn set_edge_type(&mut self, new_edge_type: u64) -> &mut Self;
    /// Edge destiny node getter.
    fn dest(&self) -> usize;
    /// Edge type getter.
    fn e_type(&self) -> T;
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
    pub struct TinyLabelTinyEdge(u16);
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
    pub struct TinyLabelCompactEdge(u32);
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
    pub struct TinyLabelStandardEdge(u64);
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
    pub struct CompactColorCompactEdge(u64);
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
    pub struct StandardColorStandardEdge(u128);
    impl BitAnd;
    impl BitOr;
    impl BitXor;
    impl new;
    u128;
    u64, edge_type, set_edge_type: 63, 0;
    u64, dest_node, set_dest_node: 127, 64;
}

#[derive(GenericEdge)]
#[repr(C)]
pub struct UnlabeledTinyEdge {
    dest_node: u16,
    #[edge_type]
    label: VoidLabel,
}

#[derive(GenericEdge)]
#[repr(C)]
pub struct UnlabeledCompactEdge {
    dest_node: u32,
    #[edge_type]
    label: VoidLabel,
}

#[derive(GenericEdge)]
#[repr(C)]
pub struct UnlabeledStandardEdge {
    dest_node: u64,
    #[edge_type]
    label: VoidLabel,
}

#[derive(GenericEdgeType)]
#[repr(C)]
struct VoidLabel; // zero-sized edge

#[derive(GenericEdgeType)]
#[repr(C)]
pub enum TinyEdgeType {
    FF,
    FR,
    RF,
    RR,
}

#[derive(GenericEdgeType)]
#[generic_edge_type(is_directed = "true")]
#[repr(C)]
pub struct SubStandardColoredEdgeType {
    color: usize,
}

#[derive(GenericEdgeType)]
#[repr(C)]
pub struct ColoredEdgeType {
    color: usize,
}

impl convert::From<u64> for VoidLabel {
    fn from(_v: u64) -> Self {
        VoidLabel
    }
}

impl convert::From<VoidLabel> for u64 {
    fn from(_v: VoidLabel) -> u64 {
        0u64
    }
}

impl convert::From<usize> for VoidLabel {
    fn from(_v: usize) -> Self {
        Self
    }
}

impl convert::From<VoidLabel> for usize {
    fn from(_v: VoidLabel) -> usize {
        0usize
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    // static type debug
    const fn _static_checker<
        U: GenericEdgeType,
        T: GenericEdge<U>
            + Copy
            + Clone
            + PartialEq
            + Eq
            + Debug
            + Display
            + PartialOrd
            + Ord
            + Pod
            + Zeroable,
    >() {
    }
    const _: () = _static_checker::<TinyEdgeType, TinyLabelTinyEdge>();
    const _: () = _static_checker::<TinyEdgeType, TinyLabelCompactEdge>();
    const _: () = _static_checker::<TinyEdgeType, TinyLabelStandardEdge>();
    const _: () = _static_checker::<SubStandardColoredEdgeType, CompactColorCompactEdge>();
    const _: () = _static_checker::<ColoredEdgeType, StandardColorStandardEdge>();
    const _: () = _static_checker::<VoidLabel, UnlabeledTinyEdge>();
    const _: () = _static_checker::<VoidLabel, UnlabeledCompactEdge>();
    const _: () = _static_checker::<VoidLabel, UnlabeledStandardEdge>();

    #[test]
    fn named_struct_auto_builder() {
        #[repr(C)]
        #[derive(GenericEdge)]
        struct NamedTest1 {
            pub dest_node: i32,
            pub edge_type: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest1>();

        let t1_test = NamedTest1::new(142351, 124141);
        let t1_expect = NamedTest1 {
            dest_node: 142351,
            edge_type: SubStandardColoredEdgeType { color: 124141usize },
        };
        assert!(
            t1_test == t1_expect,
            "named struct proc macro not recognizing implicit fields auto builder"
        );
        assert!(
            t1_test.dest() == 142351,
            "named struct proc macro implicit fields auto dest getter malfunction"
        );
        assert!(
            t1_test.e_type() == SubStandardColoredEdgeType::from(124141usize),
            "named struct proc macro implicit fields auto e_type getter malfunction"
        );
    }

    #[test]
    fn named_struct_auto_default() {
        #[repr(C)]
        #[derive(GenericEdge)]
        struct NamedTest2 {
            pub dest_node: i32,
            pub edge_type: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest2>();

        let mut t2_test = NamedTest2::default();
        let t2_expect = NamedTest2 {
            dest_node: 0,
            edge_type: SubStandardColoredEdgeType { color: 0 },
        };
        assert!(
            t2_test == t2_expect,
            "named struct proc macro not recognizing implicit fields auto default"
        );
        assert!(
            t2_test.dest() == 0,
            "named struct proc macro implicit fields auto dest getter malfunction"
        );
        assert!(
            t2_test.e_type() == SubStandardColoredEdgeType::from(0usize),
            "named struct proc macro implicit fields auto e_type getter malfunction"
        );
        t2_test.set_edge_dest(1);
        assert!(
            t2_test.dest() == 1,
            "named struct proc macro implicit fields auto set_edge_dest malfunction"
        );
        t2_test.set_edge_type(2352);
        assert!(
            t2_test.e_type() == SubStandardColoredEdgeType::from(2352u64),
            "named struct proc macro implicit fields auto set_edge_type malfunction"
        );
    }

    #[test]
    fn named_struct_auto_builder_annotated_fields() {
        #[repr(C)]
        #[derive(GenericEdge)]
        struct NamedTest3 {
            #[edge_dest]
            pub edge_type: i32,
            #[edge_type]
            pub node_dest: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest3>();
        let t3_test = NamedTest3::new(142351, 124141);
        let t3_expect = NamedTest3 {
            edge_type: 142351,
            node_dest: SubStandardColoredEdgeType { color: 124141usize },
        };
        assert!(
            t3_test == t3_expect,
            "named struct proc macro not recognizing auto builder annotated fields"
        );
        assert!(
            t3_test.dest() == 142351,
            "named struct proc macro annotated fields dest getter malfunction"
        );
        assert!(
            t3_test.e_type() == SubStandardColoredEdgeType::from(124141usize),
            "named struct proc macro annotated fields e_type getter malfunction"
        );
    }

    #[test]
    fn named_struct_auto_builder_partial_annotated_fields() {
        #[repr(C)]
        #[derive(GenericEdge)]
        struct NamedTest4 {
            pub dest_node: i32,
            #[edge_type]
            pub color: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest4>();
        let t4_test = NamedTest4::new(932351, 51241);
        let t4_expect = NamedTest4 {
            dest_node: 932351,
            color: SubStandardColoredEdgeType { color: 51241usize },
        };
        assert!(
            t4_test == t4_expect,
            "named struct proc macro not recognizing auto builder partial annotated fields"
        );
        assert!(
            t4_test.dest() == 932351,
            "named struct proc macro partial annotated fields dest getter malfunction"
        );
        assert!(
            t4_test.e_type() == SubStandardColoredEdgeType::from(51241usize),
            "named struct proc macro partial annotated fields e_type getter malfunction"
        );
    }

    #[test]
    fn named_struct_auto_default_partial_annotated_fields() {
        #[repr(C)]
        #[derive(GenericEdge)]
        struct NamedTest5 {
            pub dest_node: i32,
            #[edge_type]
            pub color: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest5>();
        let mut t5_test = NamedTest5::default();
        let mut t5_expect = NamedTest5 {
            dest_node: 0,
            color: SubStandardColoredEdgeType { color: 0 },
        };
        assert!(
            t5_test == t5_expect,
            "named struct proc macro not recognizing auto default partial annotated fields"
        );
        t5_test.set_edge_type(956);
        t5_expect.color.color = 956;
        assert!(
            t5_test == t5_expect,
            "named struct proc macro not recognizing partial annotated fields auto e_type setter"
        );
        t5_test.set_edge_dest(7);
        t5_expect.dest_node = 7;
        assert!(
            t5_test == t5_expect,
            "named struct proc macro not recognizing partial annotated fields auto dest getter"
        );
    }

    #[test]
    fn named_struct_auto_default_annotated_fields_custom_getter_setter() {
        #[repr(C)]
        #[derive(GenericEdge)]
        struct NamedTest6 {
            #[edge_dest]
            pub dest: i32,
            #[edge_type(getter = "c", setter = "random_setter_name_12242")]
            pub color: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest6>();

        impl NamedTest6 {
            pub fn c(&self) -> SubStandardColoredEdgeType {
                SubStandardColoredEdgeType { color: 3 }
            }
            pub fn random_setter_name_12242(
                &mut self,
                _blah: SubStandardColoredEdgeType,
            ) -> &mut Self {
                self.color = SubStandardColoredEdgeType { color: 555555 };
                self
            }
        }

        let mut t6_test = NamedTest6::default();
        let t6_expect = NamedTest6::new(0, 0);
        assert!(
            t6_test == t6_expect,
            "named struct proc macro not recognizing auto default or builder annotated fields custom getter setter"
        );
        t6_test.set_edge_type(1);
        assert!(
            t6_test.e_type() == SubStandardColoredEdgeType { color: 3 }
                && t6_test.color.color == 555555,
            "named struct proc macro annotated fields custom getter setter e_type getter or setter malfunction"
        );
    }

    #[test]
    fn named_struct_auto_default_annotated_fields_custom_douche_getter_setter() {
        #[repr(C)]
        #[derive(GenericEdge)]
        struct NamedTest7 {
            #[edge_dest(getter = "banana", setter = "savannah")]
            pub pi: i32,
            #[edge_type(setter = "random_setter_name_12242")]
            pub euler: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest7>();

        impl NamedTest7 {
            pub fn banana(&self) -> usize {
                usize::try_from(self.pi).unwrap()
            }
            pub fn savannah(&mut self, baboon: i32) -> &mut Self {
                self.pi = baboon;
                self
            }
            pub fn random_setter_name_12242(
                &mut self,
                blah: SubStandardColoredEdgeType,
            ) -> &mut Self {
                self.euler = blah;
                self
            }
        }

        let mut t7_test = NamedTest7::default();
        let t7_expect = NamedTest7::new(0, 0);
        assert!(
            t7_test == t7_expect,
            "named struct proc macro not recognizing auto default or builder annotated fields custom douche getter setter"
        );
        t7_test.set_edge_type(1);
        assert!(
            t7_test.e_type() == SubStandardColoredEdgeType { color: 1 } && t7_test.euler.color == 1,
            "named struct proc macro annotated fields custom douche getter setter e_type getter or setter malfunction"
        );
        t7_test.set_edge_dest(1);
        assert!(
            t7_test.dest() == 1usize && t7_test.pi == 1i32,
            "named struct proc macro annotated fields custom douche getter setter dest getter or setter malfunction"
        );
    }

    #[test]
    fn named_struct_auto_default_annotated_fields_custom_phony_builder_default_getter_setter() {
        #[repr(C)]
        #[derive(GenericEdge)]
        #[generic_edge(new = "new_test", default = "default_test")]
        struct NamedTest8 {
            #[edge_dest(getter = "banana", setter = "savannah")]
            pub pi: i32,
            #[edge_type(setter = "random_setter_name_12242")]
            pub euler: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest8>();

        impl NamedTest8 {
            pub fn new_test(_e: u64, _t: u64) -> Self {
                NamedTest8 {
                    pi: 53234,
                    euler: SubStandardColoredEdgeType { color: 1234 },
                }
            }
            pub fn default_test() -> Self {
                NamedTest8 {
                    pi: 53234,
                    euler: SubStandardColoredEdgeType { color: 1234 },
                }
            }
            pub fn banana(&self) -> usize {
                usize::try_from(self.pi).unwrap()
            }
            pub fn savannah(&mut self, baboon: i32) -> &mut Self {
                self.pi = baboon;
                self
            }
            pub fn random_setter_name_12242(
                &mut self,
                blah: SubStandardColoredEdgeType,
            ) -> &mut Self {
                self.euler = blah;
                self
            }
        }

        let mut t8_test = NamedTest8::default();
        let t8_expect = NamedTest8::new(353214, 142312421);
        assert!(
            t8_test == t8_expect,
            "named struct proc macro not recognizing auto default or builder annotated fields custom phony builder default getter setter"
        );
        t8_test.set_edge_type(1);
        assert!(
            t8_test.e_type() == SubStandardColoredEdgeType { color: 1 } && t8_test.euler.color == 1,
            "named struct proc macro annotated fields custom phony builder default getter setter e_type getter or setter malfunction"
        );
        t8_test.set_edge_dest(1);
        assert!(
            t8_test.dest() == 1usize && t8_test.pi == 1i32,
            "named struct proc macro annotated fields custom phony builder default getter setter dest getter or setter malfunction"
        );
    }

    #[test]
    fn named_struct_auto_default_annotated_fields_custom_builder_default_getter_setter() {
        #[repr(C)]
        #[derive(GenericEdge)]
        #[generic_edge(builder = "new", default = "default")]
        struct NamedTest9 {
            #[edge_dest(getter = "banana", setter = "savannah")]
            pub pi: i32,
            #[edge_type(setter = "random_setter_name_12242")]
            pub euler: SubStandardColoredEdgeType,
        }
        // static type debug
        _static_checker::<SubStandardColoredEdgeType, NamedTest9>();

        impl NamedTest9 {
            pub fn new(e: u64, t: u64) -> Self {
                NamedTest9 {
                    pi: i32::try_from(e).unwrap(),
                    euler: SubStandardColoredEdgeType { color: t as usize },
                }
            }
            pub fn default() -> Self {
                NamedTest9 {
                    pi: 0,
                    euler: SubStandardColoredEdgeType { color: 0 },
                }
            }
            pub fn banana(&self) -> usize {
                usize::try_from(self.pi).unwrap()
            }
            pub fn savannah(&mut self, baboon: i32) -> &mut Self {
                self.pi = baboon;
                self
            }
            pub fn random_setter_name_12242(
                &mut self,
                blah: SubStandardColoredEdgeType,
            ) -> &mut Self {
                self.euler = blah;
                self
            }
        }
        let t9_test = NamedTest9::new(12414, 141241);
        assert!(
            t9_test
                == NamedTest9 {
                    pi: 12414i32,
                    euler: SubStandardColoredEdgeType { color: 141241usize },
                },
            "named struct proc macro not recognizing auto builder annotated fields custom builder default getter setter"
        );

        let t9_test = NamedTest9::default();
        let t9_expect = NamedTest9::new(0, 0);
        assert!(
            t9_test == t9_expect,
            "named struct proc macro not recognizing auto default annotated fields custom builder default getter setter"
        );
    }

    #[test]
    fn unnamed_struct_len_one() {
        bitfield! {
        #[derive(GenericEdge)]
        #[edge_type(setter = "set_edge_type", getter = "edge_type", real_type = "ColoredEdgeType")]
        #[edge_dest(setter = "set_dest_node", getter = "dest_node", real_type = "u64")]
        #[repr(C)]
            pub struct UnnamedTest1(u128);
            impl BitAnd;
            impl BitOr;
            impl BitXor;
            impl new;
            u128;
            u64, edge_type, set_edge_type: 63, 0;
            u64, dest_node, set_dest_node: 127, 64;
        }
        // static type debug
        _static_checker::<ColoredEdgeType, UnnamedTest1>();

        let t10_test = UnnamedTest1::new(12414, 141241);
        assert!(
            t10_test.dest_node() == 141241 && t10_test.edge_type() == 12414,
            "unnnamed struct lenght one proc macro not recognizing self builder"
        );
        let t10_test = <UnnamedTest1 as GenericEdge<ColoredEdgeType>>::new(12414, 141241);
        assert!(
            t10_test.dest_node() == 12414 && t10_test.edge_type() == 141241,
            "unnnamed struct lenght one proc macro not recognizing auto builder"
        );
        let t10_test = UnnamedTest1::default();
        let t10_expect = UnnamedTest1::new(0, 0);
        assert!(
            t10_test == t10_expect,
            "unnamed struct lenght one proc macro not recognizing auto default"
        );
        let mut t10_test = <UnnamedTest1 as GenericEdge<ColoredEdgeType>>::new(12414, 141241);
        assert!(
            t10_test.dest() == 12414 && t10_test.e_type() == ColoredEdgeType { color: 141241 },
            "unnnamed struct lenght one proc macro not recognizing auto dest or e_type getter"
        );
        t10_test.set_edge_type(333);
        t10_test.set_edge_dest(123);
        assert!(
            t10_test.dest() == 123 && t10_test.e_type() == ColoredEdgeType { color: 333 },
            "unnnamed struct lenght one proc macro not recognizing auto dest or e_type setter"
        );
    }

    #[test]
    fn unnamed_struct_len_one_custom_builder_default_setter_getter() {
        bitfield! {
        #[derive(GenericEdge)]
        #[generic_edge(constructor = "new_test", from_void = "default")]
        #[edge_type(setter = "savannah", getter = "edge_type", real_type = "ColoredEdgeType")]
        #[edge_dest(setter = "random_setter_name_12242", getter = "banana", real_type = "u64")]
        #[repr(C)]
            pub struct UnnamedTest2(u128);
            impl BitAnd;
            impl BitOr;
            impl BitXor;
            impl new;
            u128;
            u64, edge_type, set_edge_type: 63, 0;
            u64, dest_node, set_dest_node: 127, 64;
        }
        // static type debug
        _static_checker::<ColoredEdgeType, UnnamedTest2>();

        impl UnnamedTest2 {
            pub fn new_test(_e: u64, _t: u64) -> Self {
                UnnamedTest2::new(111, 111)
            }
            pub fn default() -> Self {
                UnnamedTest2::new(112, 112)
            }
            pub fn banana(&self) -> usize {
                123
            }
            pub fn savannah(&mut self, _baboon: u64) -> &mut Self {
                self.set_dest_node(0);
                self
            }
            pub fn random_setter_name_12242(&mut self, _blah: u64) -> &mut Self {
                self.set_edge_type(123123);
                self
            }
        }

        let t11_test = UnnamedTest2::default();
        // fully qualified syntax as struct constructor has same identifier as trait constructor
        assert!(
            t11_test.edge_type() == 112
                && t11_test.dest_node() == 112
                && t11_test.dest() == 123
                && t11_test.e_type() == ColoredEdgeType { color: 112 },
            "unnnamed struct lenght one proc macro not recognizing custom default or getters"
        );
        let mut t11_test = <UnnamedTest2 as GenericEdge<ColoredEdgeType>>::new(99035432, 995226);
        assert!(
            t11_test.edge_type() == 111
                && t11_test.dest_node() == 111
                && t11_test.dest() == 123
                && t11_test.edge_type() == 111,
            "unnnamed struct lenght one proc macro not recognizing custom builder or getters"
        );

        // to avoid using fully qualified syntax define struct setter methods with different
        // identifiers than `GenericEdge` trait methods
        <UnnamedTest2 as GenericEdge<ColoredEdgeType>>::set_edge_type(&mut t11_test, 43124);
        <UnnamedTest2 as GenericEdge<ColoredEdgeType>>::set_edge_dest(&mut t11_test, 124351);
        assert!(
            t11_test.edge_type() == 123123 && t11_test.dest_node() == 0,
            "unnnamed struct lenght one proc macro not recognizing custom setters"
        );
    }

    #[test]
    fn unnamed_struct_len_two_auto() {
        #[repr(C)]
        #[derive(GenericEdge)]
        pub struct UnnamedTest3(u64, ColoredEdgeType);
        // static type debug
        _static_checker::<ColoredEdgeType, UnnamedTest3>();

        let t12_test = UnnamedTest3::new(357, 753);
        assert!(
            t12_test.0 == 357 && t12_test.1 == ColoredEdgeType { color: 753 },
            "unnnamed struct lenght two proc macro not recognizing auto builder"
        );
        let t12_test = UnnamedTest3::default();
        let t12_expect = UnnamedTest3::new(0, 0);
        assert!(
            t12_test == t12_expect,
            "unnnamed struct lenght two proc macro not recognizing auto default"
        );
        let mut t12_test = UnnamedTest3::new(12414, 141241);
        assert!(
            t12_test.dest() == 12414 && t12_test.e_type() == ColoredEdgeType { color: 141241 },
            "unnnamed struct lenght two proc macro not recognizing auto getters"
        );
        t12_test.set_edge_type(311);
        t12_test.set_edge_dest(123);
        assert!(
            t12_test.dest() == 123 && t12_test.e_type() == ColoredEdgeType { color: 311 },
            "unnnamed struct lenght two proc macro not recognizing auto setters"
        );
    }

    // pub unnamed_struct_len_two_full_custom() {
    //
    //     #[repr(C)]
    //     #[derive(GenericEdge)]
    //     #[generic_edge(constructor = "new_test", from_void = "default")]
    //     #[edge_type(
    //         setter = "savannah",
    //         getter = "edge_type",
    //         real_type = "ColoredEdgeType"
    //     )]
    //     #[edge_dest(
    //         setter = "random_setter_name_12242",
    //         getter = "banana",
    //         real_type = "u64"
    //     )]
    //     pub struct UnnamedTest4(u64, ColoredEdgeType);
    //
    //     impl UnnamedTest4 {
    //         pub fn new_test(_e: u64, _t: u64) -> Self {
    //             UnnamedTest4::new(111, 111)
    //         }
    //         pub fn default() -> Self {
    //             UnnamedTest4::new(112, 112)
    //         }
    //         // manual getters and setter for unnamed not implemented
    //         // pub fn banana(&self) -> usize {
    //         //     123
    //         // }
    //         // pub fn savannah(&mut self, _baboon: u64) -> &mut Self {
    //         //     self.0 = 0;
    //         //     self
    //         // }
    //         // pub fn random_setter_name_12242(&mut self, _blah: u64) -> &mut Self {
    //         //     self.1 = ColoredEdgeType { color: 123123 };
    //         //     self
    //         // }
    //     }
    // }
}
