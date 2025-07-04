// pub trait AbstractGraph {
//     // number of nodes in graph
//     fn size(&self) -> u64;
//
//     // number of edges in graph
//     fn widht(&self) -> u64;
//
//     // get node by id
//     fn node(id: u64) -> impl AbstractNode;
//
//     fn neighbour_iter(id: u64) -> Iterator<impl AbstractNode>;
// }
//
// pub trait AbstractNode {
//     fn id(&self) -> u64;
//     fn edges(&self) -> &[impl AbstractEdge];
// }
//
// pub trait AbstractEdge {
//     fn dest_node(&self) -> u64;
// }
