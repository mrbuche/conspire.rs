use crate::geometry::{
    Coordinates,
    bbox::BoundingBox,
    bvh::node::{Node, NodeKind},
};

const KIND: NodeKind = NodeKind::Leaf { start: 0, end: 1 };

fn bbox() -> BoundingBox<3, 0> {
    BoundingBox::from(Coordinates::from([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]))
}

#[test]
fn bbox_and_kind() {
    let _ = Node::from((bbox(), KIND));
}

#[test]
fn bbox_ref_and_kind() {
    let _ = Node::from((&bbox(), KIND));
}
