use crate::geometry::{
    bbox::BoundingBox,
    bvh::{
        BoundingVolumeHierarchy,
        node::{Node, NodeKind},
        primitive::Primitive,
    },
};

impl<const D: usize, T> BoundingVolumeHierarchy<D, T>
where
    T: Copy,
{
    pub fn build_node(&mut self, primitives: &mut [Primitive<D, T>], leaf_size: usize) -> usize {
        assert!(leaf_size > 0);
        assert!(!primitives.is_empty());
        let bounding_box = BoundingBox::from(&primitives[..]);
        let node_index = self.nodes.len();
        self.nodes.push(Node::from((
            &bounding_box,
            NodeKind::Leaf { start: 0, end: 0 },
        )));
        if primitives.len() <= leaf_size {
            let start = self.items.len();
            self.items
                .extend(primitives.iter().map(|primitive| primitive.index()));
            let end = self.items.len();
            self.nodes[node_index] = Node::from((bounding_box, NodeKind::Leaf { start, end }));
            return node_index;
        }
        let axis = bounding_box.longest_axis();
        primitives.sort_by(|a, b| a.centroid()[axis].partial_cmp(&b.centroid()[axis]).unwrap());
        let (left_primitives, right_primitives) = primitives.split_at_mut(primitives.len() / 2);
        let left = self.build_node(left_primitives, leaf_size);
        let right = self.build_node(right_primitives, leaf_size);
        self.nodes[node_index] = Node::from((bounding_box, NodeKind::Tree { left, right }));
        node_index
    }
}
