#[cfg(test)]
mod test;

use crate::geometry::{
    Coordinates,
    bbox::BoundingBox,
    bvh::{
        BoundingVolumeHierarchy, Hit,
        node::{Node, NodeKind},
        primitive::Primitive,
        ray::Ray,
    },
};

impl<const D: usize> BoundingVolumeHierarchy<D> {
    pub fn build_node(&mut self, primitives: &mut [Primitive<D>], leaf_size: usize) -> usize {
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

impl BoundingVolumeHierarchy<3> {
    pub fn intersect(
        &self,
        ray: &Ray<3>,
        coordinates: &Coordinates<3>,
        elements: &[&[usize]],
    ) -> Option<Hit> {
        let mut hit = None;
        if !self.nodes.is_empty() {
            self.intersect_node(0, ray, coordinates, elements, &mut hit);
        }
        hit
    }
    fn intersect_node(
        &self,
        node_index: usize,
        ray: &Ray<3>,
        coordinates: &Coordinates<3>,
        elements: &[&[usize]],
        hit: &mut Option<Hit>,
    ) {
        let node = &self.nodes[node_index];
        let entry = match ray.intersects(node.bounding_box()) {
            Some(entry) => entry,
            None => return,
        };
        if hit
            .as_ref()
            .is_some_and(|closest| entry >= closest.distance())
        {
            return;
        }
        match node.kind() {
            NodeKind::Leaf { start, end } => {
                self.items[*start..*end].iter().for_each(|&item| {
                    let element = elements[item];
                    if let Some(distance) = ray.intersects_triangle(
                        &coordinates[element[0]],
                        &coordinates[element[1]],
                        &coordinates[element[2]],
                    ) && hit
                        .as_ref()
                        .is_none_or(|closest| distance < closest.distance())
                    {
                        *hit = Some(Hit {
                            distance,
                            index: item,
                        });
                    }
                });
            }
            // TODO: visit the nearer child first for tighter pruning.
            NodeKind::Tree { left, right } => {
                self.intersect_node(*left, ray, coordinates, elements, hit);
                self.intersect_node(*right, ray, coordinates, elements, hit);
            }
        }
    }
}
