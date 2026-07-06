#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        bbox::BoundingBox,
        bvh::{
            BoundingVolumeHierarchy, Hit,
            node::{Node, NodeKind},
            primitive::Primitive,
            ray::Ray,
        },
    },
    math::Scalar,
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
        let mid = primitives.len() / 2;
        primitives.select_nth_unstable_by(mid, |a, b| {
            a.centroid()[axis].partial_cmp(&b.centroid()[axis]).unwrap()
        });
        let (left_primitives, right_primitives) = primitives.split_at_mut(mid);
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
        if !self.nodes.is_empty()
            && let Some(entry) = ray.intersects(self.nodes[0].bounding_box())
        {
            self.intersect_node(0, entry, ray, coordinates, elements, &mut hit);
        }
        hit
    }
    pub fn intersections(
        &self,
        ray: &Ray<3>,
        coordinates: &Coordinates<3>,
        elements: &[&[usize]],
    ) -> usize {
        let mut count = 0;
        if !self.nodes.is_empty() {
            self.count_node(0, ray, coordinates, elements, &mut count);
        }
        count
    }
    fn count_node(
        &self,
        node_index: usize,
        ray: &Ray<3>,
        coordinates: &Coordinates<3>,
        elements: &[&[usize]],
        count: &mut usize,
    ) {
        let node = &self.nodes[node_index];
        if ray.intersects(node.bounding_box()).is_none() {
            return;
        }
        match node.kind() {
            NodeKind::Leaf { start, end } => {
                self.items[*start..*end].iter().for_each(|&item| {
                    let element = elements[item];
                    if ray
                        .intersects_triangle(
                            &coordinates[element[0]],
                            &coordinates[element[1]],
                            &coordinates[element[2]],
                        )
                        .is_some()
                    {
                        *count += 1;
                    }
                });
            }
            NodeKind::Tree { left, right } => {
                self.count_node(*left, ray, coordinates, elements, count);
                self.count_node(*right, ray, coordinates, elements, count);
            }
        }
    }
    fn intersect_node(
        &self,
        node_index: usize,
        entry: Scalar,
        ray: &Ray<3>,
        coordinates: &Coordinates<3>,
        elements: &[&[usize]],
        hit: &mut Option<Hit>,
    ) {
        if hit
            .as_ref()
            .is_some_and(|closest| entry >= closest.distance())
        {
            return;
        }
        let node = &self.nodes[node_index];
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
            NodeKind::Tree { left, right } => {
                let left_entry = ray.intersects(self.nodes[*left].bounding_box());
                let right_entry = ray.intersects(self.nodes[*right].bounding_box());
                match (left_entry, right_entry) {
                    (Some(left_entry), Some(right_entry)) => {
                        let (near, near_entry, far, far_entry) = if left_entry <= right_entry {
                            (*left, left_entry, *right, right_entry)
                        } else {
                            (*right, right_entry, *left, left_entry)
                        };
                        self.intersect_node(near, near_entry, ray, coordinates, elements, hit);
                        self.intersect_node(far, far_entry, ray, coordinates, elements, hit);
                    }
                    (Some(left_entry), None) => {
                        self.intersect_node(*left, left_entry, ray, coordinates, elements, hit);
                    }
                    (None, Some(right_entry)) => {
                        self.intersect_node(*right, right_entry, ray, coordinates, elements, hit);
                    }
                    (None, None) => {}
                }
            }
        }
    }
    pub fn overlapping(&self, query: &BoundingBox<3>) -> Vec<usize> {
        let mut found = Vec::new();
        if !self.nodes.is_empty() {
            self.overlapping_node(0, query, &mut found);
        }
        found
    }
    fn overlapping_node(&self, node_index: usize, query: &BoundingBox<3>, found: &mut Vec<usize>) {
        let node = &self.nodes[node_index];
        if !query.overlaps(node.bounding_box()) {
            return;
        }
        match node.kind() {
            NodeKind::Leaf { start, end } => found.extend_from_slice(&self.items[*start..*end]),
            NodeKind::Tree { left, right } => {
                self.overlapping_node(*left, query, found);
                self.overlapping_node(*right, query, found);
            }
        }
    }
    pub fn closest_point(
        &self,
        point: &Coordinate<3>,
        coordinates: &Coordinates<3>,
        elements: &[&[usize]],
    ) -> Option<(Coordinate<3>, usize)> {
        let mut closest = None;
        if !self.nodes.is_empty() {
            self.closest_point_node(0, point, coordinates, elements, &mut closest);
        }
        closest.map(|(_, candidate, index)| (candidate, index))
    }
    fn closest_point_node(
        &self,
        node_index: usize,
        point: &Coordinate<3>,
        coordinates: &Coordinates<3>,
        elements: &[&[usize]],
        closest: &mut Option<(Scalar, Coordinate<3>, usize)>,
    ) {
        let node = &self.nodes[node_index];
        if closest.as_ref().is_some_and(|(distance, ..)| {
            point_box_distance_squared(point, node.bounding_box()) >= *distance
        }) {
            return;
        }
        match node.kind() {
            NodeKind::Leaf { start, end } => {
                self.items[*start..*end].iter().for_each(|&item| {
                    let element = elements[item];
                    let candidate = closest_point_on_triangle(
                        point,
                        &coordinates[element[0]],
                        &coordinates[element[1]],
                        &coordinates[element[2]],
                    );
                    let offset = &candidate - point;
                    let distance = &offset * &offset;
                    if closest
                        .as_ref()
                        .is_none_or(|(nearest, ..)| distance < *nearest)
                    {
                        *closest = Some((distance, candidate, item));
                    }
                });
            }
            NodeKind::Tree { left, right } => {
                let (near, far) =
                    if point_box_distance_squared(point, self.nodes[*left].bounding_box())
                        <= point_box_distance_squared(point, self.nodes[*right].bounding_box())
                    {
                        (*left, *right)
                    } else {
                        (*right, *left)
                    };
                self.closest_point_node(near, point, coordinates, elements, closest);
                self.closest_point_node(far, point, coordinates, elements, closest);
            }
        }
    }
}

fn point_box_distance_squared<const D: usize>(
    point: &Coordinate<D>,
    bounding_box: &BoundingBox<D>,
) -> Scalar {
    (0..D)
        .map(|axis| {
            let value = point[axis];
            let (low, high) = (bounding_box.minimum()[axis], bounding_box.maximum()[axis]);
            let delta = if value < low {
                low - value
            } else if value > high {
                value - high
            } else {
                0.0
            };
            delta * delta
        })
        .sum()
}

fn closest_point_on_triangle(
    point: &Coordinate<3>,
    a: &Coordinate<3>,
    b: &Coordinate<3>,
    c: &Coordinate<3>,
) -> Coordinate<3> {
    let ab = b - a;
    let ac = c - a;
    let ap = point - a;
    let d1 = &ab * &ap;
    let d2 = &ac * &ap;
    if d1 <= 0.0 && d2 <= 0.0 {
        return a.clone();
    }
    let bp = point - b;
    let d3 = &ab * &bp;
    let d4 = &ac * &bp;
    if d3 >= 0.0 && d4 <= d3 {
        return b.clone();
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        return a + &(&ab * (d1 / (d1 - d3)));
    }
    let cp = point - c;
    let d5 = &ab * &cp;
    let d6 = &ac * &cp;
    if d6 >= 0.0 && d5 <= d6 {
        return c.clone();
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        return a + &(&ac * (d2 / (d2 - d6)));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        return b + &(&(c - b) * ((d4 - d3) / ((d4 - d3) + (d5 - d6))));
    }
    let denominator = 1.0 / (va + vb + vc);
    &(a + &(&ab * (vb * denominator))) + &(&ac * (vc * denominator))
}
