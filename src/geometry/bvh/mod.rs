use crate::geometry::{BoundingBox, BoundingBoxUnion, Coordinate};

pub struct BoundingVolumeHierarchy<const D: usize, const I: usize> {
    items: Vec<usize>,
    nodes: Vec<Node<D, I>>,
}

pub struct Item<const D: usize, const I: usize> {
    bounding_box: BoundingBox<D, I>,
    centroid: Coordinate<D, I>,
    index: usize,
}

pub struct Node<const D: usize, const I: usize> {
    bounding_box: BoundingBox<D, I>,
    kind: NodeKind,
}

pub enum NodeKind {
    Leaf { start: usize, end: usize },
    Tree { left: usize, right: usize },
}

fn bbox_of_items<const D: usize, const I: usize>(items: &[Item<D, I>]) -> BoundingBox<D, I> {
    assert!(!items.is_empty());

    items.iter()
        .skip(1)
        .fold(items[0].bounding_box.clone(), |bbox, item| {
            bbox.union(&item.bounding_box)
        })
}

impl<const D: usize, const I: usize> BoundingVolumeHierarchy<D, I> {
    fn build_node(
        nodes: &mut Vec<Node<D, I>>,
        items_out: &mut Vec<usize>,
        items: &mut [Item<D, I>],
        leaf_size: usize,
    ) -> usize {
        let bounding_box = bbox_of_items(items);
        let node_index = nodes.len();

        nodes.push(Node {
            bounding_box: bounding_box.clone(),
            kind: NodeKind::Leaf { start: 0, end: 0 },
        });

        if items.len() <= leaf_size {
            let start = items_out.len();
            items_out.extend(items.iter().map(|item| item.index));
            let end = items_out.len();

            nodes[node_index] = Node {
                bounding_box,
                kind: NodeKind::Leaf { start, end },
            };

            return node_index;
        }

        let axis = bounding_box.longest_axis();

        items.sort_by(|a, b| {
            a.centroid[axis]
                .partial_cmp(&b.centroid[axis])
                .unwrap_or(Ordering::Equal)
        });

        let mid = items.len() / 2;
        let (left_items, right_items) = items.split_at_mut(mid);

        let left = Self::build_node(nodes, items_out, left_items, leaf_size);
        let right = Self::build_node(nodes, items_out, right_items, leaf_size);

        nodes[node_index] = Node {
            bounding_box,
            kind: NodeKind::Tree { left, right },
        };

        node_index
    }
}
