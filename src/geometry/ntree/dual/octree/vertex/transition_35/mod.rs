use crate::geometry::ntree::{Octree, dual::octree::N};

pub fn template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<usize>,
    U: Copy + Into<usize>,
{
    for (index, node) in tree.nodes.iter().enumerate() {
        if !node.is_leaf() {
            continue;
        }
        for bits in 0..N {
            let (bit_a, bit_b, bit_c) = (bits & 1, (bits >> 1) & 1, (bits >> 2) & 1);
            let toward = |a: usize, b: usize, c: usize| a + 2 * b + 4 * c;
            let leaf = |cell: U, orthant: usize| tree.leaves(&tree.nodes[cell.into()])[orthant];
            if let Some(cell_a) = node.facets[bit_a]
                && let Some(cell_b) = node.facets[2 + bit_b]
                && let Some(cell_c) = node.facets[4 + bit_c]
                && let Some(cell_ab) = tree.nodes[cell_a.into()].facets[2 + bit_b]
                && let Some(cell_ac) = tree.nodes[cell_a.into()].facets[4 + bit_c]
                && let Some(cell_bc) = tree.nodes[cell_b.into()].facets[4 + bit_c]
                && let Some(cell_abc) = tree.nodes[cell_ab.into()].facets[4 + bit_c]
                && let Some(mid_a) = leaf(cell_a, toward(1 - bit_a, bit_b, bit_c))
                && let Some(mid_b) = leaf(cell_b, toward(bit_a, 1 - bit_b, bit_c))
                && let Some(mid_c) = leaf(cell_c, toward(bit_a, bit_b, 1 - bit_c))
                && let Some(mid_ab) = leaf(cell_ab, toward(1 - bit_a, 1 - bit_b, bit_c))
                && let Some(mid_ac) = leaf(cell_ac, toward(1 - bit_a, bit_b, 1 - bit_c))
                && let Some(mid_bc) = leaf(cell_bc, toward(bit_a, 1 - bit_b, 1 - bit_c))
                && let Some(&child) = tree.nodes[cell_abc.into()]
                    .orthants()
                    .map(|orthants| &orthants[toward(1 - bit_a, 1 - bit_b, 1 - bit_c)])
                && let Some(fine) = leaf(child, toward(1 - bit_a, 1 - bit_b, 1 - bit_c))
            {
                let corner = center_nodes[index];
                let mid_a = center_nodes[mid_a.into()];
                let mid_b = center_nodes[mid_b.into()];
                let mid_c = center_nodes[mid_c.into()];
                let mid_ab = center_nodes[mid_ab.into()];
                let mid_ac = center_nodes[mid_ac.into()];
                let mid_bc = center_nodes[mid_bc.into()];
                let fine = center_nodes[fine.into()];
                if (bit_a + bit_b + bit_c) % 2 == 1 {
                    connectivity.push([corner, mid_a, mid_ab, mid_b, mid_c, mid_ac, fine, mid_bc]);
                } else {
                    connectivity.push([corner, mid_b, mid_ab, mid_a, mid_c, mid_bc, fine, mid_ac]);
                }
            }
        }
    }
}
