use crate::{
    geometry::ntree::{
        Octree,
        dual::octree::{
            N,
            edge::transition_5::{EDGES, config},
        },
    },
    math::Scalar,
};

pub fn template<T, U>(
    tree: &Octree<T, U>,
    center_nodes: &[usize],
    connectivity: &mut Vec<[usize; N]>,
) where
    T: Copy + Into<Scalar> + Into<usize>,
    U: Copy + Into<usize>,
{
    for (index, node) in tree.nodes.iter().enumerate() {
        if !node.is_leaf() {
            continue;
        }
        for &(facet_m, facet_n) in EDGES.iter() {
            let axis = 3 - (facet_m >> 1) - (facet_n >> 1);
            let corner: usize = node.corner[axis].into();
            let length: usize = node.length.into();
            if (corner + length).is_multiple_of(2 * length)
                && let Some(above) = node.facets[2 * axis + 1]
                && tree.nodes[above.into()].is_leaf()
                && let Some(config_a) = config(tree, node, index, facet_m, facet_n, center_nodes)
                && let Some(config_b) = config(
                    tree,
                    &tree.nodes[above.into()],
                    above.into(),
                    facet_m,
                    facet_n,
                    center_nodes,
                )
                && config_a.length == config_b.length
            {
                connectivity.push([
                    config_b.center,
                    config_b.n_lo,
                    config_b.ring_lo,
                    config_b.m_lo,
                    config_a.center,
                    config_a.n_hi,
                    config_a.ring_hi,
                    config_a.m_hi,
                ]);
            }
        }
    }
}
