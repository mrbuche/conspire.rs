use crate::geometry::ntree::{
    Orthotree,
    node::{Kind, split::Split},
};
use std::{array::from_fn, ops::Add};

const fn mirror_facet(facet: usize) -> usize {
    facet ^ 1
}

pub(crate) const fn insert_bit(x: usize, axis: usize, bit: usize) -> usize {
    let low_mask = (1usize << axis) - 1;
    let low = x & low_mask;
    let high = x >> axis;
    low | (bit << axis) | (high << (axis + 1))
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy,
{
    fn nodes_on_face(facet: usize) -> [usize; L] {
        from_fn(|k| insert_bit(k, facet / 2, facet % 2))
    }
    fn nodes_on_other_face(face: usize) -> [usize; L] {
        Self::nodes_on_face(mirror_facet(face))
    }
    pub fn subdivide(&mut self, index: U) -> Result<(), &'static str> {
        let indices = from_fn(|n| (self.len() + n).into());
        let mut orthants = self[index].subdivide(indices)?;
        for (facet, node_facet) in self[index].facets.into_iter().enumerate() {
            if let Some(facet_node) = node_facet
                && let Some(neighbors) = self[facet_node].orthants().copied()
            {
                for (node, neighbor) in Self::nodes_on_face(facet)
                    .into_iter()
                    .zip(Self::nodes_on_other_face(facet))
                {
                    if orthants[node].facets[facet].is_none() {
                        orthants[node].facets[facet] = Some(neighbors[neighbor])
                    } else {
                        panic!("temporary to assess need for Option<>")
                    }
                    if self[neighbors[neighbor]].facets[mirror_facet(facet)].is_none() {
                        self[neighbors[neighbor]].facets[mirror_facet(facet)] = Some(indices[node])
                    } else {
                        panic!("temporary to assess need for Option<>")
                    }
                }
            }
        }
        self.extend(orthants);
        self[index].kind = Kind::Tree(indices);
        self[index].value = None;
        Ok(())
    }
}
