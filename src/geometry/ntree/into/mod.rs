use crate::{
    geometry::{Coordinates, mesh::PrimitiveMesh, ntree::Orthotree},
    math::TensorRank1,
};
use std::{array::from_fn, collections::HashMap};

impl<const D: usize, const M: usize, const N: usize, const I: usize, U, V>
    From<Orthotree<D, M, N, u16, U>> for PrimitiveMesh<D, I, D, N, V>
where
    V: Copy + From<usize>,
{
    fn from(orthotree: Orthotree<D, M, N, u16, U>) -> Self {
        let mut coord_map: HashMap<u64, usize> = HashMap::new();
        let mut coords: Vec<TensorRank1<D, I>> = Vec::new();
        let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
        let connectivity: Vec<[V; N]> = orthotree
            .nodes
            .iter()
            .filter(|node| node.is_leaf())
            .map(|node| {
                from_fn(|i| {
                    let face = i & face_mask;
                    let vertex_i = (i & !face_mask) | (face ^ (face >> 1));
                    let vertex: [u16; D] = from_fn(|ax| {
                        if (vertex_i >> ax) & 1 == 1 {
                            node.corner[ax] + node.length
                        } else {
                            node.corner[ax]
                        }
                    });
                    let key: u64 =
                        (0..D).fold(0u64, |acc, ax| acc | ((vertex[ax] as u64) << (16 * ax)));
                    if let Some(&idx) = coord_map.get(&key) {
                        V::from(idx)
                    } else {
                        let idx = coords.len();
                        coords.push(from_fn(|ax| vertex[ax] as f64).into());
                        coord_map.insert(key, idx);
                        V::from(idx)
                    }
                })
            })
            .collect();
        (connectivity, Coordinates::from(coords)).into()
    }
}
