use crate::{
    geometry::{
        Coordinates,
        mesh::PrimitiveMesh,
        ntree::{Orthotree, leaf::morton::Morton},
    },
    math::TensorRank1,
};
use std::{array::from_fn, collections::HashMap, ops::AddAssign};

impl<const D: usize, const N: usize, const I: usize, T, U, V> From<Orthotree<D, N, T, U>>
    for PrimitiveMesh<D, I, D, N, V>
where
    T: AddAssign + Copy + Into<u64>,
    V: Copy + From<usize>,
{
    fn from(orthotree: Orthotree<D, N, T, U>) -> Self {
        let mut coord_map: HashMap<u64, usize> = HashMap::new();
        let mut coords: Vec<TensorRank1<D, I>> = Vec::new();
        let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
        let connectivity: Vec<[V; N]> = orthotree
            .leaves
            .iter()
            .map(|leaf| {
                from_fn(|i| {
                    let face = i & face_mask;
                    let morton_i = (i & !face_mask) | (face ^ (face >> 1));
                    let vertex: [T; D] = from_fn(|ax| {
                        if (morton_i >> ax) & 1 == 1 {
                            let mut c = leaf.corner[ax];
                            c += leaf.length;
                            c
                        } else {
                            leaf.corner[ax]
                        }
                    });
                    let key = vertex.morton();
                    if let Some(&idx) = coord_map.get(&key) {
                        V::from(idx)
                    } else {
                        let idx = coords.len();
                        coords.push(
                            from_fn(|ax| {
                                let v: u64 = vertex[ax].into();
                                v as f64
                            })
                            .into(),
                        );
                        coord_map.insert(key, idx);
                        V::from(idx)
                    }
                })
            })
            .collect();
        (connectivity, Coordinates::from(coords)).into()
    }
}
