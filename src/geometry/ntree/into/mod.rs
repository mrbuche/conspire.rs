use crate::geometry::{Coordinate, Coordinates, mesh::Connectivity, ntree::Orthotree};
use std::{array::from_fn, collections::HashMap};

impl<const D: usize, const L: usize, const M: usize, const N: usize, U>
    From<Orthotree<D, L, M, N, u16, U>> for (Vec<[usize; N]>, Coordinates<D>)
{
    fn from(orthotree: Orthotree<D, L, M, N, u16, U>) -> Self {
        let mut coord_map: HashMap<u64, usize> = HashMap::new();
        let mut coords: Vec<Coordinate<D>> = Vec::new();
        let face_mask: usize = if D <= 2 { (1 << D) - 1 } else { 3 };
        let connectivity: Vec<[usize; N]> = orthotree
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
                        idx
                    } else {
                        let idx = coords.len();
                        coords.push(from_fn(|ax| vertex[ax] as f64).into());
                        coord_map.insert(key, idx);
                        idx
                    }
                })
            })
            .collect();
        (connectivity, coords.into())
    }
}

impl<const L: usize, const M: usize, U> From<Orthotree<2, L, M, 4, u16, U>>
    for (Connectivity, Coordinates<2>)
{
    fn from(orthotree: Orthotree<2, L, M, 4, u16, U>) -> Self {
        let (connectivity, coordinates): (Vec<[usize; 4]>, _) = orthotree.into();
        (
            Connectivity::Quadrilateral(connectivity.into()),
            coordinates,
        )
    }
}

impl<const L: usize, const M: usize, U> From<Orthotree<3, L, M, 8, u16, U>>
    for (Connectivity, Coordinates<3>)
{
    fn from(orthotree: Orthotree<3, L, M, 8, u16, U>) -> Self {
        let (connectivity, coordinates): (Vec<[usize; 8]>, _) = orthotree.into();
        (Connectivity::Hexahedral(connectivity.into()), coordinates)
    }
}
