use crate::{
    geometry::{
        Coordinates,
        ntree::{
            Orthotree,
            leaf::{Leaf, morton::Morton},
        },
    },
    math::TensorVec,
};
use std::{array::from_fn, collections::BTreeMap};

impl<const D: usize, const N: usize, const I: usize> From<(Coordinates<D, I>, f64)>
    for Orthotree<D, N, u16, ()>
{
    fn from((coordinates, min_length): (Coordinates<D, I>, f64)) -> Self {
        if coordinates.is_empty() {
            return Self {
                leaves: vec![Leaf {
                    corner: [0u16; D],
                    length: 1,
                    data: (),
                }],
            };
        }
        let mut min_coord: [f64; D] = from_fn(|_| f64::INFINITY);
        let mut max_coord: [f64; D] = from_fn(|_| f64::NEG_INFINITY);
        for point in &coordinates {
            for ax in 0..D {
                let v = point[ax];
                if v < min_coord[ax] {
                    min_coord[ax] = v;
                }
                if v > max_coord[ax] {
                    max_coord[ax] = v;
                }
            }
        }
        let max_extent = (0..D)
            .map(|ax| max_coord[ax] - min_coord[ax])
            .fold(0.0f64, f64::max);
        let levels = if max_extent <= 0.0 {
            0u16
        } else {
            (max_extent / min_length).log2().ceil().max(0.0) as u16
        };
        let root_length: u16 = 1 << levels;
        let center: [f64; D] = from_fn(|ax| (min_coord[ax] + max_coord[ax]) / 2.0);
        let root_corner = [0u16; D];
        let mut map: BTreeMap<u64, Leaf<D, u16, ()>> = BTreeMap::new();
        map.insert(
            root_corner.morton(),
            Leaf {
                corner: root_corner,
                length: root_length,
                data: (),
            },
        );
        // Sort by Morton code for cache-local BTreeMap access; dedup since
        // two coords mapping to the same cell produce identical subdivisions.
        let mut int_coords: Vec<[u16; D]> = (&coordinates)
            .into_iter()
            .map(|point| {
                from_fn(|ax| {
                    let v = ((point[ax] - center[ax]) / min_length
                        + (root_length as f64) / 2.0)
                        .floor() as i64;
                    v.clamp(0, root_length as i64 - 1) as u16
                })
            })
            .collect();
        int_coords.sort_unstable_by_key(|c| c.morton());
        int_coords.dedup();
        for int_coord in &int_coords {
            loop {
                let target = int_coord.morton();
                let key = match map.range(..=target).next_back() {
                    Some((&k, leaf)) => {
                        let contained = (0..D).all(|ax| {
                            let end = leaf.corner[ax] + leaf.length;
                            leaf.corner[ax] <= int_coord[ax] && int_coord[ax] < end
                        });
                        if contained { k } else { break }
                    }
                    None => break,
                };
                let leaf = *map.get(&key).unwrap();
                if leaf.length <= 1 {
                    break;
                }
                let length = leaf.length;
                let corner = leaf.corner;
                let parent_length = length + length;
                let parent_corner: [u16; D] = from_fn(|ax| {
                    if corner[ax].is_multiple_of(parent_length) {
                        corner[ax]
                    } else {
                        corner[ax] - length
                    }
                });
                let orthant_length = length / 2;
                for i in 0..N {
                    let sc: [u16; D] = from_fn(|ax| {
                        if (i >> ax) & 1 == 1 {
                            parent_corner[ax] + length
                        } else {
                            parent_corner[ax]
                        }
                    });
                    let sc_key = sc.morton();
                    if map.get(&sc_key).is_some_and(|l| l.length == length) {
                        map.remove(&sc_key);
                        for j in 0..N {
                            let oc: [u16; D] = from_fn(|ax| {
                                if (j >> ax) & 1 == 1 {
                                    sc[ax] + orthant_length
                                } else {
                                    sc[ax]
                                }
                            });
                            map.insert(
                                oc.morton(),
                                Leaf {
                                    corner: oc,
                                    length: orthant_length,
                                    data: (),
                                },
                            );
                        }
                    }
                }
            }
        }
        Self {
            leaves: map.into_values().collect(),
        }
    }
}
