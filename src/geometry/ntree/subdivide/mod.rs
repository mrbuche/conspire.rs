use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    leaf::{Leaf, morton::Morton, split::Split},
};
use std::{
    array::from_fn,
    ops::{AddAssign, Rem, SubAssign},
};

pub enum Pairing {
    Generalized,
    Regular,
    None,
}

impl<const D: usize, const N: usize, T, U> Orthotree<D, N, T, U>
where
    T: AddAssign + Copy + Default + Into<u64> + PartialEq + Rem<Output = T> + Split + SubAssign,
    U: Copy,
{
    pub fn subdivide(&mut self, index: usize, pairing: Pairing) -> Result<(), OrthotreeError> {
        if index >= self.leaves.len() {
            return Err(OrthotreeError::IndexOutOfBounds);
        }
        match pairing {
            Pairing::None => self.subdivide_leaf(index),
            Pairing::Regular => {
                let corner = self.leaves[index].corner;
                let length = self.leaves[index].length;
                let mut parent_length = length;
                parent_length += length;
                let parent_corner: [T; D] = from_fn(|ax| {
                    if corner[ax] % parent_length == T::default() {
                        corner[ax]
                    } else {
                        let mut pc = corner[ax];
                        pc -= length;
                        pc
                    }
                });
                let sibling_corners: [[T; D]; N] = from_fn(|i| {
                    from_fn(|ax| {
                        if (i >> ax) & 1 == 1 {
                            let mut c = parent_corner[ax];
                            c += length;
                            c
                        } else {
                            parent_corner[ax]
                        }
                    })
                });
                let to_subdivide: Vec<usize> = sibling_corners
                    .iter()
                    .filter_map(|sc| {
                        let target = sc.morton();
                        let pos = self.leaves.partition_point(|l| l.corner.morton() < target);
                        if pos < self.leaves.len()
                            && self.leaves[pos].corner.morton() == target
                            && self.leaves[pos].length == length
                        {
                            Some(pos)
                        } else {
                            None
                        }
                    })
                    .collect();
                for j in to_subdivide.into_iter().rev() {
                    self.subdivide_leaf(j)?;
                }
                Ok(())
            }
            Pairing::Generalized => {
                unimplemented!();
            }
        }
    }
    fn subdivide_leaf(&mut self, index: usize) -> Result<(), OrthotreeError> {
        let corner = self.leaves[index].corner;
        let length = self.leaves[index].length;
        let data = self.leaves[index].data;
        let orthant_length = length.split();
        let orthants: [Leaf<D, T, U>; N] = from_fn(|i| {
            let mut orthant_corner = corner;
            orthant_corner
                .iter_mut()
                .enumerate()
                .for_each(|(axis, coord)| {
                    if (i >> axis) & 1 == 1 {
                        *coord += orthant_length;
                    }
                });
            Leaf {
                corner: orthant_corner,
                length: orthant_length,
                data,
            }
        });
        self.leaves.splice(index..=index, orthants);
        Ok(())
    }
}
