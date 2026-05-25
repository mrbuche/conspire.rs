use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    leaf::{Leaf, split::Split},
};
use std::{array::from_fn, ops::AddAssign};

impl<const D: usize, const N: usize, T, U> Orthotree<D, N, T, U>
where
    T: AddAssign + Copy + Into<u64> + Split,
    U: Copy,
{
    pub fn subdivide(&mut self, index: usize) -> Result<(), OrthotreeError> {
        if index >= self.leaves.len() {
            return Err(OrthotreeError::IndexOutOfBounds);
        }
        let Leaf {
            corner,
            length,
            data,
        } = self.leaves.remove(index);
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
        let insert_pos = self
            .leaves
            .partition_point(|l| morton(&l.corner) < morton(&orthants[0].corner));
        for (offset, orthant) in orthants.into_iter().enumerate() {
            self.leaves.insert(insert_pos + offset, orthant);
        }
        Ok(())
    }
}

fn morton<const D: usize, T>(corner: &[T; D]) -> u64
where
    T: Copy + Into<u64>,
{
    let bits = 64 / D;
    let mut result = 0u64;
    for bit in 0..bits {
        for (axis, &coord) in corner.iter().enumerate() {
            if (coord.into() >> bit) & 1 == 1 {
                result |= 1u64 << (bit * D + axis);
            }
        }
    }
    result
}
