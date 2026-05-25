use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    leaf::{
        Leaf,
        base::{Split, morton},
    },
};
use std::{array::from_fn, ops::AddAssign};

impl<const D: usize, const N: usize, T, U> Orthotree<D, N, T, U>
where
    T: AddAssign + Copy + Into<u64> + Split,
    U: Clone,
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
        let child_length = length.split();
        let children: [Leaf<D, T, U>; N] = from_fn(|i| {
            let mut child_corner = corner;
            child_corner
                .iter_mut()
                .enumerate()
                .for_each(|(axis, coord)| {
                    if (i >> axis) & 1 == 1 {
                        *coord += child_length;
                    }
                });
            Leaf {
                corner: child_corner,
                length: child_length,
                data: data.clone(),
            }
        });
        let insert_pos = self
            .leaves
            .partition_point(|l| morton(&l.corner) < morton(&children[0].corner));
        for (offset, child) in children.into_iter().enumerate() {
            self.leaves.insert(insert_pos + offset, child);
        }
        Ok(())
    }
}
