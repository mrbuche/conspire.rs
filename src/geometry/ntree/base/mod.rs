use crate::geometry::ntree::{Orthotree, OrthotreeError, node::base::Split};
use std::{array::from_fn, ops::AddAssign};

impl<const D: usize, const N: usize, T, U> Orthotree<D, N, T, U>
where
    T: AddAssign + Copy + Split,
    U: From<usize> + Into<usize>,
{
    pub fn subdivide(&mut self, index: U) -> Result<(), OrthotreeError> {
        let indices = from_fn(|n| (self.nodes.len() + n).into());
        let nodes = self[index].try_subdivide(indices)?;
        self.nodes.extend(nodes);
        Ok(())
    }
}
