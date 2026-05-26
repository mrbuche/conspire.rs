use crate::geometry::ntree::{
    Orthotree, balance::Balancing, error::OrthotreeError, node::split::Split, pair::Pairing,
};
use std::{array::from_fn, ops::Add};

const D: usize = 2;
const L: usize = 2;
const M: usize = 4;
const N: usize = 4;

impl<T, U> Orthotree<D, L, M, N, T, U>
where
    T: Add<Output = T> + Copy + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
{
    pub fn equilibrate(
        &mut self,
        balancing: Balancing,
        pairing: Pairing,
    ) -> Result<(), OrthotreeError> {
        let mut balanced = false;
        let mut paired = false;
        while !balanced || !paired {
            balanced = self.balance(balancing);
            paired = self.pair(pairing)?;
        }
        Ok(())
    }
    pub fn balance(&mut self, balancing: Balancing) -> bool {
        todo!()
    }
}
