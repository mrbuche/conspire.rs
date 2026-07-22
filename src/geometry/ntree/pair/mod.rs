mod general;
mod regular;

use crate::geometry::ntree::{Orthotree, node::split::Split};
use std::ops::Add;

#[derive(Clone, Copy)]
pub enum Pairing {
    Generalized,
    Regular,
    None,
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U, V>
    Orthotree<D, L, M, N, T, U, V>
where
    T: Add<Output = T> + Copy + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize>,
    V: Copy,
{
    pub fn pair(&mut self, pairing: Pairing) -> Result<bool, &'static str> {
        match pairing {
            Pairing::Generalized => self.pair_generalized(),
            Pairing::Regular => self.pair_regular(),
            Pairing::None => Ok(true),
        }
    }
}
