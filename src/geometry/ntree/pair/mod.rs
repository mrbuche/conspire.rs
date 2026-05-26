use crate::geometry::ntree::{
    Orthotree,
    error::OrthotreeError,
    node::{sentinel::Sentinel, split::Split},
};
use std::ops::Add;

#[derive(Clone, Copy)]
pub enum Pairing {
    Generalized,
    Regular,
    None,
}

impl<const D: usize, const L: usize, const M: usize, const N: usize, T, U>
    Orthotree<D, L, M, N, T, U>
where
    T: Add<Output = T> + Copy + PartialEq + Split + Into<usize>,
    U: Copy + From<usize> + Into<usize> + PartialEq + Sentinel,
{
    pub fn pair(&mut self, pairing: Pairing) -> Result<bool, OrthotreeError> {
        match pairing {
            Pairing::Generalized => unimplemented!(),
            Pairing::Regular => {
                let mut index = 0;
                let mut paired = true;
                while index < self.nodes.len() {
                    if let Some(nodes) = self[index.into()].orthants() {
                        let mut any_leaf = false;
                        let mut any_tree = false;
                        let mut leaves = Vec::with_capacity(N);
                        for &node in nodes.iter() {
                            if self[node].is_leaf() {
                                any_leaf = true;
                                leaves.push(node);
                            } else if self[node].is_tree() {
                                any_tree = true;
                            }
                        }
                        if any_tree && any_leaf {
                            for node in leaves {
                                paired = false;
                                self.subdivide(node)?;
                            }
                        }
                    }
                    index += 1;
                }
                Ok(paired)
            }
            Pairing::None => Ok(true),
        }
    }
}
