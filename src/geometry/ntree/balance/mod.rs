pub(super) mod octree;
pub(super) mod quadtree;

use crate::geometry::ntree::pair::Pairing;

/// Constraint on the level difference between neighboring nodes.
#[derive(Clone, Copy, Debug)]
pub enum Balancing {
    /// Level difference of at most `n` between nodes sharing a face, an edge
    /// or a vertex.
    Strong(usize),
    /// Level difference of at most `n` between nodes sharing a face only.
    Weak(usize),
    /// No constraint on the level difference.
    None,
}

pub trait Balance {
    fn equilibrate(&mut self, balancing: Balancing, pairing: Pairing) -> Result<(), &'static str> {
        let mut balanced = false;
        let mut paired = false;
        while !balanced || !paired {
            balanced = self.balance(balancing);
            paired = self.pair_up(pairing)?;
        }
        Ok(())
    }
    fn balance(&mut self, balancing: Balancing) -> bool;
    fn pair_up(&mut self, pairing: Pairing) -> Result<bool, &'static str>;
}
