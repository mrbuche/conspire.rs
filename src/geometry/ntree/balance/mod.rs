pub mod octree;
pub mod quadtree;

use crate::geometry::ntree::pair::Pairing;

#[derive(Clone, Copy)]
pub enum Balancing {
    Strong,
    Weak(usize),
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
