pub(crate) mod solid;

use std::fmt::Debug;

pub trait Elements
where
    Self: Debug,
{
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]);
}
