pub mod point;
pub mod solid;
#[cfg(test)]
mod test;

use crate::domain::{
    block::{add_node_neighbors, element::Elements},
    qmm::Discretization,
};
use point::Point;
use std::fmt::{self, Debug, Formatter};

pub struct Block<C> {
    constitutive_model: C,
    points: Vec<Point>,
}

impl<C> Debug for Block<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Block {{ {} quadrature points }}", self.points.len())
    }
}

impl<C> From<(C, Discretization)> for Block<C> {
    fn from((constitutive_model, discretization): (C, Discretization)) -> Self {
        Self {
            constitutive_model,
            points: discretization.into_points(),
        }
    }
}

impl<C> Elements for Block<C> {
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(self.points.iter().map(Point::neighbors), neighbors)
    }
}
