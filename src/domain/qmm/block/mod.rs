pub mod mass;
pub mod point;
pub mod solid;
#[cfg(test)]
mod test;
mod time_scale;

use crate::{
    domain::{block::element::Items, qmm::Discretization},
    geometry::mesh::InnerProducts,
    math::Quantity,
    units::{Density, Volume},
};
use point::Point;
use std::fmt::{self, Debug, Formatter};

pub struct Block<C, R = ()> {
    constitutive_model: C,
    points: Vec<Point>,
    inner_products: InnerProducts<Volume>,
    density: R,
}

impl<C, R> Debug for Block<C, R> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "Block {{ {} quadrature points }}", self.points.len())
    }
}

impl<C> From<(C, Discretization)> for Block<C> {
    fn from((constitutive_model, discretization): (C, Discretization)) -> Self {
        let (points, inner_products) = discretization.into_parts();
        Self {
            constitutive_model,
            points,
            inner_products,
            density: (),
        }
    }
}

impl<C> Block<C> {
    pub fn with_density(self, density: Quantity<Density>) -> Block<C, Quantity<Density>> {
        Block {
            constitutive_model: self.constitutive_model,
            points: self.points,
            inner_products: self.inner_products,
            density,
        }
    }
}

impl<C, R> Items for Block<C, R> {
    fn number_of_items(&self) -> usize {
        self.points.len()
    }
    fn nodes_of(&self, item: usize) -> &[usize] {
        self.points[item].neighbors()
    }
}
