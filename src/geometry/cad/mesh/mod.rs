//! Wiring [`Brep`] into the shared [`Solid`](crate::geometry::solid::Solid)
//! meshing driver.

#[cfg(test)]
mod test;

use super::brep::{Brep, oracle::BrepOracle};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Class, Mesh},
        solid::Solid,
    },
    math::Scalar,
};
use std::array::from_fn;

const D: usize = 3;

impl Solid for Brep {
    type Oracle = BrepOracle;

    fn bounding_box(&self) -> Result<(Coordinate<D>, Coordinate<D>), &'static str> {
        if self.vertices.is_empty() {
            return Err("brep has no vertices");
        }
        let mut low = [Scalar::INFINITY; D];
        let mut high = [Scalar::NEG_INFINITY; D];
        for vertex in &self.vertices {
            for (axis, (lo, hi)) in low.iter_mut().zip(high.iter_mut()).enumerate() {
                *lo = lo.min(vertex[axis].value());
                *hi = hi.max(vertex[axis].value());
            }
        }
        let low: Coordinate<D> = from_fn(|axis| low[axis]).into();
        let high: Coordinate<D> = from_fn(|axis| high[axis]).into();
        Ok((low, high))
    }

    fn oracle(&self) -> Result<Self::Oracle, &'static str> {
        Brep::oracle(self)
    }

    fn classify(&self, mesh: &Mesh<D>) -> Result<Vec<Class>, &'static str> {
        Brep::classify(self, mesh)
    }
}
