//! Wiring [`Brep`] into the shared [`Solid`](crate::geometry::solid::Solid)
//! meshing driver.

#[cfg(test)]
mod test;

use super::brep::{Brep, oracle::BrepOracle, surface::Surface};
use crate::{
    geometry::{
        Coordinate,
        mesh::{Class, Mesh},
        solid::{Solid, classify_by_flood_fill},
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
        // A curved face can bulge past its edge vertices, so union the oracle's
        // per-face bounds in.
        let (mut low, mut high) = Brep::oracle(self)
            .map(|oracle| {
                let (low, high) = oracle.bounds();
                (
                    from_fn::<Scalar, D, _>(|k| low[k].value()),
                    from_fn::<Scalar, D, _>(|k| high[k].value()),
                )
            })
            .unwrap_or(([Scalar::INFINITY; D], [Scalar::NEG_INFINITY; D]));
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
        if self
            .faces
            .iter()
            .all(|face| matches!(face.surface, Surface::Plane(_)))
        {
            Brep::classify(self, mesh)
        } else {
            classify_by_flood_fill(&Brep::oracle(self)?, mesh)
        }
    }
}
