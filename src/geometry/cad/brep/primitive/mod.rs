//! Recognising a [`Brep`] as an analytic [`Primitive`] when its faces make one
//! exactly, so meshing can take the closed-form `csg` path.

#[cfg(test)]
mod test;

use super::{Brep, D, surface::Surface};
use crate::geometry::{
    Coordinate,
    csg::{Cuboid, Cylinder, Primitive},
};
use std::array::from_fn;

const EPSILON: f64 = 1.0e-9;

impl Brep {
    /// This solid as an analytic [`Primitive`] — an axis-aligned box or a capped
    /// cylinder — or `None` for anything the recogniser doesn't reduce.
    pub fn primitive(&self) -> Option<Primitive> {
        self.as_cuboid()
            .map(Primitive::Cuboid)
            .or_else(|| self.as_cylinder().map(Primitive::Cylinder))
    }

    fn as_cuboid(&self) -> Option<Cuboid> {
        if self.faces.len() != 6 {
            return None;
        }
        let mut low = [f64::NAN; D];
        let mut high = [f64::NAN; D];
        for face in &self.faces {
            let Surface::Plane(_) = &face.surface else {
                return None;
            };
            let (axis, sign) = principal_axis(face.normal()?)?;
            // The trimming loop, not the (orientation-only) plane origin, fixes
            // where the face sits; every vertex shares the on-axis coordinate.
            let ring = face.bounds.first()?.vertices(&self.edges).ok()?;
            let &first = ring.first()?;
            let coordinate = self.vertices[first][axis].value();
            if ring
                .iter()
                .any(|&vertex| (self.vertices[vertex][axis].value() - coordinate).abs() > EPSILON)
            {
                return None;
            }
            let slot = if sign > 0.0 { &mut high } else { &mut low };
            slot[axis] = coordinate;
        }
        if low.iter().chain(&high).any(|value| value.is_nan()) {
            return None;
        }
        Cuboid::new(low.into(), high.into()).ok()
    }

    fn as_cylinder(&self) -> Option<Cylinder> {
        if self.faces.len() != 3 {
            return None;
        }
        let mut lateral = None;
        let mut caps = Vec::new();
        for face in &self.faces {
            match &face.surface {
                Surface::Cylinder(cylinder) => match lateral {
                    None => lateral = Some(cylinder),
                    Some(_) => return None,
                },
                Surface::Plane(plane) => caps.push(plane),
            }
        }
        let lateral = lateral?;
        let [base_cap, top_cap] = caps.as_slice() else {
            return None;
        };
        let axis: [f64; D] = from_fn(|k| lateral.axis[k].value());
        let origin: [f64; D] = from_fn(|k| lateral.origin[k].value());

        let mut axial = [0.0; 2];
        for (slot, cap) in axial.iter_mut().zip([base_cap, top_cap]) {
            let normal: [f64; D] = from_fn(|k| cap.normal[k].value());
            if (1.0 - dot(normal, axis).abs()).abs() > EPSILON {
                return None;
            }
            let offset: [f64; D] = from_fn(|k| cap.origin[k].value() - origin[k]);
            *slot = dot(offset, axis);
        }
        let (low, high) = (axial[0].min(axial[1]), axial[0].max(axial[1]));
        if high - low <= EPSILON {
            return None;
        }
        let base: Coordinate<D> = from_fn(|k| origin[k] + low * axis[k]).into();
        Cylinder::new(base, lateral.axis.clone(), lateral.radius, high - low).ok()
    }
}

/// `(axis, sign)` when `normal` is one coordinate axis to within `EPSILON`.
fn principal_axis(normal: [f64; D]) -> Option<(usize, f64)> {
    let mut found = None;
    for (axis, &component) in normal.iter().enumerate() {
        if component.abs() > 1.0 - EPSILON {
            match found {
                None => found = Some((axis, component.signum())),
                Some(_) => return None,
            }
        } else if component.abs() > EPSILON {
            return None;
        }
    }
    found
}

fn dot(a: [f64; D], b: [f64; D]) -> f64 {
    (0..D).map(|k| a[k] * b[k]).sum()
}
