//! Recognising a [`Brep`] as an analytic [`Primitive`] when its faces make one
//! exactly, so meshing can take the closed-form `csg` path.

#[cfg(test)]
mod test;

use super::{
    Brep, D,
    surface::{Plane, Surface},
};
use crate::geometry::{
    Coordinate, Direction,
    csg::{Cone, Cuboid, Cylinder, Primitive, Sphere},
};
use std::{array::from_fn, f64::consts::FRAC_PI_2};

const EPSILON: f64 = 1.0e-9;

impl Brep {
    /// This solid as an analytic [`Primitive`] — an axis-aligned box, a sphere,
    /// a capped cylinder, or a truncated cone — or `None` for anything the
    /// recogniser doesn't reduce.
    pub fn primitive(&self) -> Option<Primitive> {
        self.as_cuboid()
            .map(Primitive::Cuboid)
            .or_else(|| self.as_sphere().map(Primitive::Sphere))
            .or_else(|| self.as_cylinder().map(Primitive::Cylinder))
            .or_else(|| self.as_cone().map(Primitive::Cone))
    }

    fn as_sphere(&self) -> Option<Sphere> {
        let mut surfaces = self.faces.iter().map(|face| match &face.surface {
            Surface::Sphere(sphere) => Some(sphere),
            _ => None,
        });
        let first = surfaces.next()??;
        let center: [f64; D] = from_fn(|k| first.origin[k].value());
        for surface in surfaces {
            let surface = surface?;
            if (surface.radius - first.radius).abs() > EPSILON
                || (0..D).any(|k| (surface.origin[k].value() - center[k]).abs() > EPSILON)
            {
                return None;
            }
        }
        Sphere::new(first.origin.clone(), first.radius).ok()
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
                _ => return None,
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
            *slot = cap_axial(cap, axis, origin)?;
        }
        let (low, high) = (axial[0].min(axial[1]), axial[0].max(axial[1]));
        if high - low <= EPSILON {
            return None;
        }
        let base: Coordinate<D> = from_fn(|k| origin[k] + low * axis[k]).into();
        Cylinder::new(base, lateral.axis.clone(), lateral.radius, high - low).ok()
    }

    fn as_cone(&self) -> Option<Cone> {
        if !(2..=3).contains(&self.faces.len()) {
            return None;
        }
        let mut cone = None;
        let mut caps = Vec::new();
        for face in &self.faces {
            match &face.surface {
                Surface::Cone(surface) => match cone {
                    None => cone = Some(surface),
                    Some(_) => return None,
                },
                Surface::Plane(plane) => caps.push(plane),
                _ => return None,
            }
        }
        let cone = cone?;
        if !(EPSILON..FRAC_PI_2 - EPSILON).contains(&cone.semi_angle) {
            return None;
        }
        let slope = cone.semi_angle.tan();
        let axis: [f64; D] = from_fn(|k| cone.axis[k].value());
        let origin: [f64; D] = from_fn(|k| cone.origin[k].value());
        let radius_at = |h: f64| cone.radius + h * slope;

        let (base, base_axis, base_radius, tip_radius, height) = match caps.as_slice() {
            [first, second] => {
                let a = cap_axial(first, axis, origin)?;
                let b = cap_axial(second, axis, origin)?;
                let (low, high) = (a.min(b), a.max(b));
                let (low_radius, high_radius) = (radius_at(low), radius_at(high));
                if low_radius < -EPSILON || high_radius < -EPSILON {
                    return None;
                }
                let base: [f64; D] = from_fn(|k| origin[k] + low * axis[k]);
                (
                    base,
                    axis,
                    low_radius.max(0.0),
                    high_radius.max(0.0),
                    high - low,
                )
            }
            [only] => {
                let cap = cap_axial(only, axis, origin)?;
                let cap_radius = radius_at(cap);
                if cap_radius <= EPSILON || slope <= EPSILON {
                    return None;
                }
                let apex = -cone.radius / slope;
                let base: [f64; D] = from_fn(|k| origin[k] + cap * axis[k]);
                if apex >= cap {
                    (base, axis, cap_radius, 0.0, apex - cap)
                } else {
                    (base, from_fn(|k| -axis[k]), cap_radius, 0.0, cap - apex)
                }
            }
            _ => return None,
        };
        if height <= EPSILON {
            return None;
        }
        Cone::new(
            Coordinate::from(base),
            Direction::const_from(base_axis),
            base_radius,
            tip_radius,
            height,
        )
        .ok()
    }
}

/// The axial coordinate of a planar cap along `axis` from `origin`, or `None`
/// when the cap is not perpendicular to the axis.
fn cap_axial(cap: &Plane, axis: [f64; D], origin: [f64; D]) -> Option<f64> {
    let normal: [f64; D] = from_fn(|k| cap.normal[k].value());
    if (1.0 - dot(normal, axis).abs()).abs() > EPSILON {
        return None;
    }
    Some(dot(from_fn(|k| cap.origin[k].value() - origin[k]), axis))
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
