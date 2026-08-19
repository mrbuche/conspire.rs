#[cfg(test)]
mod test;

use crate::{
    geometry::{
        Coordinate, CoordinateList, Direction,
        bbox::BoundingBox,
        primitive::{Cylinder, Solid},
    },
    math::{Quantity, Scalar, Tensor},
    units::Length,
};
use std::array::from_fn;

const D: usize = 3;

impl Cylinder {
    /// Builds a cylinder of the given radius spanning two endpoints.
    pub fn new(ends: [Coordinate<D>; 2], radius: Quantity<Length>) -> Self {
        let [base, tip] = ends;
        let span = &tip - &base;
        let height = span.norm();
        assert!(
            height > Quantity::default(),
            "a cylinder needs distinct endpoints"
        );
        assert!(
            radius > Quantity::default(),
            "a cylinder needs a positive radius"
        );
        Self {
            base,
            axis: span.normalized(),
            height,
            radius,
        }
    }
    pub fn axis(&self) -> &Direction<D> {
        &self.axis
    }
    pub fn base(&self) -> &Coordinate<D> {
        &self.base
    }
    pub fn ends(&self) -> [Coordinate<D>; 2] {
        [self.base.clone(), &self.base + &self.axis * self.height]
    }
    pub fn height(&self) -> Quantity<Length> {
        self.height
    }
    pub fn radius(&self) -> Quantity<Length> {
        self.radius
    }
    /// Resolves a point into the cylinder's frame, as how far along the axis it
    /// sits, how far off the axis it sits, and the direction it lies off along.
    ///
    /// A point on the axis has no radial direction of its own, so one is taken
    /// from the axis' orthonormal basis; every radial direction is equally
    /// close to the surface there, and picking any keeps the caller total.
    fn local(&self, point: &Coordinate<D>) -> (Quantity<Length>, Quantity<Length>, Direction<D>) {
        let offset = point - &self.base;
        let axial = &offset * &self.axis;
        let radial_vector = offset - &self.axis * axial;
        let radial = radial_vector.norm();
        let direction = if radial > Quantity::default() {
            radial_vector.normalized()
        } else {
            self.axis.orthonormal_basis()[1].clone()
        };
        (axial, radial, direction)
    }
}

impl Solid<D> for Cylinder {
    /// The exact distance to the cylinder's surface, caps included.
    ///
    /// Outside, the radial and axial overshoots combine as the legs of a right
    /// triangle, which is what rounds the distance correctly around the rim
    /// where both are positive. Inside, both are negative and the nearer
    /// surface — the larger of the two — is the one that governs.
    fn signed_distance(&self, point: &Coordinate<D>) -> Quantity<Length> {
        let zero = Quantity::default();
        let (axial, radial, _) = self.local(point);
        let radial_excess = radial - self.radius;
        let axial_excess = (-axial).max(axial - self.height);
        let outside = Quantity::new(
            (radial_excess.max(zero).value().powi(2) + axial_excess.max(zero).value().powi(2))
                .sqrt(),
        );
        radial_excess.max(axial_excess).min(zero) + outside
    }
    fn closest_point(&self, point: &Coordinate<D>) -> (Coordinate<D>, Direction<D>) {
        let zero = Quantity::default();
        let (axial, radial, radial_direction) = self.local(point);
        let lateral = self.radius - radial;
        let below = axial;
        let above = self.height - axial;
        if lateral < zero || below < zero || above < zero {
            // Outside, so clamping onto the surface lands on the nearest of the
            // lateral surface, a cap, or the rim joining them.
            let closest = &self.base
                + &self.axis * axial.max(zero).min(self.height)
                + &radial_direction * radial.min(self.radius);
            let offset = point - &closest;
            if offset.norm() > zero {
                return (closest, offset.normalized());
            }
            // The point already sits on the surface, leaving no offset to take a
            // direction from, so the nearest face supplies the normal instead.
        }
        if lateral <= below && lateral <= above {
            (
                &self.base + &self.axis * axial + &radial_direction * self.radius,
                radial_direction,
            )
        } else if below <= above {
            (&self.base + &radial_direction * radial, -&self.axis)
        } else {
            (
                &self.base + &self.axis * self.height + &radial_direction * radial,
                self.axis.clone(),
            )
        }
    }
    /// The tight box around the cylinder.
    ///
    /// The lateral surface reaches furthest along an axis where the cylinder's
    /// own axis leans least, by the sine of the angle between them, so each
    /// side of the box stands off the endpoints by that much of the radius
    /// rather than by the radius itself.
    fn bounding_box(&self) -> BoundingBox<D> {
        let [base, tip] = self.ends();
        let extent: [Quantity<Length>; D] =
            from_fn(|axis| self.radius * (1.0 - self.axis[axis].value().powi(2)).max(0.0).sqrt());
        let minimum = Coordinate::from(from_fn::<Scalar, D, _>(|axis| {
            (base[axis].min(tip[axis]) - extent[axis]).value()
        }));
        let maximum = Coordinate::from(from_fn::<Scalar, D, _>(|axis| {
            (base[axis].max(tip[axis]) + extent[axis]).value()
        }));
        BoundingBox::from(CoordinateList::const_from([minimum, maximum]))
    }
}
