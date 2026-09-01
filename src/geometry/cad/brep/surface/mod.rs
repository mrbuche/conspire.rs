use super::curve::Curve;
use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Surface {
    Plane(Plane),
    Cylinder(Cylinder),
    Sphere(Sphere),
    Cone(Cone),
    Torus(Torus),
    BSpline(BSplineSurface),
    Revolution(Revolution),
}

pub struct Plane {
    pub origin: Coordinate<D>,
    pub normal: Direction<D>,
    pub reference_direction: Direction<D>,
}

pub struct Cylinder {
    pub origin: Coordinate<D>,
    pub axis: Direction<D>,
    pub reference_direction: Direction<D>,
    pub radius: f64,
}

pub struct Sphere {
    pub origin: Coordinate<D>,
    pub axis: Direction<D>,
    pub reference_direction: Direction<D>,
    pub radius: f64,
}

pub struct Cone {
    /// A point on the axis where the cone's radius equals `radius`.
    pub origin: Coordinate<D>,
    pub axis: Direction<D>,
    pub reference_direction: Direction<D>,
    pub radius: f64,
    /// Half the apex angle, in radians; the radius grows by `tan(semi_angle)`
    /// per unit along `axis`.
    pub semi_angle: f64,
}

pub struct Torus {
    pub origin: Coordinate<D>,
    pub axis: Direction<D>,
    pub reference_direction: Direction<D>,
    /// Centre of the tube circle to the torus axis.
    pub major_radius: f64,
    /// Radius of the tube.
    pub minor_radius: f64,
}

/// A B-spline (or, when `weights` is set, NURBS) surface. The control net is
/// `control_points[u][v]`. Stored as read; not yet evaluated.
pub struct BSplineSurface {
    pub u_degree: usize,
    pub v_degree: usize,
    pub control_points: Vec<Vec<Coordinate<D>>>,
    pub u_knots: Vec<f64>,
    pub v_knots: Vec<f64>,
    pub u_multiplicities: Vec<usize>,
    pub v_multiplicities: Vec<usize>,
    pub weights: Option<Vec<Vec<f64>>>,
}

/// A surface swept by revolving `curve` about the line through `origin` along
/// `axis`. Stored as read; not yet evaluated.
pub struct Revolution {
    pub curve: Curve,
    pub origin: Coordinate<D>,
    pub axis: Direction<D>,
}
