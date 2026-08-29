use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Surface {
    Plane(Plane),
    Cylinder(Cylinder),
    Sphere(Sphere),
    Cone(Cone),
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
