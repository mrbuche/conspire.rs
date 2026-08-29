use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Surface {
    Plane(Plane),
    Cylinder(Cylinder),
    Sphere(Sphere),
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
