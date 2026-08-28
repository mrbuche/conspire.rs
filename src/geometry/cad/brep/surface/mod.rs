use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Surface {
    Plane(Plane),
    Cylinder(Cylinder),
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
