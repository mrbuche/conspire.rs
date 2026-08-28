use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Surface {
    Plane(Plane),
}

pub struct Plane {
    pub origin: Coordinate<D>,
    pub normal: Direction<D>,
    pub reference_direction: Direction<D>,
}
