use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Curve {
    Line(Line),
}

pub struct Line {
    pub origin: Coordinate<D>,
    pub direction: Direction<D>,
}
