use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Curve {
    Line(Line),
    Circle(Circle),
}

pub struct Line {
    pub origin: Coordinate<D>,
    pub direction: Direction<D>,
}

pub struct Circle {
    pub center: Coordinate<D>,
    pub axis: Direction<D>,
    pub reference_direction: Direction<D>,
    pub radius: f64,
}
