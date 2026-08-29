use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Curve {
    Line(Line),
    Circle(Circle),
    Ellipse(Ellipse),
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

pub struct Ellipse {
    pub center: Coordinate<D>,
    /// The plane normal.
    pub axis: Direction<D>,
    /// The major-axis direction.
    pub reference_direction: Direction<D>,
    pub major_radius: f64,
    pub minor_radius: f64,
}
