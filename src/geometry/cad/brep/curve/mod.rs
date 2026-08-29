use crate::geometry::{Coordinate, Direction};

const D: usize = 3;

pub enum Curve {
    Line(Line),
    Circle(Circle),
    Ellipse(Ellipse),
    BSpline(BSpline),
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

/// A B-spline (or, when `weights` is set, NURBS) curve. Stored as read; not yet
/// evaluated.
pub struct BSpline {
    pub degree: usize,
    pub control_points: Vec<Coordinate<D>>,
    pub knots: Vec<f64>,
    pub multiplicities: Vec<usize>,
    pub weights: Option<Vec<f64>>,
}
