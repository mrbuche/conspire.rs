//! Shared, discretization-method-agnostic layer for fem/vem/etc.

pub(crate) mod block;

use crate::{
    geometry::Coordinates,
    math::{Current, Reference, TensorRank1Vec, TensorRank1Vec2D},
    units::{Length, Velocity},
};

/// The coordinates of a mesh, given the length they are measured in.
///
/// A mesh is a shape rather than a body, so its coordinates carry no unit until
/// a model is made of it. This is the one place a length is named, and every
/// unit a model carries follows from it.
pub(crate) fn nodal_coordinates<const D: usize>(
    coordinates: Coordinates<D>,
) -> NodalReferenceCoordinates<D> {
    coordinates
        .into_iter()
        .map(|coordinate| coordinate.with_unit())
        .collect()
}

pub type NodalCoordinates<const D: usize> = TensorRank1Vec<D, Current, Length>;
pub type NodalCoordinatesHistory<const D: usize> = TensorRank1Vec2D<D, Current, Length>;
pub type NodalReferenceCoordinates<const D: usize> = TensorRank1Vec<D, Reference, Length>;
pub type NodalVelocities<const D: usize> = TensorRank1Vec<D, Current, Velocity>;
pub type NodalVelocitiesHistory<const D: usize> = TensorRank1Vec2D<D, Current, Velocity>;
