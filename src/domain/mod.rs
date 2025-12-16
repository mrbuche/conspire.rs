use crate::{
    math::{TensorRank1Vec2D, TensorRank2Vec2D},
    mechanics::{Coordinates, Forces},
};

pub type NodalCoordinates = Coordinates<1>;
pub type NodalCoordinatesHistory = TensorRank1Vec2D<3, 1>;
pub type NodalForcesSolid = Forces;
pub type NodalReferenceCoordinates = Coordinates<0>;
pub type NodalStiffnessesSolid = TensorRank2Vec2D<3, 1, 1>;
pub type NodalVelocities = Coordinates<1>;
pub type NodalVelocitiesHistory = TensorRank1Vec2D<3, 1>;
