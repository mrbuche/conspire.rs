use crate::math::{TensorRank1Vec, TensorRank1Vec2D, TensorRank2Vec2D};

pub type NodalCoordinates = TensorRank1Vec<3, 1>;
pub type NodalCoordinatesHistory = TensorRank1Vec2D<3, 1>;
pub type NodalForcesSolid = TensorRank1Vec<3, 1>;
pub type NodalReferenceCoordinates = TensorRank1Vec<3, 0>;
pub type NodalStiffnessesSolid = TensorRank2Vec2D<3, 1, 1>;
pub type NodalVelocities = TensorRank1Vec<3, 1>;
pub type NodalVelocitiesHistory = TensorRank1Vec2D<3, 1>;
