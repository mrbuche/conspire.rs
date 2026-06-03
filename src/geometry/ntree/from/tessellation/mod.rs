use crate::{
    geometry::{
        Coordinates,
        mesh::Tessellation,
        ntree::{
            Octree,
            balance::Balancing,
            node::{Kind, Node},
            pair::Pairing,
        },
    },
    math::{Scalar, TensorVec},
};
use std::{array::from_fn, f64::consts::FRAC_PI_3};

impl<T, U> Octree<T, U> {
    fn from_sdf(tessellation: Tessellation, scale: Scalar) -> Self {
        let sdf = tessellation.shape_diameter_function(FRAC_PI_3, 3, 8);
        todo!()
    }
}
