pub mod element;
pub mod solid;

use crate::{
    defeat_message,
    math::{
        Scalar, TensorRank1Vec2D, TestError,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
            SecondOrderOptimization, ZerothOrderRootFinding,
        },
    },
    mechanics::Coordinates,
    vem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::element::{ElementNodalCoordinates, VirtualElement},
    },
};
use std::{
    any::type_name,
    fmt::{self, Debug, Display, Formatter},
};

pub type Connectivity = Vec<Vec<usize>>;

pub struct Block<C, F> {
    constitutive_model: C,
    coordinates: NodalReferenceCoordinates,
    elements: Vec<F>,
    element_faces: Connectivity,
    face_nodes: Connectivity,
}

impl<C, F> Block<C, F> {
    fn constitutive_model(&self) -> &C {
        &self.constitutive_model
    }
    fn coordinates(&self) -> &NodalReferenceCoordinates {
        &self.coordinates
    }
    fn elements(&self) -> &[F] {
        &self.elements
    }
    fn element_faces(&self) -> &Connectivity {
        &self.element_faces
    }
    // fn face_nodes(&self) -> &Connectivity {
    //     &self.face_nodes
    // }
    fn element_coordinates<'a>(
        &self,
        coordinates: &'a NodalCoordinates,
        faces: &'a [usize],
    ) -> Vec<Vec<&'a crate::math::TensorRank1<3, 1>>> {
        faces
            .iter()
            .map(|&face| {
                self.face_nodes[face]
                    .iter()
                    .map(|&node| &coordinates[node])
                    .collect()
            })
            .collect()
    }
    fn element_nodes<'a>(&'a self, faces: &'a [usize]) -> Vec<&'a usize> {
        faces
            .iter()
            .flat_map(|&face| &self.face_nodes[face])
            .collect()
    }
}

impl<C, F> Debug for Block<C, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ElementBlock {{ constitutive_model: {}, elements: [Virtual; {}] }}",
            type_name::<C>()
                .rsplit("::")
                .next()
                .unwrap()
                .split("<")
                .next()
                .unwrap(),
            self.elements().len()
        )
    }
}

pub trait VirtualElementBlock<C, F>
where
    F: VirtualElement,
    Self: From<(C, NodalReferenceCoordinates, Connectivity, Connectivity)>,
{
}

impl<C, F> From<(C, NodalReferenceCoordinates, Connectivity, Connectivity)> for Block<C, F>
where
    F: VirtualElement,
{
    fn from(
        (constitutive_model, coordinates, element_faces, face_nodes): (
            C,
            NodalReferenceCoordinates,
            Connectivity,
            Connectivity,
        ),
    ) -> Self {
        let elements = element_faces
            .iter()
            .map(|faces| {
                <F>::from(
                    faces
                        .iter()
                        .map(|&face| {
                            face_nodes[face]
                                .iter()
                                .map(|&node| coordinates[node].clone())
                                .collect()
                        })
                        .collect(),
                )
            })
            .collect();
        Self {
            constitutive_model,
            coordinates,
            elements,
            element_faces,
            face_nodes,
        }
    }
}

pub enum VirtualElementBlockError {
    Upstream(String, String),
}

impl From<VirtualElementBlockError> for String {
    fn from(error: VirtualElementBlockError) -> Self {
        match error {
            VirtualElementBlockError::Upstream(error, block) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In virtual element block: {block}."
                )
            }
        }
    }
}

impl From<VirtualElementBlockError> for TestError {
    fn from(error: VirtualElementBlockError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl Debug for VirtualElementBlockError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, block) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In block: {block}."
                )
            }
        };
        write!(f, "\n{error}\n\x1b[0;2;31m{}\x1b[0m\n", defeat_message())
    }
}

impl Display for VirtualElementBlockError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Upstream(error, block) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In block: {block}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

pub trait ZerothOrderRoot<C, E, X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl ZerothOrderRootFinding<X>,
    ) -> Result<X, OptimizationError>;
}

pub trait FirstOrderRoot<C, E, F, J, X> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderRootFinding<F, J, X>,
    ) -> Result<X, OptimizationError>;
}

pub trait FirstOrderMinimize<C, E, X> {
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl FirstOrderOptimization<Scalar, X>,
    ) -> Result<X, OptimizationError>;
}

pub trait SecondOrderMinimize<C, E, J, H, X> {
    fn minimize(
        &self,
        equality_constraint: EqualityConstraint,
        solver: impl SecondOrderOptimization<Scalar, J, H, X>,
    ) -> Result<X, OptimizationError>;
}
