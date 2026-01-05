pub mod element;
pub mod solid;

use crate::{
    defeat_message,
    math::{
        Scalar, TestError,
        optimize::{
            EqualityConstraint, FirstOrderOptimization, FirstOrderRootFinding, OptimizationError,
            SecondOrderOptimization, ZerothOrderRootFinding,
        },
    },
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
    elements_nodes: Connectivity,
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
    fn element_coordinates<'a>(
        &self,
        coordinates: &'a NodalCoordinates,
        nodes: &[usize],
    ) -> ElementNodalCoordinates<'a> {
        nodes.iter().map(|&node| &coordinates[node]).collect()
    }
    fn elements_nodes(&self) -> &Connectivity {
        &self.elements_nodes
    }
}

impl<C, F> Debug for Block<C, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Block {{ constitutive model: {}, elements: [Virtual; {}] }}",
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
        (constitutive_model, coordinates, elements_faces, faces_nodes): (
            C,
            NodalReferenceCoordinates,
            Connectivity,
            Connectivity,
        ),
    ) -> Self {
        let (elements, elements_nodes) = elements_faces
            .into_iter()
            .map(|element_faces| {
                let element_coordinates = element_faces
                    .iter()
                    .map(|&face| {
                        faces_nodes[face]
                            .iter()
                            .map(|&node| coordinates[node].clone())
                            .collect()
                    })
                    .collect();
                let mut element_nodes = element_faces
                    .iter()
                    .flat_map(|&face| faces_nodes[face].clone())
                    .collect::<Vec<_>>();
                element_nodes.sort();
                element_nodes.dedup();
                (
                    <F>::from((
                        element_coordinates,
                        element_faces,
                        &element_nodes,
                        &faces_nodes,
                    )),
                    element_nodes,
                )
            })
            .unzip();
        Self {
            constitutive_model,
            coordinates,
            elements,
            elements_nodes,
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
