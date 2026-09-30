use crate::{
    domain::{
        Model, NodalReferenceCoordinates,
        meshfree::block::{Block, point::Point},
        nodal_coordinates,
        solid::NodalForcesSolid,
    },
    geometry::{
        Coordinates,
        mesh::{Basis, Mesh},
    },
    math::{Quantity, Scalar},
    mechanics::Traction,
    units::Length,
};

/// The seeds of a basis, by their spacing and how many spacings their support reaches.
#[derive(Clone, Copy, Debug)]
pub struct Support {
    pub spacing: Quantity<Length>,
    pub reach: Scalar,
}

/// Seeds and quadrature points that discretize the solution over a fine mesh.
#[derive(Clone, Debug)]
pub struct Discretization {
    coordinates: NodalReferenceCoordinates<3>,
    basis: Basis,
    points: Vec<Point>,
}

impl Discretization {
    /// Linear-reproducing approximation functions at the seeds of one packing,
    /// with the gradients projected onto the functions of a second, finer packing.
    pub fn new(
        mesh: &Mesh<3>,
        approximation: Support,
        quadrature: Support,
        seed: u64,
    ) -> Result<Self, &'static str> {
        let basis = |support: Support, seed: u64| {
            let seeds = mesh.sample(support.spacing, seed);
            let radius = support.spacing * support.reach;
            mesh.reproducing_basis(&seeds, radius, 1)
                .map(|basis| (seeds, basis))
        };
        let (seeds, approximation) = basis(approximation, seed)?;
        let (_, quadrature) = basis(quadrature, seed + 1)?;
        let weights = mesh.integrals(&quadrature)?;
        let gradients = mesh.projected_gradients(&approximation, &quadrature)?;
        let points = weights
            .into_iter()
            .zip(gradients.values)
            .map(|(weight, entries)| {
                let (neighbors, gradient_vectors) = entries.into_iter().unzip();
                Point::new(weight, neighbors, gradient_vectors)
            })
            .collect();
        let coordinates: Coordinates<3> = seeds
            .iter()
            .map(|&seed| mesh.coordinates()[seed].clone())
            .collect();
        Ok(Self {
            coordinates: nodal_coordinates(coordinates),
            basis: approximation,
            points,
        })
    }
    pub fn coordinates(&self) -> &NodalReferenceCoordinates<3> {
        &self.coordinates
    }
    pub fn traction(
        &self,
        mesh: &Mesh<3>,
        faces: &[Vec<usize>],
        traction: &Traction,
    ) -> Result<NodalForcesSolid<3>, &'static str> {
        Ok(mesh
            .face_integrals(&self.basis, faces)?
            .into_iter()
            .map(|area| traction * area)
            .collect())
    }
    pub fn model<C>(self, constitutive_model: C) -> Model<Block<C>, 3> {
        (
            Block::from((constitutive_model, self)),
            self.coordinates.clone(),
        )
            .into()
    }
    pub(crate) fn into_points(self) -> Vec<Point> {
        self.points
    }
}
