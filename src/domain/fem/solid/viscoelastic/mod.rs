use crate::{
    fem::{
        Blocks, FiniteElementModel, FiniteElementModelError, Model, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        solid::{NodalForcesSolid, NodalStiffnessesSolid, SolidFiniteElements},
    },
    math::{
        Scalar,
        integrate::{ImplicitDaeFirstOrderRoot, IntegrationError},
        optimize::{EqualityConstraint, FirstOrderRootFinding},
    },
    mechanics::Times,
};

pub trait ViscoelasticFiniteElements
where
    Self: SolidFiniteElements,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalForcesSolid, FiniteElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError>;
}

impl<B> ViscoelasticFiniteElements for Model<B>
where
    B: ViscoelasticFiniteElements,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        self.blocks
            .nodal_forces(nodal_coordinates, nodal_velocities)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        self.blocks
            .nodal_stiffnesses(nodal_coordinates, nodal_velocities)
    }
}

impl<B1, B2> ViscoelasticFiniteElements for Blocks<B1, B2>
where
    B1: ViscoelasticFiniteElements,
    B2: ViscoelasticFiniteElements,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalForcesSolid, FiniteElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, nodal_velocities)?
            + self.1.nodal_forces(nodal_coordinates, nodal_velocities)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
    ) -> Result<NodalStiffnessesSolid, FiniteElementModelError> {
        Ok(self
            .0
            .nodal_stiffnesses(nodal_coordinates, nodal_velocities)?
            + self
                .1
                .nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
    }
}

pub trait FirstOrderRoot {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeFirstOrderRoot<
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalVelocities,
            NodalVelocitiesHistory,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError>;
}

impl<B> FirstOrderRoot for Model<B>
where
    B: ViscoelasticFiniteElements,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeFirstOrderRoot<
            NodalForcesSolid,
            NodalStiffnessesSolid,
            NodalVelocities,
            NodalVelocitiesHistory,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<NodalForcesSolid, NodalStiffnessesSolid, NodalCoordinates>,
    ) -> Result<(Times, NodalCoordinatesHistory, NodalVelocitiesHistory), IntegrationError> {
        integrator.integrate(
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates,
             nodal_velocities: &NodalVelocities| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates,
             nodal_velocities: &NodalVelocities| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            solver,
            time,
            self.coordinates().clone().into(),
            |_: Scalar| equality_constraint.clone(),
        )
    }
}
