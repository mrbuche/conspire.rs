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

pub trait ViscoelasticFiniteElements<const D: usize>
where
    Self: SolidFiniteElements,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError>;
}

impl<B, const D: usize> ViscoelasticFiniteElements<D> for Model<B, D>
where
    B: ViscoelasticFiniteElements<D>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError> {
        self.blocks
            .nodal_forces(nodal_coordinates, nodal_velocities)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError> {
        self.blocks
            .nodal_stiffnesses(nodal_coordinates, nodal_velocities)
    }
}

impl<B1, B2, const D: usize> ViscoelasticFiniteElements<D> for Blocks<B1, B2>
where
    B1: ViscoelasticFiniteElements<D>,
    B2: ViscoelasticFiniteElements<D>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalForcesSolid<D>, FiniteElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, nodal_velocities)?
            + self.1.nodal_forces(nodal_coordinates, nodal_velocities)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalStiffnessesSolid<D>, FiniteElementModelError> {
        Ok(self
            .0
            .nodal_stiffnesses(nodal_coordinates, nodal_velocities)?
            + self
                .1
                .nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
    }
}

pub trait FirstOrderRoot<const D: usize> {
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeFirstOrderRoot<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalVelocities<D>,
            NodalVelocitiesHistory<D>,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, NodalVelocitiesHistory<D>), IntegrationError>;
}

impl<B, const D: usize> FirstOrderRoot<D> for Model<B, D>
where
    B: ViscoelasticFiniteElements<D>,
{
    fn root(
        &self,
        equality_constraint: EqualityConstraint,
        integrator: impl ImplicitDaeFirstOrderRoot<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalVelocities<D>,
            NodalVelocitiesHistory<D>,
        >,
        time: &[Scalar],
        solver: impl FirstOrderRootFinding<
            NodalForcesSolid<D>,
            NodalStiffnessesSolid<D>,
            NodalCoordinates<D>,
        >,
    ) -> Result<(Times, NodalCoordinatesHistory<D>, NodalVelocitiesHistory<D>), IntegrationError>
    {
        integrator.integrate(
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates<D>,
             nodal_velocities: &NodalVelocities<D>| {
                Ok(self.nodal_forces(nodal_coordinates, nodal_velocities)?)
            },
            |_: Scalar,
             nodal_coordinates: &NodalCoordinates<D>,
             nodal_velocities: &NodalVelocities<D>| {
                Ok(self.nodal_stiffnesses(nodal_coordinates, nodal_velocities)?)
            },
            solver,
            time,
            self.coordinates().clone().into(),
            |_: Scalar| equality_constraint.clone(),
        )
    }
}
