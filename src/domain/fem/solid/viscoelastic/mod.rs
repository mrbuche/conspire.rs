use crate::{
    fem::{
        Blocks, ElementModel, ElementModelError, Elements, Model, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory,
        solid::{NodalForcesSolid, NodalStiffnessesSolid},
    },
    math::{
        Scalar,
        integrate::{ImplicitDaeFirstOrderRoot, IntegrationError},
        optimize::{EqualityConstraint, FirstOrderRootFinding},
    },
    mechanics::Times,
};

pub trait ViscoelasticElements<const D: usize>
where
    Self: Elements,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalForcesSolid<D>, ElementModelError>;
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError>;
}

impl<B, const D: usize> ViscoelasticElements<D> for Model<B, D>
where
    B: ViscoelasticElements<D>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        self.blocks
            .nodal_forces(nodal_coordinates, nodal_velocities)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
        self.blocks
            .nodal_stiffnesses(nodal_coordinates, nodal_velocities)
    }
}

impl<B1, B2, const D: usize> ViscoelasticElements<D> for Blocks<B1, B2>
where
    B1: ViscoelasticElements<D>,
    B2: ViscoelasticElements<D>,
{
    fn nodal_forces(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalForcesSolid<D>, ElementModelError> {
        Ok(self.0.nodal_forces(nodal_coordinates, nodal_velocities)?
            + self.1.nodal_forces(nodal_coordinates, nodal_velocities)?)
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates<D>,
        nodal_velocities: &NodalVelocities<D>,
    ) -> Result<NodalStiffnessesSolid<D>, ElementModelError> {
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
    B: ViscoelasticElements<D>,
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
