use super::{Solution, motion::Motion};
use crate::{
    domain::{
        Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
        NodalCoordinatesHistory, NodalVelocities, NodalVelocitiesHistory, Stepping,
        boundary::{ConstrainAt, LoadAt, LoadStiffnessAt, Solid},
        solid::{
            NodalForcesSolid, NodalStiffnessesSolid,
            elastic::ElasticElements,
            mass::{InertialMass, InverseMass, MassChoice},
        },
    },
    math::{
        Quantity,
        integrate::{ImplicitDynamics, IntegrationError},
        optimize::RootFinding,
    },
    units::Time,
};

/// The motion of an elastic model with implicit time integration.
pub trait ElasticImplicitDynamics
where
    Self: Sized,
{
    /// Integrates the motion of the model with an implicit dynamics integrator.
    ///
    /// The masses are those of `mass` ([`Lumped`](crate::domain::solid::mass::Lumped) or
    /// [`Consistent`](crate::domain::solid::mass::Consistent)) that the blocks of the model make,
    /// and are added to the sparse tangent.
    /// Fixed degrees of freedom are held like the explicit integration does, and linear
    /// constraints are not supported.
    #[expect(clippy::too_many_arguments)]
    fn integrate<C>(
        &self,
        mass: C,
        integrator: &impl ImplicitDynamics<
            NodalCoordinates<3>,
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        solver: impl RootFinding<NodalForcesSolid<3>, NodalStiffnessesSolid<3>, NodalCoordinates<3>>,
        time: &[Quantity<Time>],
        initial_state: (NodalCoordinates<3>, NodalVelocities<3>),
        applied_loads: &(
             impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>
             + LoadStiffnessAt<NodalCoordinates<3>, NodalStiffnessesSolid<3>>
         ),
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: MassChoice<Self>;
}

impl<B> ElasticImplicitDynamics for Model<B, 3>
where
    B: ElasticElements<3>,
{
    fn integrate<C>(
        &self,
        mass: C,
        integrator: &impl ImplicitDynamics<
            NodalCoordinates<3>,
            NodalForcesSolid<3>,
            NodalStiffnessesSolid<3>,
            NodalCoordinatesHistory<3>,
            NodalVelocitiesHistory<3>,
            NodalAccelerationsHistory<3>,
        >,
        solver: impl RootFinding<NodalForcesSolid<3>, NodalStiffnessesSolid<3>, NodalCoordinates<3>>,
        time: &[Quantity<Time>],
        (mut initial_coordinates, initial_velocities): (NodalCoordinates<3>, NodalVelocities<3>),
        applied_loads: &(
             impl LoadAt<NodalCoordinates<3>, NodalForcesSolid<3>>
             + LoadStiffnessAt<NodalCoordinates<3>, NodalStiffnessesSolid<3>>
         ),
        conditions: &impl ConstrainAt<Solid<3>>,
    ) -> Result<Solution, IntegrationError>
    where
        C: MassChoice<Self>,
    {
        let initial_time = *time
            .first()
            .ok_or(IntegrationError::LengthTimeLessThanTwo)?;
        let masses = mass.masses(self);
        let (inverse, initial_velocities, motion) = Motion::held(
            &masses,
            conditions,
            initial_time,
            &mut initial_coordinates,
            initial_velocities,
            Stepping::Implicit,
        )?;
        let sparse = self.sparse_solver(&motion.constraint_at(initial_time), 3, false);
        integrator.integrate(
            |time, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>| {
                let mut unbalanced_forces = self
                    .nodal_forces(coordinates)
                    .map_err(|error| error.to_string())?;
                applied_loads.subtract_at(time, coordinates, &mut unbalanced_forces)?;
                Ok(inverse.nodal_accelerations(&unbalanced_forces))
            },
            |time,
             coordinates: &NodalCoordinates<3>,
             _: &NodalVelocities<3>,
             accelerations: &NodalAccelerations<3>| {
                let mut residual = InertialMass::inertial_forces(&masses, accelerations)
                    + self
                        .nodal_forces(coordinates)
                        .map_err(|error| error.to_string())?;
                applied_loads.subtract_at(time, coordinates, &mut residual)?;
                Ok(residual)
            },
            |time, coordinates: &NodalCoordinates<3>, _: &NodalVelocities<3>, dt, beta, _| {
                let mut tangent = self
                    .nodal_stiffnesses(coordinates)
                    .map_err(|error| error.to_string())?;
                masses.add_to_stiffnesses(1.0 / (beta * dt.value().powi(2)), &mut tangent);
                applied_loads.add_stiffness_at(time, coordinates, &mut tangent)?;
                Ok(tangent)
            },
            solver,
            |time| motion.constraint_at(time),
            Some(sparse),
            time,
            initial_coordinates,
            initial_velocities,
        )
    }
}
