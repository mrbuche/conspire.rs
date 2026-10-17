mod explicit;
mod implicit;

pub use explicit::ElasticExplicitDynamics;
pub use implicit::ElasticImplicitDynamics;

use crate::{
    domain::{
        NodalAccelerationsHistory, NodalCoordinatesHistory, NodalVelocities,
        NodalVelocitiesHistory, factor::fixed_indices, solid::mass::MassMatrix,
    },
    math::{
        integrate::{IntegrationError, Times},
        optimize::EqualityConstraints,
    },
};

type Solution = (
    Times,
    NodalCoordinatesHistory<3>,
    NodalVelocitiesHistory<3>,
    NodalAccelerationsHistory<3>,
);

fn held<M>(
    masses: &M,
    equality_constraint: EqualityConstraints,
    mut velocities: NodalVelocities<3>,
) -> Result<(M::Inverse, NodalVelocities<3>), IntegrationError>
where
    M: MassMatrix<3>,
{
    let fixed = fixed_indices(equality_constraint, "dynamics")?;
    let inverse = masses
        .inverse(&fixed)
        .map_err(|error| IntegrationError::Intermediate(error.to_string()))?;
    fixed
        .iter()
        .for_each(|&index| velocities[index / 3][index % 3] = Default::default());
    Ok((inverse, velocities))
}
