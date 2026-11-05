#[cfg(test)]
pub(crate) mod test;

mod explicit;
mod implicit;
mod motion;

#[cfg(feature = "gpu")]
pub use explicit::DeviceExplicitDynamics;
pub use explicit::{
    Checked, Consistent, ElasticExplicitDynamics, Execution, Lumped, MassChoice, Stability,
    Unchecked,
};
pub use implicit::ElasticImplicitDynamics;

use crate::{
    domain::{NodalAccelerationsHistory, NodalCoordinatesHistory, NodalVelocitiesHistory},
    math::integrate::Times,
};

type Solution = (
    Times,
    NodalCoordinatesHistory<3>,
    NodalVelocitiesHistory<3>,
    NodalAccelerationsHistory<3>,
);
