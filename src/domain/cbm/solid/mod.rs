#[cfg(test)]
mod dynamics;

pub mod mass {
    pub use crate::domain::solid::mass::{
        FixedLumpedMasses, InertialMass, InverseMass, LumpedMassElements, MassMatrix,
        NodalLumpedMasses,
    };
}
