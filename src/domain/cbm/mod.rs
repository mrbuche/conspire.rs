//! Continuum bond methods.

pub mod block;
#[cfg(test)]
mod dynamics;

pub mod thermal {
    pub mod capacity {
        pub use crate::domain::thermal::capacity::{
            FixedLumpedHeatCapacities, HeatCapacityMatrix, InertialHeatCapacity,
            InverseHeatCapacity, LumpedHeatCapacityElements, NodalLumpedHeatCapacities,
            NodalTemperatureRates,
        };
    }
    pub mod conduction {
        pub use crate::domain::thermal::conduction::ThermalConductionElements;
    }
    pub mod transient {
        pub use crate::domain::thermal::transient::{
            ThermalConductionImplicitTransient, ThermalConductionTransient,
        };
    }
    pub mod time_scale {
        pub use crate::domain::thermal::time_scale::ThermalTimeScaleElements;
    }
}

pub mod mass {
    pub use crate::domain::solid::mass::{
        FixedLumpedMasses, InertialMass, InverseMass, LumpedMassElements, MassMatrix,
        NodalLumpedMasses,
    };
}

pub use crate::domain::{
    ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
    NodalCoordinatesHistory, NodalReferenceCoordinates, NodalVelocities, NodalVelocitiesHistory,
    Root,
    block::element::Elements,
    solid::{
        NodalDampingsSolid, NodalForcesSolid, NodalStiffnessesSolid, SolidElements,
        dynamics::{ElasticExplicitDynamics, ElasticImplicitDynamics},
        elastic::ElasticElements,
        elastic_hyperviscous::ElasticHyperviscousElements,
        elastic_viscoplastic::{ElasticViscoplasticBCs, ElasticViscoplasticElements},
        hyperelastic::HyperelasticElements,
        hyperelastic_viscoplastic::HyperelasticViscoplasticElements,
        hyperviscoelastic::HyperviscoelasticElements,
        viscoelastic::ViscoelasticElements,
    },
};
pub use block::node::Weighting;
