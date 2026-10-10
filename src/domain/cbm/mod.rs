//! Continuum bond methods.

pub mod block;
#[cfg(test)]
mod dynamics;

pub mod thermal {
    pub mod capacity {
        pub use crate::domain::thermal::capacity::{
            FixedLumpedHeatCapacities, HeatCapacityMatrix, InverseHeatCapacity,
            LumpedHeatCapacityElements, NodalLumpedHeatCapacities, NodalTemperatureRates,
        };
    }
    pub mod conduction {
        pub use crate::domain::thermal::conduction::ThermalConductionElements;
    }
    pub mod dynamics {
        pub use crate::domain::thermal::dynamics::ThermalConductionDynamics;
    }
    pub mod time_scale {
        pub use crate::domain::thermal::time_scale::ThermalTimeScaleElements;
    }
}

pub mod mass {
    pub use crate::domain::solid::mass::{
        FixedLumpedMasses, InverseMass, LumpedMassElements, MassMatrix, NodalLumpedMasses,
    };
}

pub use crate::domain::{
    ElementModelError, Model, NodalAccelerations, NodalAccelerationsHistory, NodalCoordinates,
    NodalCoordinatesHistory, NodalReferenceCoordinates, NodalVelocities, NodalVelocitiesHistory,
    Root,
    block::element::Elements,
    solid::{
        NodalDampingsSolid, NodalForcesSolid, NodalStiffnessesSolid, SolidElements,
        dynamics::ElasticDynamics,
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
