pub(crate) mod dielectric;

use crate::{
    math::{QuantitySparseVec2D, QuantityVector},
    units::{Capacitance, Charge, ElectricPotential},
};

pub type NodalPotentials = QuantityVector<ElectricPotential>;
pub type NodalForcesElectrical = QuantityVector<Charge>;
pub type NodalStiffnessesElectrical = QuantitySparseVec2D<Capacitance>;
