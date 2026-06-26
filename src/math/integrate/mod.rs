#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
mod test;

mod dae;
mod error;
mod ode;

pub use error::IntegrationError;

pub use dae::{
    ExplicitDaeFirstOrderMinimize, ExplicitDaeFirstOrderRoot, ExplicitDaeSecondOrderMinimize,
    ExplicitDaeZerothOrderRoot, ImplicitDaeFirstOrderMinimize, ImplicitDaeFirstOrderRoot,
    ImplicitDaeSecondOrderMinimize, ImplicitDaeZerothOrderRoot,
    explicit::variable_step::{
        explicit::{
            ExplicitDaeVariableStepExplicit, ExplicitDaeVariableStepExplicitFirstOrderMinimize,
            ExplicitDaeVariableStepExplicitFirstOrderRoot,
            ExplicitDaeVariableStepExplicitSecondOrderMinimize,
            ExplicitDaeVariableStepExplicitZerothOrderRoot, ExplicitDaeVariableStepFirstSameAsLast,
        },
        implicit::{
            ImplicitDaeVariableStepExplicit, ImplicitDaeVariableStepExplicitFirstOrderMinimize,
            ImplicitDaeVariableStepExplicitFirstOrderRoot,
            ImplicitDaeVariableStepExplicitSecondOrderMinimize,
            ImplicitDaeVariableStepExplicitZerothOrderRoot,
        },
    },
};
pub use ode::{
    FixedStep, OdeIntegrator, VariableStep,
    explicit::{
        Explicit,
        fixed_step::{
            FixedStepExplicit, bogacki_shampine::BogackiShampine as BogackiShampineFixedStep,
            dormand_prince::DormandPrince as DormandPrinceFixedStep, euler::Euler, heun::Heun,
            midpoint::Midpoint, ralston::Ralston, verner_8::Verner8 as Verner8FixedStep,
            verner_9::Verner9 as Verner9FixedStep,
        },
        variable_step::{
            VariableStepExplicit,
            VariableStepExplicitFirstSameAsLast,
            bogacki_shampine::BogackiShampine,
            dormand_prince::DormandPrince,
            // heun_euler::HeunEuler,
            // midpoint_euler::MidpointEuler,
            // ralston_euler::RalstonEuler,
            verner_8::Verner8,
            verner_9::Verner9,
        },
    },
    implicit::{
        ImplicitFirstOrder, ImplicitZerothOrder, backward_euler::BackwardEuler,
        midpoint::Midpoint as ImplicitMidpoint, trapezoidal::Trapezoidal,
    },
};

/// Alias for [`Euler`].
pub type Ode1 = Euler;

/// Alias for [`BackwardEuler`].
pub type Ode1be = BackwardEuler;

// /// Alias for [`HeunEuler`].
// pub type Ode12 = HeunEuler;

/// Alias for [`Heun`].
pub type Ode2 = Heun;

/// Alias for [`BogackiShampine`].
pub type Ode23 = BogackiShampine;

/// Alias for [`BogackiShampineFixedStep`].
pub type Ode3 = BogackiShampineFixedStep;

/// Alias for [`DormandPrince`].
pub type Ode45 = DormandPrince;

/// Alias for [`DormandPrinceFixedStep`].
pub type Ode5 = DormandPrinceFixedStep;

/// Alias for [`Verner8`].
pub type Ode78 = Verner8;

/// Alias for [`Verner8FixedStep`].
pub type Ode8 = Verner8FixedStep;

/// Alias for [`Verner9`].
pub type Ode89 = Verner9;

/// Alias for [`Verner9FixedStep`].
pub type Ode9 = Verner9FixedStep;
