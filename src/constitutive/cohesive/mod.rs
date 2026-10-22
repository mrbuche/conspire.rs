//! Cohesive constitutive models.

#[cfg(test)]
pub(crate) mod test;

pub mod damage;
pub mod elastic;

use crate::{
    constitutive::{Constitutive, ConstitutiveError},
    math::{Current, Quantity, Tensor, TensorArray, TensorRank2, TensorTuple},
    mechanics::{Normal, Separation, Traction},
    units::{Length, Stress, StressPerLength},
};

type Dyad = TensorRank2<3, Current, Current>;
type DyadSeparation = TensorRank2<3, Current, Current, Length>;

pub(crate) type StiffnessCohesive = TensorTuple<
    TensorRank2<3, Current, Current, StressPerLength>,
    TensorRank2<3, Current, Current, Stress>,
>;

/// The normal and tangential stiffnesses, and the two that couple them.
pub type Stiffnesses = (
    Quantity<StressPerLength>,
    Quantity<StressPerLength>,
    Quantity<StressPerLength>,
    Quantity<StressPerLength>,
);

/// The normal, coupling, and tangential stiffnesses of a symmetric tangent.
pub type SymmetricStiffnesses = (
    Quantity<StressPerLength>,
    Quantity<StressPerLength>,
    Quantity<StressPerLength>,
);

pub(crate) type Tractions = (Quantity<Stress>, Quantity<Stress>);

/// Required methods for cohesive constitutive models.
pub trait Cohesive
where
    Self: Constitutive,
{
}

fn components(
    separation: &Separation,
    normal: &Normal,
) -> (Quantity<Length>, Separation, Quantity<Length>) {
    let normal_component = separation * normal;
    let tangential_separation = separation - normal * normal_component;
    let tangential_component = tangential_separation.norm();
    (
        normal_component,
        tangential_separation,
        tangential_component,
    )
}

pub(crate) fn traction(
    separation: Separation,
    normal: Normal,
    tractions: impl FnOnce(Quantity<Length>, Quantity<Length>) -> Result<Tractions, ConstitutiveError>,
) -> Result<Traction, ConstitutiveError> {
    let (normal_component, tangential_separation, tangential_component) =
        components(&separation, &normal);
    let (normal_traction, tangential_traction) = tractions(normal_component, tangential_component)?;
    if !tangential_component.is_zero() {
        Ok(normal * normal_traction
            + (tangential_separation / tangential_component) * tangential_traction)
    } else {
        Ok(normal * normal_traction)
    }
}

pub(crate) fn stiffness(
    separation: Separation,
    normal: Normal,
    evaluate: impl FnOnce(
        Quantity<Length>,
        Quantity<Length>,
    ) -> Result<(Tractions, Stiffnesses), ConstitutiveError>,
) -> Result<StiffnessCohesive, ConstitutiveError> {
    let (normal_component, tangential_separation, tangential_component) =
        components(&separation, &normal);
    let ((normal_traction, tangential_traction), (k_nn, k_nt, k_tn, k_tt)) =
        evaluate(normal_component, tangential_component)?;
    let (tangent, q_t, k_nt, k_tn) = if !tangential_component.is_zero() {
        (
            tangential_separation / tangential_component,
            tangential_traction / tangential_component,
            k_nt,
            k_tn,
        )
    } else {
        (Normal::zero(), k_tt, Quantity::zero(), Quantity::zero())
    };
    let nn = Dyad::from((&normal, &normal));
    let nt = Dyad::from((&normal, &tangent));
    let tn = Dyad::from((&tangent, &normal));
    let tt = Dyad::from((&tangent, &tangent));
    let nu = DyadSeparation::from((&normal, &separation));
    let tu = DyadSeparation::from((&tangent, &separation));
    let identity = Dyad::identity();
    let stiffness_u =
        nn * (k_nn - q_t) + &nt * k_nt + tn * k_tn + &tt * (k_tt - q_t) + &identity * q_t;
    let stiffness_n = nu * (k_nn - q_t)
        + tu * k_tn
        + identity * (normal_traction - q_t * normal_component)
        + tt * ((q_t - k_tt) * normal_component)
        - nt * (k_nt * normal_component);
    Ok(TensorTuple(stiffness_u, stiffness_n))
}
