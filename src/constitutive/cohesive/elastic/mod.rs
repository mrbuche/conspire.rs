//! Elastic cohesive constitutive models.

use crate::{
    constitutive::{Constitutive, ConstitutiveError, cohesive::Cohesive},
    math::{Current, Quantity, Tensor, TensorArray, TensorRank2, TensorTuple},
    mechanics::{Normal, Scalar, Separation, Traction},
    units::{Length, Stress, StressPerLength},
};

type Dyad = TensorRank2<3, Current, Current>;
type DyadSeparation = TensorRank2<3, Current, Current, Length>;

pub(crate) type StiffnessCohesive = TensorTuple<
    TensorRank2<3, Current, Current, StressPerLength>,
    TensorRank2<3, Current, Current, Stress>,
>;

/// Required methods for elastic cohesive constitutive models.
pub trait Elastic
where
    Self: Cohesive,
{
    fn traction(
        &self,
        separation: Separation,
        normal: Normal,
    ) -> Result<Traction, ConstitutiveError> {
        let normal_component = &separation * &normal;
        let normal_separation = &normal * normal_component;
        let tangential_separation = separation - normal_separation;
        let tangential_component = tangential_separation.norm();
        let (normal_traction, tangential_traction) =
            self.tractions(normal_component, tangential_component)?;
        if !tangential_component.is_zero() {
            Ok(normal * normal_traction
                + (tangential_separation / tangential_component) * tangential_traction)
        } else {
            Ok(normal * normal_traction)
        }
    }
    /// Calculates and returns the normal and tangential tractions.
    fn tractions(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError>;
    fn stiffness(
        &self,
        separation: Separation,
        normal: Normal,
    ) -> Result<StiffnessCohesive, ConstitutiveError> {
        let normal_component = &separation * &normal;
        let normal_separation = &normal * normal_component;
        let tangential_separation = &separation - normal_separation;
        let tangential_component = tangential_separation.norm();
        let (normal_traction, tangential_traction) =
            self.tractions(normal_component, tangential_component)?;
        let (k_nn, k_tt) = self.stiffnesses(normal_component, tangential_component)?;
        let (tangent, ratio, q_t) = if !tangential_component.is_zero() {
            (
                tangential_separation / tangential_component,
                normal_component / tangential_component,
                tangential_traction / tangential_component,
            )
        } else {
            (Normal::zero(), 0.0.into(), k_tt)
        };
        let nn = Dyad::from((&normal, &normal));
        let nu = DyadSeparation::from((&normal, &separation));
        let tt = Dyad::from((&tangent, &tangent));
        let tu = DyadSeparation::from((&tangent, &separation));
        let identity = Dyad::identity();
        let stiffness_u = nn * (k_nn - q_t) + tt * (k_tt - q_t) + &identity * q_t;
        let stiffness_n = nu * (k_nn - q_t)
            + identity * (normal_traction - tangential_traction * ratio)
            - tu * ((k_tt - q_t) * ratio);
        Ok(TensorTuple(stiffness_u, stiffness_n))
    }
    /// Calculates and returns the normal and tangential stiffnesses.
    fn stiffnesses(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> Result<(Quantity<StressPerLength>, Quantity<StressPerLength>), ConstitutiveError>;
}

/// The linear elastic cohesive constitutive model.
#[derive(Clone, Debug)]
pub struct LinearElastic {
    /// The normal stiffness $`k_n`$.
    pub normal_stiffness: Scalar,
    /// The tangential stiffness $`k_t`$.
    pub tangential_stiffness: Scalar,
}

impl Constitutive for LinearElastic {}

impl Cohesive for LinearElastic {}

impl LinearElastic {
    /// Returns the normal stiffness.
    fn normal_stiffness(&self) -> Quantity<StressPerLength> {
        self.normal_stiffness.into()
    }
    /// Returns the tangential stiffness.
    fn tangential_stiffness(&self) -> Quantity<StressPerLength> {
        self.tangential_stiffness.into()
    }
}

impl Elastic for LinearElastic {
    fn tractions(
        &self,
        normal_separation: Quantity<Length>,
        tangential_separation: Quantity<Length>,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError> {
        Ok((
            self.normal_stiffness() * normal_separation,
            self.tangential_stiffness() * tangential_separation,
        ))
    }
    fn stiffnesses(
        &self,
        _normal_separation: Quantity<Length>,
        _tangential_separation: Quantity<Length>,
    ) -> Result<(Quantity<StressPerLength>, Quantity<StressPerLength>), ConstitutiveError> {
        Ok((self.normal_stiffness(), self.tangential_stiffness()))
    }
}
