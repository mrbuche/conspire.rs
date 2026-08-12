//! Elastic cohesive constitutive models.

use crate::math::Current;
use crate::{
    constitutive::{Constitutive, ConstitutiveError, cohesive::Cohesive},
    math::{Quantity, Stress, Tensor, TensorArray, TensorRank2, TensorRank2List},
    mechanics::{Normal, Scalar, Separation, Traction},
};

/// A dyad of the directions a cohesive stiffness is built from.
type Dyad = TensorRank2<3, Current, Current>;

pub type StiffnessCohesive = TensorRank2List<3, Current, Current, 2, Stress>;

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
        if tangential_component > 0.0 {
            Ok(normal * normal_traction
                + (tangential_separation / tangential_component) * tangential_traction)
        } else {
            Ok(normal * normal_traction)
        }
    }
    /// Calculates and returns the normal and tangential tractions.
    fn tractions(
        &self,
        normal_separation: Scalar,
        tangential_separation: Scalar,
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
        // The tangent is a direction, so where there is no tangential separation
        // to take it from, it is a zero separation rather than a zero traction.
        let (tangent, ratio, q_t) = if tangential_component > 0.0 {
            (
                tangential_separation / tangential_component,
                normal_component / tangential_component,
                tangential_traction / tangential_component,
            )
        } else {
            (Separation::zero(), 0.0, k_tt)
        };
        let nn = Dyad::from((&normal, &normal));
        let nu = Dyad::from((&normal, &separation));
        let tt = Dyad::from((&tangent, &tangent));
        let tu = Dyad::from((&tangent, &separation));
        let identity = Dyad::identity();
        let stiffness_u = nn * (k_nn - q_t) + tt * (k_tt - q_t) + &identity * q_t;
        let stiffness_n = nu * (k_nn - q_t)
            + identity * (normal_traction - tangential_traction * ratio)
            - tu * ((k_tt - q_t) * ratio);
        Ok([stiffness_u, stiffness_n].into())
    }
    /// Returns the normal and tangential stiffnesses.
    fn stiffnesses(
        &self,
        normal_separation: Scalar,
        tangential_separation: Scalar,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError>;
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
    fn normal_stiffness(&self) -> Quantity<Stress> {
        self.normal_stiffness.into()
    }
    /// Returns the tangential stiffness.
    fn tangential_stiffness(&self) -> Quantity<Stress> {
        self.tangential_stiffness.into()
    }
}

impl Elastic for LinearElastic {
    fn tractions(
        &self,
        normal_separation: Scalar,
        tangential_separation: Scalar,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError> {
        Ok((
            self.normal_stiffness() * normal_separation,
            self.tangential_stiffness() * tangential_separation,
        ))
    }
    fn stiffnesses(
        &self,
        _normal_separation: Scalar,
        _tangential_separation: Scalar,
    ) -> Result<(Quantity<Stress>, Quantity<Stress>), ConstitutiveError> {
        Ok((self.normal_stiffness(), self.tangential_stiffness()))
    }
}
