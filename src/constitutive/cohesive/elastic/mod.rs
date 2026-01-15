//! Elastic cohesive constitutive models.

use crate::{
    constitutive::{Constitutive, ConstitutiveError, cohesive::Cohesive},
    math::{Rank2, Tensor, TensorArray, TensorRank1, TensorRank2, TensorRank2List},
    mechanics::{Normal, Scalar, Separation, Stiffness, Traction},
};

pub type Tractions = TensorRank1<2, 8>;
pub type Stiffnesses = TensorRank2<2, 8, 8>;
pub type StiffnessCohesive = TensorRank2List<3, 1, 1, 2>;

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
        let [normal_traction, tangential_traction] = self
            .tractions(normal_component, tangential_component)?
            .into();
        if tangential_component > 0.0 {
            Ok(normal * normal_traction
                + (tangential_separation / tangential_component) * tangential_traction)
        } else {
            Ok(normal * normal_traction)
        }
    }
    fn tractions(
        &self,
        normal_separation: Scalar,
        tangential_separation: Scalar,
    ) -> Result<Tractions, ConstitutiveError>;
    fn stiffness(
        &self,
        separation: Separation,
        normal: Normal,
    ) -> Result<StiffnessCohesive, ConstitutiveError> {
        let normal_component = &separation * &normal;
        let normal_separation = &normal * normal_component;
        let tangential_separation = &separation - normal_separation;
        let tangential_component = tangential_separation.norm();
        let [normal_traction, tangential_traction] = self
            .tractions(normal_component, tangential_component)?
            .into();
        let [[k_nn, k_nt], [k_tn, k_tt]] = self
            .stiffnesses(normal_component, tangential_component)?
            .into();
        let tangent = tangential_separation / tangential_component;
        let ratio = normal_component / tangential_component;
        let nn = Stiffness::from((&normal, &normal));
        let nt = Stiffness::from((&normal, &tangent));
        let tn = nt.transpose();
        let tt = Stiffness::from((&tangent, &tangent));
        let q_t = tangential_traction / tangential_component;
        let identity = Stiffness::identity();
        let stiffness_u = nn * (k_nn - ratio * k_nt - q_t)
            + tn * (k_tn - ratio * (k_tt + q_t))
            + nt * k_nt
            + tt * (k_tt - q_t)
            + &identity * q_t;
        let nu = Stiffness::from((&normal, &separation));
        let tu = Stiffness::from((&tangent, &separation));
        let stiffness_t = nu * (k_nn - ratio * k_nt + q_t)
            + tu * (k_tn + ratio * (q_t - k_tt))
            + identity * (normal_traction + ratio * tangential_traction);
        Ok([stiffness_u, stiffness_t].into())
    }
    fn stiffnesses(
        &self,
        normal_separation: Scalar,
        tangential_separation: Scalar,
    ) -> Result<Stiffnesses, ConstitutiveError>;
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

impl Elastic for LinearElastic {
    fn tractions(
        &self,
        normal_separation: Scalar,
        tangential_separation: Scalar,
    ) -> Result<Tractions, ConstitutiveError> {
        Ok([
            normal_separation * self.normal_stiffness,
            tangential_separation * self.tangential_stiffness,
        ]
        .into())
    }
    fn stiffnesses(
        &self,
        _normal_separation: Scalar,
        _tangential_separation: Scalar,
    ) -> Result<Stiffnesses, ConstitutiveError> {
        Ok([
            [self.normal_stiffness, 0.0],
            [0.0, self.tangential_stiffness],
        ]
        .into())
    }
}
