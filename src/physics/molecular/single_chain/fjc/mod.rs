use crate::{
    math::{
        Scalar,
        special::{inverse_langevin, langevin},
    },
    physics::molecular::single_chain::{
        Ensemble, Isometric, Isotensional, Legendre, SingleChain, Thermodynamics,
    },
};

/// The freely-jointed chain model.
pub struct FreelyJointedChain {
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: Scalar,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for FreelyJointedChain {
    fn number_of_links(&self) -> Scalar {
        self.number_of_links
    }
}

impl Thermodynamics for FreelyJointedChain {
    fn ensemble(&self) -> &Ensemble {
        &self.ensemble
    }
}

impl Isometric for FreelyJointedChain {
    fn nondimensional_helmholtz_free_energy(&self, _nondimensional_extension: Scalar) -> Scalar {
        //
        // Need to Error when reach extensibility
        //
        todo!()
    }
    fn nondimensional_force(&self, nondimensional_extension: Scalar) -> Scalar {
        //
        // Need to Error when reach extensibility
        //
        let orders = [0, 1]; // why was this an input argument?
        let n = self.number_of_links() as u128;
        let p: i32 = self.number_of_links() as i32 - 2;
        let m = -nondimensional_extension * 0.5 + 0.5;
        let k = (self.number_of_links() * m).ceil() as u128;
        let sums: Vec<f64> = orders
            .iter()
            .map(|order| {
                (0..=k - 1)
                    .collect::<Vec<u128>>()
                    .iter()
                    .map(|s| {
                        (-1.0_f64).powf(*s as f64)
                            * (((1..=n).product::<u128>()
                                / (1..=*s).product::<u128>()
                                / (1..=n - s).product::<u128>())
                                as f64)
                            * (m - (*s as f64) / self.number_of_links()).powi(p - order)
                    })
                    .sum()
            })
            .collect();
        (1.0 / nondimensional_extension + (0.5 * self.number_of_links() - 1.0) * sums[1] / sums[0])
            / self.number_of_links()
    }
}

impl Isotensional for FreelyJointedChain {
    fn nondimensional_gibbs_free_energy(&self, nondimensional_force: Scalar) -> Scalar {
        (nondimensional_force / nondimensional_force.sinh()).ln()
    }
    fn nondimensional_extension(&self, nondimensional_force: Scalar) -> Scalar {
        langevin(nondimensional_force)
    }
}

impl Legendre for FreelyJointedChain {
    fn nondimensional_force(&self, nondimensional_extension: Scalar) -> Scalar {
        //
        // Need to Error when reach extensibility
        //
        inverse_langevin(nondimensional_extension)
    }
}
