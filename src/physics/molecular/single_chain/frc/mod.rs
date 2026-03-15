use crate::{
    math::Scalar,
    physics::molecular::single_chain::{Ensemble, Inextensible, SingleChain},
};

/// The freely-rotating chain model.
#[derive(Clone, Debug)]
pub struct FreelyRotatingChain {
    /// The link angle $`\theta_b`$.
    pub link_angle: Scalar,
    /// The link length $`\ell_b`$.
    pub link_length: Scalar,
    /// The number of links $`N_b`$.
    pub number_of_links: u8,
    /// The thermodynamic ensemble.
    pub ensemble: Ensemble,
}

impl SingleChain for FreelyRotatingChain {
    fn link_length(&self) -> Scalar {
        self.link_length
    }
    fn number_of_links(&self) -> u8 {
        self.number_of_links
    }
}

impl Inextensible for FreelyRotatingChain {
    /// ```math
    /// \lim_{\eta\to\infty}\gamma(\eta) = \frac{\sin\left(\frac{N_b \theta_b}{2}\right)}{N_b \sin\left(\frac{\theta_b}{2}\right)}
    /// ```
    fn maximum_nondimensional_extension(&self) -> Scalar {
        ((self.number_of_links as Scalar * self.link_angle / 2.0).sin()
            / (self.link_angle / 2.0).sin())
            / (self.number_of_links as Scalar)
    }
}
