use crate::{math::Scalar, physics::molecular::single_chain::SingleChain};

pub enum Ensemble {
    Isometric,
    Isotensional,
}

pub trait Thermodynamics
where
    Self: Isometric + Isotensional + Legendre + SingleChain,
{
    fn ensemble(&self) -> &Ensemble;
    fn nondimensional_helmholtz_free_energy(&self, nondimensional_extension: Scalar) -> Scalar {
        match self.ensemble() {
            Ensemble::Isometric => {
                Isometric::nondimensional_helmholtz_free_energy(self, nondimensional_extension)
            }
            Ensemble::Isotensional => {
                Legendre::nondimensional_helmholtz_free_energy(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_force(&self, nondimensional_extension: Scalar) -> Scalar {
        match self.ensemble() {
            Ensemble::Isometric => Isometric::nondimensional_force(self, nondimensional_extension),
            Ensemble::Isotensional => {
                Legendre::nondimensional_force(self, nondimensional_extension)
            }
        }
    }
    fn nondimensional_gibbs_free_energy(&self, nondimensional_force: Scalar) -> Scalar {
        match self.ensemble() {
            Ensemble::Isometric => {
                Legendre::nondimensional_gibbs_free_energy(self, nondimensional_force)
            }
            Ensemble::Isotensional => {
                Isotensional::nondimensional_gibbs_free_energy(self, nondimensional_force)
            }
        }
    }
    fn nondimensional_extension(&self, nondimensional_force: Scalar) -> Scalar {
        match self.ensemble() {
            Ensemble::Isometric => Legendre::nondimensional_extension(self, nondimensional_force),
            Ensemble::Isotensional => {
                Isotensional::nondimensional_extension(self, nondimensional_force)
            }
        }
    }
}

pub trait Isometric {
    fn nondimensional_helmholtz_free_energy(&self, nondimensional_extension: Scalar) -> Scalar;
    fn nondimensional_force(&self, nondimensional_extension: Scalar) -> Scalar;
}

pub trait Isotensional {
    fn nondimensional_gibbs_free_energy(&self, nondimensional_extension: Scalar) -> Scalar;
    fn nondimensional_extension(&self, nondimensional_force: Scalar) -> Scalar;
}

pub trait Legendre
where
    Self: Isometric + Isotensional + SingleChain,
{
    fn nondimensional_helmholtz_free_energy(&self, nondimensional_extension: Scalar) -> Scalar {
        let nondimensional_force = Legendre::nondimensional_force(self, nondimensional_extension);
        Isotensional::nondimensional_gibbs_free_energy(self, nondimensional_force)
            + self.number_of_links() * nondimensional_force * nondimensional_extension
    }
    fn nondimensional_force(&self, nondimensional_extension: Scalar) -> Scalar {
        todo!("default impl invert isotensional nondimensional extension")
    }
    fn nondimensional_gibbs_free_energy(&self, nondimensional_force: Scalar) -> Scalar {
        let nondimensional_extension =
            Legendre::nondimensional_extension(self, nondimensional_force);
        Isometric::nondimensional_helmholtz_free_energy(self, nondimensional_extension)
            - self.number_of_links() * nondimensional_force * nondimensional_extension
    }
    fn nondimensional_extension(&self, nondimensional_force: Scalar) -> Scalar {
        todo!("default impl invert isometric nondimensional force")
    }
}
