use crate::{constitutive::diffusion::surface::Linear, units::Power};

/// Newton's law of cooling, where the heat flux is proportional to the difference between the
/// temperature and the temperature of the surroundings.
pub type Newton = Linear<Power>;
