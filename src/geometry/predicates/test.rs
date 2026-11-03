/// A small deterministic random number generator for the tests of the predicates.
pub struct Lcg(pub u64);

impl Lcg {
    pub fn bits(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    /// A number in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        self.bits() as f64 / (1u64 << 53) as f64
    }
    /// An integer in `[-2^bits, 2^bits)`.
    pub fn integer(&mut self, bits: u32) -> i64 {
        (self.bits() % (1u64 << (bits + 1))) as i64 - (1i64 << bits)
    }
}
