pub trait Morton {
    fn morton(&self) -> u64;
}

impl<const D: usize, T: Copy + Into<u64>> Morton for [T; D] {
    fn morton(&self) -> u64 {
        let bits = 64 / D;
        let mut result = 0u64;
        for bit in 0..bits {
            for (axis, &coord) in self.iter().enumerate() {
                if (coord.into() >> bit) & 1 == 1 {
                    result |= 1u64 << (bit * D + axis);
                }
            }
        }
        result
    }
}
