use crate::geometry::ntree::leaf::Leaf;

impl<const D: usize, T, U> Leaf<D, T, U> {
    pub fn corner(&self) -> &[T; D] {
        &self.corner
    }
    pub fn data(&self) -> &U {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut U {
        &mut self.data
    }
    pub fn length(&self) -> &T {
        &self.length
    }
}

pub trait Split {
    fn split(self) -> Self;
}

impl Split for usize {
    fn split(self) -> Self {
        self / 2
    }
}

impl Split for u16 {
    fn split(self) -> Self {
        self / 2
    }
}

pub(crate) fn morton<const D: usize, T: Copy + Into<u64>>(corner: &[T; D]) -> u64 {
    let bits = 64 / D;
    let mut result = 0u64;
    for bit in 0..bits {
        for (axis, &coord) in corner.iter().enumerate() {
            if (coord.into() >> bit) & 1 == 1 {
                result |= 1u64 << (bit * D + axis);
            }
        }
    }
    result
}
