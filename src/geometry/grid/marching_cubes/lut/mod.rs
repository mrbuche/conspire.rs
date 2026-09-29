#[cfg(test)]
mod test;

pub(super) struct Lut<const N: usize> {
    pub(super) l1: usize,
    pub(super) l2: usize,
    pub(super) values: [i8; N],
}

impl<const N: usize> Lut<N> {
    pub(super) fn get1(&self, i0: usize) -> i32 {
        i32::from(self.values[i0])
    }
    pub(super) fn get2(&self, i0: usize, i1: usize) -> i32 {
        i32::from(self.values[i0 * self.l1 + i1])
    }
    pub(super) fn get3(&self, i0: usize, i1: usize, i2: usize) -> i32 {
        i32::from(self.values[i0 * self.l1 * self.l2 + i1 * self.l2 + i2])
    }
}
