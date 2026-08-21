#[cfg(test)]
mod test;

pub trait Slot: Copy {
    fn at(slot: usize) -> Option<Self>;
    fn slot(self) -> usize;
}

macro_rules! slot {
    ($($slot: ty), +) => {
        $(impl Slot for $slot {
            fn at(slot: usize) -> Option<Self> {
                (slot <= Self::MAX as usize).then_some(slot as Self)
            }
            fn slot(self) -> usize {
                self as usize
            }
        })+
    }
}

slot!(u8, u16, u32, u64, usize);
