#[cfg(test)]
mod test;

use std::num::{NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize};

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

macro_rules! nonzero {
    ($($slot: ty), +) => {
        $(impl Slot for $slot {
            fn at(slot: usize) -> Option<Self> {
                slot.checked_add(1)
                    .and_then(|slot| slot.try_into().ok())
                    .and_then(Self::new)
            }
            fn slot(self) -> usize {
                self.get() as usize - 1
            }
        })+
    }
}

nonzero!(NonZeroU16, NonZeroU32, NonZeroU64, NonZeroUsize);
