//! Owned array iteration and (de)serialization.

use super::*;

/// Owned iterator over an array type.
#[doc(hidden)]
pub struct ArrayIter<T, const N: usize> {
    pub(crate) array: MaybeUninit<[T; { N }]>,
    pub(crate) pos:   usize,
}

impl<T, const N: usize> Iterator for ArrayIter<T, { N }> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos == N {
            None
        } else {
            let pos = self.pos;
            self.pos += 1;
            let arrayp: *mut MaybeUninit<T> = unsafe { mem::transmute(&mut self.array) };
            Some(unsafe { arrayp.add(pos).replace(MaybeUninit::uninit()).assume_init() })
        }
    }
}

#[cfg(feature = "serde")]
#[doc(hidden)]
pub struct ArrayVisitor<A> {
    marker: PhantomData<A>,
}

#[cfg(feature = "serde")]
impl<A> ArrayVisitor<A> {
    pub(crate) const fn new() -> Self {
        ArrayVisitor {
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "serde")]
impl<'de, T, const N: usize> Visitor<'de> for ArrayVisitor<[T; N]>
where
    T: Deserialize<'de>,
{
    type Value = [T; N];

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        if N == 0 {
            write!(formatter, "an empty array")
        } else {
            write!(formatter, "an array of length {}", N)
        }
    }

    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut to = MaybeUninit::<[T; N]>::uninit();
        let top: *mut T = unsafe { mem::transmute(&mut to) };
        for i in 0..N {
            if let Some(element) = seq.next_element()? {
                unsafe {
                    top.add(i).write(element);
                }
            } else {
                return Err(A::Error::invalid_length(i, &self));
            }
        }
        unsafe { Ok(to.assume_init()) }
    }
}
