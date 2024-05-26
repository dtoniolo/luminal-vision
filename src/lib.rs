#![feature(generic_const_exprs)]

use luminal::{
    nn::convolution::Conv2D,
    prelude::{Graph, InitModule},
};

/// Most of the convolution operations in a ResNet are a sequence of
/// 1. a 3x3 convolution with no padding, stride or dilation.
/// 1. Batch normalization.
/// 1. rectifier as an activation.
///
/// This block preserves both the spatial dimentions and the number of channels.
///
/// This type doesn't add any functionality, it was defined only for convenience.
///
/// **Note*+ batch norm is currently not present because it's not implemented in luminal.
pub struct Conv<const CHAN: usize>
where
    Conv2D<CHAN, CHAN, 3, 3, 1, 1, 0, 0, { CHAN * 3 * 3 }>: Sized,
{
    conv: Conv2D<CHAN, CHAN, 3, 3, 1, 1, 0, 0, { CHAN * 3 * 3 }>,
}

impl<const CHAN: usize> InitModule for Conv<CHAN>
where
    Conv2D<CHAN, CHAN, 3, 3, 1, 1, 0, 0, { CHAN * 3 * 3 }>: Sized,
{
    fn initialize(cx: &mut Graph) -> Self {
        Self {
            conv: Conv2D::<CHAN, CHAN, 3, 3, 1, 1, 0, 0, { CHAN * 3 * 3 }>::initialize(cx),
        }
    }
}
