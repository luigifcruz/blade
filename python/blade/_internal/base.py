from blade._const_impl import *

#
# Constants 
#

class Constant:
    def __init__(self, value):
        self._value = value
    
    def __repr__(self):
        return f'Constant(value={self._value})'

    @property
    def value(self):
        return self._value
    
# Telescope
ata     = Constant('ata')
vla     = Constant('vla')
meerkat = Constant('meerkat')
generic = Constant('generic')

# Device
cpu     = Constant('cpu')
cuda    = Constant('cuda')
unified = Constant('unified')

# Data Types
f16 = Constant('f16')
f32 = Constant('f32')
f64 = Constant('f64')
i8  = Constant('i8')
i16 = Constant('i16')
i32 = Constant('i32')
i64 = Constant('i64')
u8  = Constant('u8')
u16 = Constant('u16')
u32 = Constant('u32')
u64 = Constant('u64')

# Complex Data Types
cf16 = Constant('cf16')
cf32 = Constant('cf32')
cf64 = Constant('cf64')
ci8  = Constant('ci8')
ci16 = Constant('ci16')
ci32 = Constant('ci32')
ci64 = Constant('ci64')
cu8  = Constant('cu8')
cu16 = Constant('cu16')
cu32 = Constant('cu32')
cu64 = Constant('cu64')

# Modules
beamformer   = Constant('beamformer')
bfr5_reader  = Constant('bfr5-reader')
cast         = Constant('cast')
channelizer  = Constant('channelizer')
duplicate    = Constant('duplicate')
detector     = Constant('detector')
gather       = Constant('gather')
guppi_reader = Constant('guppi-reader')
guppi_writer = Constant('guppi-writer')
permutation  = Constant('permutation')
phasor       = Constant('phasor')
polarizer    = Constant('polarizer')
