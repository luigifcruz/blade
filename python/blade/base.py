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