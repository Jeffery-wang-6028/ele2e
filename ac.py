#multivariate static arithmetic coding for writing code bins
import numpy as np
import torch
AC__MinLength = 0x01000000
AC__MaxLength = 0xFFFFFFFF


DM__LengthShift = 16
DM__MaxCount    = 1 << DM__LengthShift

class Static_Data_Model(object):
    def __init__(self,cdf):
        self.data_symbols = cdf.shape[0]-1
        self.distribution = cdf
        self.last_symbol=self.data_symbols - 1

    def set_distribution(self,cdf):
        self.distribution = cdf

class Arithmetic_Codec(object):
    def __init__(self,max_code_bytes,user_buffer):
        self.mode = self.buffer_size = 0
        self.new_buffer = self.code_buffer = 0
        self.buffer_size = max_code_bytes
        self.code_buffer = user_buffer
        self.value=0

    def buffer(self):
        return self.code_buffer

    def start_encoder(self):
        if (self.mode != 0):
            raise ValueError("cannot start encoder")
        if (self.buffer_size == 0):
            raise ValueError("no code buffer set")
        self.mode   = 1
        self.base   = 0
        self.length = AC__MaxLength
        self.ac_pointer = 0

    def start_decoder(self):
        if (self.mode != 0):
            raise ValueError("cannot start encoder")
        if (self.buffer_size == 0):
            raise ValueError("no code buffer set")
        self.mode = 2
        self.length = AC__MaxLength
        self.ac_pointer = 3
        self.value=(np.int64(self.code_buffer[0])<< 24)+(np.int64(self.code_buffer[1])<< 16)+ (np.int64(self.code_buffer[2])<<  8)+ (np.int64(self.code_buffer[3]))

    def stop_encoder(self):
        if (self.mode != 1):
            raise ValueError("invalid to stop encoder")
        self.mode = 0

        if (self.length > 2 * AC__MinLength):
            self.base += AC__MinLength
            self.length = AC__MinLength >> 1
        else:
            self.base += AC__MinLength >> 1
            self.length = AC__MinLength >> 9

        if (self.base>AC__MaxLength):
            assert self.base < (AC__MaxLength<<1)
            self.propagate_carry()

        self.renorm_enc_interval()

        code_bytes = self.ac_pointer
        if (code_bytes > self.buffer_size):
            raise ValueError("code buffer overflow")
        return code_bytes

    def stop_decoder(self):
        if (self.mode != 2):
            raise ValueError("invalid to stop encoder")
        code_bytes = self.ac_pointer
        self.mode = 0
        return code_bytes

    def encode(self,data,M):
        if (data == M.last_symbol):
            x = M.distribution[data] * (self.length >> DM__LengthShift)
            self.base += x
            self.length -= x
        else:
            self.length = self.length>>DM__LengthShift
            x = M.distribution[data] * (self.length)
            self.base += x
            self.length  = M.distribution[data+1] * self.length - x

        if (self.base>AC__MaxLength):
            assert self.base<(AC__MaxLength<<1)
            self.base = self.base & AC__MaxLength
            self.propagate_carry()

        if (self.length < AC__MinLength):
            self.renorm_enc_interval()

    def decode(self,M):
        y = self.length
        x = s = 0
        self.length >>= DM__LengthShift
        n = M.data_symbols
        m = n >> 1
        while (m!= s):
            z = self.length * M.distribution[m]
            if (z > self.value):
                n = m
                y = z
            else:
                s = m
                x = z
            m = (s + n) >> 1

        self.value -= x
        self.length = y - x

        if (self.length < AC__MinLength):
            self.renorm_dec_interval()
        return s

    def put_bits(self,data,bits):
        if (self.mode != 1):
            raise ValueError("encoder not initialized")
        if ((bits < 1) or (bits > 20)):
            raise ValueError("invalid number of bits")
        if (data >= (1 << bits)):
            raise ValueError("invalid data")
        init_base = self.base
        self.length >>= bits
        self.base += data * (self.length)
        if (self.base>AC__MaxLength):
            assert self.base < (AC__MaxLength << 1)
            self.base = self.base & AC__MaxLength
            self.propagate_carry()
        if (self.length < AC__MinLength):
            self.renorm_enc_interval()

    def get_bits(self,bits):
        if (self.mode != 2):
            raise ValueError("decoder not initialized")
        if ((bits < 1) or (bits > 20)):
            raise ValueError("invalid number of bits")
        self.length >>= bits
        s = self.value // (self.length)
        self.value -= self.length * s
        if (self.length < AC__MinLength):
            self.renorm_dec_interval()
        return s

    def propagate_carry(self):
        p = self.ac_pointer - 1
        while self.code_buffer[p] == 0xFF:
            self.code_buffer[p] = 0
            p-=1
        self.code_buffer[p]+=1

    def renorm_enc_interval(self):
        while (self.length < AC__MinLength):
            self.code_buffer[self.ac_pointer] = (self.base >> 24)
            self.ac_pointer+=1
            self.base <<= 8
            self.base = self.base & AC__MaxLength
            self.length <<= 8
        assert self.length<AC__MaxLength

    def renorm_dec_interval(self):
        while (self.length< AC__MinLength):
            self.ac_pointer+=1
            self.value = ((self.value << 8) + self.code_buffer[self.ac_pointer])& AC__MaxLength
            self.length <<= 8
        assert self.value<AC__MaxLength

    def flush(self,flush_buffer):
        if self.ac_pointer>(self.buffer_size>>1)+4:
            self.code_buffer[:self.buffer_size>>1].copy(flush_buffer)

def xencodeCU(codec,cofs,cdf):
    cdf = cdf.astype(np.uint32)
    for n in range(cofs.shape[0]):
        coeff_model = Static_Data_Model(cdf[n,:])
        for i in range(cofs.shape[1]):
            codec.encode(cofs[n,i], coeff_model)
    return codec.code_buffer

def xdecodeCU(codec,num_sym,cdf):
    cdf = cdf.astype(np.uint32)
    cof_dec = np.zeros([cdf.shape[0],num_sym], dtype=np.uint32)
    for n in range(cdf.shape[0]):
        coeff_model = Static_Data_Model(cdf[n,:])
        for i in range(num_sym):
            cof_dec[n,i]=codec.decode(coeff_model)
    return cof_dec

