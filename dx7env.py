# Running this inside a for loop may be too slow . . .
# We could use JIT Numba: https://www.infoworld.com/article/3622013/speed-up-your-python-with-numba.html
import numpy as np

LG_N = 6

#See "velocity" section of notes of dexed. Returns velocity delta in microsteps.
def scalevelocity(velocity:int, sensitivity:int):
    velocity_data = [
      0, 70, 86, 97, 106, 114, 121, 126, 132, 138, 142, 148, 152, 156, 160, 163,
      166, 170, 173, 174, 178, 181, 184, 186, 189, 190, 194, 196, 198, 200, 202,
      205, 206, 209, 211, 214, 216, 218, 220, 222, 224, 225, 227, 229, 230, 232,
      233, 235, 237, 238, 240, 241, 242, 243, 244, 246, 246, 248, 249, 250, 251,
      252, 253, 254]
    clamped_vel = max(0, min(127, velocity))
    vel_value = velocity_data[clamped_vel >> 1] - 239
    scaled_vel = ((sensitivity * vel_value + 7) >> 3) << 4
    return scaled_vel

def scaleoutlevel(outlevel:int):
    levellut = [ 0, 5, 9, 13, 17, 20, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 42, 43, 45, 46]
    return 28 + outlevel if outlevel >= 20 else levellut[outlevel]


class EnvelopeGenerator():
    def __init__(self,r:np.array(4,dtype=int),l:np.array(4,dtype=int),ol):
        # 0:Attack,1:Release,2:Sustain,3:Decay
        self.ix = 0
        self.rates = r
        self.levels = l
        self.outlevel = ol
        #self.rate_scaling #TODO
        self.level = 0
        self.targetlevel = 0
        self.inc = 0
        self.rising = False
        self.down = True
        self.advance(0)

    # call env.keydown(false) to release note.
    def keydown(self,d:bool):
        if (self.down != d):
            self.down = d
            if(d):
                self.advance(0)
            else:
                self.advance(3)

    def setparam(self,param:int, value:int):
        if (param < 4):
            self.rates[param] = value
        elif (param < 8):
            self.levels[param - 4] = value


    def advance(self,newix:int):
        self.ix = newix
        if (self.ix < 4):
            #print("[DEBUG] env.advance() - newix: {}".format(self.ix))
            newlevel = self.levels[self.ix];
            #/*Pass from 0-99 to 0-127, then to the level of 64 values?*/
            actuallevel = scaleoutlevel(newlevel) >> 1
            #print("actuallevel {}".format(actuallevel))
            #/*Multiply (in log space . . .) the op outlevel with the level of the EG.*/
            actuallevel = (actuallevel << 6) + self.outlevel - 4256
            #print("outlevel{}".format(self.outlevel))
            #print("actuallevel {}".format(actuallevel))
            #/*Set a minimum possible level.*/
            actuallevel = 16 if (actuallevel < 16 ) else actuallevel
            #print("actuallevel {}".format(actuallevel))
            #// level here is same as Java impl
            self.targetlevel = actuallevel << 16
            self.rising = (self.targetlevel > self.level)

            #// rate

            #/* max val: 99*41 = 4059 - > turn from 12 to 6 bits. */
            qrate = (self.rates[self.ix] * 41) >> 6
            #//printf("orig_rate: %i - qrate: %i ",rates_[ix_], qrate);
            #/*(in log space) multiply rate by the rate scaling.*/
            
            # Rate scaling not applied
            #qrate += self.rate_scaling
            
            qrate = min(qrate, 63)
            self.inc = (4 + (qrate & 3)) << (2 + LG_N + (qrate >> 2))
            #print("[DEBUG] env.advance() - rising {} - targetlevel {} - new inc: {}".format(self.rising,self.targetlevel,self.inc))


# Result is in Q24/doubling log (log2) format. Also, result is subsampled
# for every N samples.
# A couple more things need to happen for this to be used as a gain
# value. First, the # of outputs scaling needs to be applied. Also,
# modulation.
# Then, of course, log to linear.

    def getsample(self):
        if (self.ix < 3 or ((self.ix < 4) and (not self.down))):
            if (self.rising):
                jumptarget = 1716
                if (self.level < (jumptarget << 16)): 
                    self.level = jumptarget << 16
                self.level += (((17 << 24) - self.level) >> 24) * self.inc
                #print(" self.level {}".format(self.level))
                #// TODO: should probably be more accurate when inc is large
                if (self.level >= self.targetlevel):
                    self.level = self.targetlevel
                    self.advance(self.ix + 1)
            else:
                #!rising
                self.level -= self.inc
                if (self.level <= self.targetlevel):
                    self.level = self.targetlevel
                    self.advance(self.ix + 1)
        #printf("Env::getsample() - argument %i - inc %i \n",(((17 << 24) - level_) >> 24),inc_);
        #const double next_gain = pow(2, 10 + level_* (1.0 / (1 << 24)))/(1<<24);
        #printf("Env::getsample() - prev_gain: %lf - next_gain: %lf\n",prev_gain,next_gain);
        #// TODO: this would be a good place to set level to 0 when under threshold
        return self.level