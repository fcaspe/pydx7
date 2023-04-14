import numpy as np
from pydx7.dx7env import scalevelocity,scaleoutlevel,EnvelopeGenerator

def render_env(rate,level,ol:int,sens,velocity:int,frames_on:int,frames_off:int,qenvelopes_ratio:float=1.0):

    output_level = scaleoutlevel(ol)
    output_level = output_level << 5
    output_level += scalevelocity(velocity, sens)
    output_level = max(0, output_level)

    e= EnvelopeGenerator(rate,level,output_level)

    e.keydown(True)
    n_frames = frames_on + frames_off
    gain = np.zeros(n_frames,dtype=float)
    qgain = np.zeros(n_frames,dtype=float)
    for i in range(n_frames):
        out = e.getsample()
        qgain[i] = out*qenvelopes_ratio
        # qgain format is doubling log format, this means:
        # qgain is an exponent that each time that adds an unit value, duplicates the output
        # qgain = out, controls the "15 MSbits" of the gain value.
        # But these 15 bits are controlled with Q28 bits precision.
        # From GAIN calculation we see that minimum gain value for ddx7 is 2^(10) / (1<<24) = 2^(10-24)
        # The envelope output is actually Q4.24 (maximum value should be about 15.999... )
        # But somehow it is clamped to 15 maximum (TODO: Check how is this achieved)
        # This value within the exponent of gain expression yields 2^(10 + 15)
        # Now, 2^25 / 2^24 = 2^1 = 2.
        # gain = 2**(10 + out * ( 1.0/(1<<24) ) )/(1<<24)
        
        a = 2**(10 + qgain[i] * ( 1.0/(1<<24) ) )/(1<<24)
        # simpler expresion (as out max value is 15.00... (in q4.24 fmt), exponent maximum is 1, output max is 2.)
        b = 2**(qgain[i] /(1<<24) - 14 ) # drifts a bit on very small values

        assert(abs(a-b) < 1e-9) # make sure this is as stated

        gain[i] = a

        if i == (frames_on):
            e.keydown(False)
    return [gain,qgain]

def render_envelopes(specs,velocity,frames_on,frames_off,
    qenvelopes_ratio=[1.0,1.0,1.0,1.0,1.0,1.0]):

    envelopes = np.zeros([6,frames_on + frames_off])
    qenvelopes = np.zeros([6,frames_on + frames_off])
    for i in range(6):
        #print("Rendering env OP{}".format(i+1))
        
        out = render_env(specs['eg_rate'][:,i],
                                specs['eg_level'][:,i],
                                specs['ol'][i],
                                specs['sensitivity'][i],
                                velocity,frames_on,frames_off,
                                qenvelopes_ratio[i])
        envelopes[i,:] = out[0]
        qenvelopes[i,:] = out[1]

    return [envelopes,qenvelopes]


# Loads the first patch
'''
TODO: Incorporate:  
    - KB RATE Scaling (from dx7 users manual: "The EG for each operator can be set for a
                      long bass decay and a short treble decay - as in an acoustic piano")
    - OP Detune parameter
'''
def load_patch(patch_file,patch_number=0,load_from_sysex=False):

    specs = {}
    bulk_patches = np.fromfile(patch_file, dtype=np.uint8)
    n_patches = int(len(bulk_patches)/128)

    patch_offset = 6 if load_from_sysex==True else 0
    for i in [patch_number]:
        patch = bulk_patches[patch_offset + i*128:patch_offset+ (i+1)*128]
        patch_name = bulk_patches[patch_offset + i*128 + 118: patch_offset + i*128 + 127]
        #patch_name = np.array(patch[118:127],dtype=np.uint8)
        patch_name = patch_name * ( patch_name < 128)
        specs['name'] = patch_name.tostring().decode('ascii')

    patch = unpack_packed_patch(patch)
    algorithm = patch[134]

    fr = np.zeros(6,dtype=float)
    ol = np.zeros(6,dtype=int)
    rates = np.zeros([4,6],dtype=int)
    levels = np.zeros([4,6],dtype=int)
    sensitivity = np.zeros(6,dtype=int)

    # https://homepages.abdn.ac.uk/d.j.benson/pages/dx7/sysex-format.txt
    # Load OP output level, EG rates and levels
    for op in range(6):
        # First in file is OP6
        off = op*21
        is_fixed = patch[off+17]
        if(is_fixed):
            print("[WARNING] tools.py: OP{} in {} is FIXED.".format(6-op,patch_name))

        ol[5-op] = patch[off+16]
        sensitivity[5-op] = patch[off+15]
        
        #compute frequency value    
        f_coarse = patch[off+18]
        f_fine = patch[off+19]
        f_detune = patch[off+20]
        fr[5-op] = compute_freq(f_coarse,f_fine,f_detune) #Detune is ignored for now.
        
        # get rates and levels
        for i in range(4):
            rates[i,5-op] = patch[off+i]
            levels[i,5-op] = patch[off+4+i]

    transpose = (patch[144]-24)
    factor = 2**(transpose/12)
    fr = factor*fr
    
    specs['fr'] = fr
    specs['ol'] = ol
    specs['eg_rate'] = rates
    specs['eg_level'] = levels
    specs['sensitivity'] = sensitivity
    specs['algorithm'] = algorithm #0-31
    specs['outmatrix'] = get_outmatrix(algorithm)
    return specs

""" Stub function that generates only 1 alg

"""
def get_modmatrix(algorithm):
    alg = []
    # alg 1       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])    
    # alg 2       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
    # alg 3       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
    # alg 4       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
    # alg 5       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
    # alg 6       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 7       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
    # alg 8       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
    # alg 9       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 10       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
    # alg 11       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])

    # alg 12       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
    # alg 13       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0] ])

    # alg 14       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
    # alg 15       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])

    # alg 16       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,1,0,1,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])
    # alg 17       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,1,0,1,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 18       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,1,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 19       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 20       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
    
    # alg 21       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,1,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 22       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 23       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 24       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 25       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 26       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
    # alg 27       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,1,0,0,0], [0,0,0,0,0,0], [0,0,0,0,1,1], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
    
    # alg 28       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,1,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,0], [0,0,0,0,0,0] ])
    
    # alg 29       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 30       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,0], [0,0,0,0,0,0] ])

    # alg 31       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,0,0] ])

    # alg 32       OP1            OP2            OP3            OP4            OP5            OP6
    alg.append([ [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0], [0,0,0,0,0,0] ])

    return np.array(alg[algorithm])

def get_outmatrix(algorithm):
    outmatrix = [
        [1,0,1,0,0,0], #1
        [1,0,1,0,0,0], #2
        [1,0,0,1,0,0], #3
        [1,0,0,1,0,0], #4
        [1,0,1,0,1,0], #5
        [1,0,1,0,1,0], #6
        [1,0,1,0,0,0], #7
        [1,0,1,0,0,0], #8
        [1,0,1,0,0,0], #9
        [1,0,0,1,0,0], #10

        [1,0,0,1,0,0], #11
        [1,0,1,0,0,0], #12
        [1,0,1,0,0,0], #13
        [1,0,1,0,0,0], #14
        [1,0,1,0,0,0], #15
        [1,0,0,0,0,0], #16
        [1,0,0,0,0,0], #17
        [1,0,0,0,0,0], #18
        [1,0,0,1,1,0], #19
        [1,1,0,1,0,0], #20

        [1,1,0,1,1,0], #21
        [1,0,1,1,1,0], #22
        [1,1,0,1,1,0], #23
        [1,1,1,1,1,0], #24
        [1,1,1,1,1,0], #25
        [1,1,0,1,0,0], #26
        [1,1,0,1,0,0], #27
        [1,0,1,0,0,1], #28
        [1,1,1,0,1,0], #29
        [1,1,1,0,0,1], #30

        [1,1,1,1,1,0], #31
        [1,1,1,1,1,1], #32
    ]
    return np.array(outmatrix[algorithm])

def compute_freq(coarse,fine,detune):
    # TODO detune parameter is -7 to 7 cents (not implemented)
    f = coarse
    if (f==0): f = 0.5
    f = f + (f/100)*fine
    return f

# Nice unpacking method extracted from https://github.com/bwhitman/learnfm
def unpack_packed_patch(p):
    # Input is a 128 byte thing from compact.bin
    # Output is a 156 byte thing that the synth knows about
    o = [0]*156
    for op in range(6):
        o[op*21:op*21 + 11] = p[op*17:op*17+11]
        leftrightcurves = p[op*17+11]
        o[op * 21 + 11] = leftrightcurves & 3
        o[op * 21 + 12] = (leftrightcurves >> 2) & 3
        detune_rs = p[op * 17 + 12]
        o[op * 21 + 13] = detune_rs & 7
        o[op * 21 + 20] = detune_rs >> 3
        kvs_ams = p[op * 17 + 13]
        o[op * 21 + 14] = kvs_ams & 3
        o[op * 21 + 15] = kvs_ams >> 2
        o[op * 21 + 16] = p[op * 17 + 14]
        fcoarse_mode = p[op * 17 + 15]
        o[op * 21 + 17] = fcoarse_mode & 1
        o[op * 21 + 18] = fcoarse_mode >> 1
        o[op * 21 + 19] = p[op * 17 + 16]

    o[126:126+9] = p[102:102+9]
    oks_fb = p[111]
    o[135] = oks_fb & 7
    o[136] = oks_fb >> 3
    o[137:137+4] = p[112:112+4]
    lpms_lfw_lks = p[116]
    o[141] = lpms_lfw_lks & 1
    o[142] = (lpms_lfw_lks >> 1) & 7
    o[143] = lpms_lfw_lks >> 4
    o[144:144+11] = p[117:117+11]
    o[155] = 0x3f #Seems that OP ON/OFF they are always on. Ignore.

    # Clamp the unpacked patches to a known max. 
    maxes =  [
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc6
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc5
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc4
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc3
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc2
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, # osc1
        3, 3, 7, 3, 7, 99, 1, 31, 99, 14,
        99, 99, 99, 99, 99, 99, 99, 99, # pitch eg rate & level 
        31, 7, 1, 99, 99, 99, 99, 1, 5, 7, 48, # algorithm etc
        126, 126, 126, 126, 126, 126, 126, 126, 126, 126, # name
        127 # operator on/off
    ]
    for i in range(156):
        if(o[i] > maxes[i]): o[i] = maxes[i]
        if(o[i] < 0): o[i] = 0
    return o
