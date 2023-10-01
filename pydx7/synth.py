import numpy as np
from numba import njit
from .dx7tools import get_modmatrix, get_outmatrix
from .dx7tools import render_envelopes
from einops import rearrange

def upsample(signal, factor):

    n = signal.shape[0]
    x = np.linspace(0,n-1,n)
    xvals = np.linspace(0,n-1,n*factor)
    if (len(signal.shape) == 2):
      interpolated = np.zeros((n*factor,signal.shape[1]))
      for i in range(signal.shape[1]):
        interpolated[:,i] = np.interp(xvals,x,signal[:,i])
    else:
      interpolated = np.interp(xvals,x,signal)
    return interpolated


@njit
def dx7_numba_render(fr : np.array, modmatrix : np.array, outmatrix : np.array,
                     pitch : np.array , ol : np.array, sr : int, scale : float = 2*np.pi):
    """
    6-operator FM Renderer with numba
    """    
    n_op = len(fr)
    out = np.zeros_like(pitch)
    phases = np.zeros(n_op) # The free runnin phase
    tstep = 1/sr
    modphases = np.zeros(n_op) # The instantly modulated phase (we just generate an instant value of mod phase.)
    
    for s in range(out.shape[0]):

        # render current phases for each oscillator.
        for mod_op in range(n_op-1,-1,-1):
            phases[mod_op] += tstep * 2 * np.pi * pitch[s] * fr[mod_op]
            if(phases[mod_op] > 2 * np.pi):
                phases[mod_op] -= 2*np.pi
        
        # Copy free running phase array to instantly modulate 
        modphases = phases.copy()
        # Apply modulation of current modulator to target carriers            
        for mod_op in range(n_op-1,-1,-1):

            for carr_op in range(n_op):

                # Select modulator ops
                if(modmatrix[carr_op,mod_op]):
                    #print("Carrier OP{} modulated by OP{}".format(carr_op+1,mod_op+1))
                    # Render sine modulator-to-carrier output, apply output level.
                    mod_ol = ol[s,mod_op]
                    mod_output_to_carrier = np.sin(modphases[mod_op]) * mod_ol * scale
                    # Modulate phase of carrier
                    modphases[carr_op] += mod_output_to_carrier
                    
        out[s] = np.sum(outmatrix * ol[s,:] * np.sin(modphases))

    return out


"""
Simple class to handle a note
"""
class midi_note():
  def __init__(self,n:int=0,v:int=0,ton:int=0,toff:int=0,silence:int=0):
    self.n = n
    self.v = v
    self.ton = ton
    self.toff = toff
    self.silence = silence


class dx7_synth():
  def __init__(self,specs,sr:int=44100,block_size:int=64):
    self.specs = specs
    self.modmatrix = get_modmatrix(specs['algorithm'])
    self.outmatrix = get_outmatrix(specs['algorithm'])
    self.fr = np.array(specs['fr'])
    self.scale = 2*np.pi
    self.sr = sr
    self.block_size = block_size
  
  def render_from_osc_envelopes(self,f0: np.array,ol: np.array):
    """
    Renders from a sequence of output levels and f0
    Args:
      f0: Fudamental frequency vector of size seq_len
      ol: Oscillator output levels format [seq_len,n_osc]
      block_size : int 
      sr: sample rate in Hz.
    """
    ol_up = upsample(ol,self.block_size)
    f0_up = upsample(f0,self.block_size)
    print(f0_up.shape)
    print(ol_up.shape)
    render = dx7_numba_render(self.fr,self.modmatrix,self.outmatrix,
                    f0_up,ol_up,self.sr,self.scale)
    return render / (4*sum(self.outmatrix))

  def render_from_midi_sequence(self,midi_sequence):
    """ 
    Renders audio from a sequence of midi notes
    Args:
      midi_sequence: List of midi_note objects
    """
    envelopes = np.empty((6,0))
    note_contour = np.empty(0)
    
    # Iterate through sequence and render envelopes
    for entry in midi_sequence:
      # Render oscillator envelopes
      if(entry.silence == 0):
        env,qenv = render_envelopes(self.specs,entry.v,entry.ton,entry.toff)
        envelopes = np.append(envelopes,env,axis=1)
        note_contour = np.append(note_contour,np.ones(entry.ton+entry.toff)*entry.n)
      else:
        # Append silence to envelopes
        envelopes = np.append(envelopes,np.zeros((6,entry.silence)),axis=1)
        # Append silence on the pitch contour
        note_contour = np.append(note_contour,np.zeros(entry.silence))
    
    f0 = 440*2**((note_contour-69)/12)
    envelopes = rearrange(envelopes,"oscillators frames -> frames oscillators")
    audio = self.render_from_osc_envelopes(f0,envelopes)

    return audio
