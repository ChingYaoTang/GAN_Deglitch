import numpy as np
import pylab as plt

def my_random_homo_2Dsphere():
    check = 1
    
    while check:
        x = np.random.rand(3)*2-1
        if np.dot(x,x)<1:
            check=0
    
    r = np.sqrt(np.dot(x,x))
    r_xy = np.sqrt(np.dot(x[:-1],x[:-1]))
    theta = np.arccos(x[2]/r)
    phi = np.arcsin(x[0]/r_xy)
    
    return theta, phi

def linear_interpolation_pycbc_timeseries(pycbc_timeseries,gps_time):
    delta_t = pycbc_timeseries.delta_t
    start_time = pycbc_timeseries.start_time
    end_time   = pycbc_timeseries.end_time
    
    #gps_time value error control
    if start_time > gps_time:
        return 0
    elif end_time < gps_time:
        return 0
    
    index  = int((gps_time-start_time)/delta_t)
    #weight = float((gps_time-start_time)/delta_t - index)
    #delta  = pycbc_timeseries[index+1] - pycbc_timeseries[index]
    
    return pycbc_timeseries[index]



#import healpy as hp
import numpy as np
import pylab as plt

theta=[]
phi=[]

for i in range(10000):
    t,p = my_random_homo_2Dsphere()
    theta.append(t)
    phi.append(p)

theta = np.array(theta)
phi = np.array(phi)

if 0:
    plt.plot(phi,theta,'b.')
    plt.xlabel('phi')
    plt.ylabel('theta')
    plt.title('uniform distribution on 2D sphere')
    plt.savefig('random_sphere.png')
    plt.show()

    x = np.linspace(0,np.pi,1000)
    n, bins, patches = plt.hist(theta, 50, density=False, facecolor='g', alpha=0.5)
    N = len(theta)
    dtheta = bins[1] - bins[0]
    y = np.sin(x)/2.0*dtheta*N
    plt.plot(x,y,label='sin(x)/2 distribution')
    plt.xlabel('theta')
    plt.ylabel('number of point')
    plt.title('distribution of theta')
    plt.xlim(0,np.pi)
    plt.legend()
    plt.savefig('random_sphere_hist.png')
    plt.show()



import healpy
import pylab
import pycbc.psd
import numpy as np
from pycbc.waveform import get_td_waveform,get_fd_waveform
from pycbc.detector import Detector
from pycbc.types.frequencyseries import FrequencySeries
from pycbc.types.timeseries import TimeSeries



apx = 'SEOBNRv4'
#apx = 'EOBNRv2'
#apx = 'SEOBNRv4_ROM'
#apx = 'IMRPhenomD'
#apx = 'SpinTaylorT4'
#apx = 'SpinTaylorT2Fourior'

data_dir = 'gwdata/'
# NOTE: Inclination runs from 0 to pi, with poles at 0 and pi
#       coa_phase runs from 0 to 2 pi.

flow=10
fhigh=1000
srate=4096

for data_num in range(10000):
    print(data_num)
    source_distance=1
    m1 = np.random.rand(1)[0]*40+20
    m2 = np.random.rand(1)[0]*40+20
    inc,coa = my_random_homo_2Dsphere()
    coa = coa + np.pi/2
    hp, hc = get_td_waveform(approximant=apx,
                             distance=source_distance,
                             mass1=m1,
                             mass2=m2,
                             spin1z=0,
                             spin2z=0,
                             inclination=inc,
                             coa_phase=coa,
                             delta_t=1.0/srate,
                             f_lower=flow)
    print 'm1=',m1,' m2=',m2
    print 'duration=',hp.duration
    duration = 16
    if hp.duration > 6:
        duration = 32
    delta_f = 1.0 / duration
    delta_t = 1.0 / srate
    shift_time = 2
    if hp.duration < 1:
        shift_time = 7
    shift_point = srate*shift_time
    flen = int(2048 / delta_f) + 1
    flow_cut  = int(flow/delta_f)
    fhigh_cut = int(fhigh/delta_f)
    #flow_cut  = 1
    #fhigh_cut = flen-1
    psd  = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, 0)
    psd[0]  = psd[1]
    psd[flen-1] = psd[flen-2]
    
    
    hp_temp = np.zeros([duration*srate])
    hp_temp[shift_point:(len(hp)+shift_point)] = hp[:]
    hp_temp_freq = np.fft.fft(hp_temp,flen)*delta_t
    hp_whiten_freq = hp_temp_freq[:]/np.sqrt(psd[:])
    hp_whiten_freq[:flow_cut]=0
    hp_whiten_freq[fhigh_cut:]=0
    hp_whiten = np.fft.ifft(hp_whiten_freq,flen)*delta_f
    
    if 0:
        pylab.loglog(psd.sample_frequencies,np.sqrt(psd))
        pylab.loglog(psd.sample_frequencies,np.abs(hp_temp_freq))
        pylab.xlim(10,2048)
        #pylab.ylim(1.e-24,1.e-22)
        pylab.show()
    
    
    hc_temp = np.zeros([duration*srate])
    hc_temp[shift_point:(len(hc)+shift_point)] = hc[:]
    hc_temp_freq = np.fft.fft(hc_temp,flen)*delta_t
    hc_whiten_freq = hc_temp_freq[:]/np.sqrt(psd[:])
    hc_whiten_freq[:flow_cut]=0
    hc_whiten_freq[fhigh_cut:]=0
    hc_whiten = np.fft.ifft(hc_whiten_freq,flen)*delta_f
    
    hp_whiten = hp_whiten.real
    hp_whiten = TimeSeries(hp_whiten,delta_t=1.0/srate)
    hp_whiten.start_time = hp.start_time - shift_time
    
    hc_whiten = hc_whiten.real
    hc_whiten = TimeSeries(hc_whiten,delta_t=1.0/srate)
    hc_whiten.start_time = hc.start_time - shift_time
    
    #pylab.plot(hc_whiten.sample_times,hc_whiten)
    #pylab.show()

    
    det_h1 = Detector('H1')
    det_l1 = Detector('L1')
    det_v1 = Detector('V1')
    det_k1 = Detector('K1')

    # Choose a GPS end time, sky location, and polarization phase for the merger
    # NOTE: Right ascension and polarization phase runs from 0 to 2pi
    #       Declination runs from pi/2. to -pi/2 with the poles at pi/2. and -pi/2.
    end_time = np.random.rand(1)[0]*1 + 0.5
    declination, right_ascension = my_random_homo_2Dsphere()
    polarization = np.random.rand(1)[0]*np.pi*2
    
    hp_whiten.start_time += end_time
    hc_whiten.start_time += end_time
    hp.start_time += end_time
    hc.start_time += end_time

    if 1:
        signal_h1 = det_h1.project_wave(hp_whiten, hc_whiten,  right_ascension, declination, polarization)
        signal_l1 = det_l1.project_wave(hp_whiten, hc_whiten,  right_ascension, declination, polarization)
        signal_v1 = det_v1.project_wave(hp_whiten, hc_whiten,  right_ascension, declination, polarization)
        signal_k1 = det_k1.project_wave(hp_whiten, hc_whiten,  right_ascension, declination, polarization)
    
    if 0:
        _signal_h1 = det_h1.project_wave(hp, hc,  right_ascension, declination, polarization)
        _signal_l1 = det_l1.project_wave(hp, hc,  right_ascension, declination, polarization)
        _signal_v1 = det_v1.project_wave(hp, hc,  right_ascension, declination, polarization)
        _signal_k1 = det_k1.project_wave(hp, hc,  right_ascension, declination, polarization)
    
    
    length = 8192
    
    strain = np.zeros([length,5],float)
    
    i=0
    times = np.linspace(0,2,length)
    for t in times:
        strain[i,0] = t
        strain[i,1] = linear_interpolation_pycbc_timeseries(signal_h1,t)
        strain[i,2] = linear_interpolation_pycbc_timeseries(signal_l1,t)
        strain[i,3] = linear_interpolation_pycbc_timeseries(signal_v1,t)
        strain[i,4] = linear_interpolation_pycbc_timeseries(signal_k1,t)
        i=i+1
    
    var = np.sum(strain[:,1:]**2)
    strain[:,1:] = strain[:,1:]/np.sqrt(var)
    
    
    parameter=[]
    parameter.append('approximation')
    parameter.append(apx)
    parameter.append('distance')
    parameter.append(1)
    parameter.append('m1')
    parameter.append(m1)
    parameter.append('m2')
    parameter.append(m2)
    parameter.append('inclination')
    parameter.append(inc)
    parameter.append('coa_phase')
    parameter.append(coa)
    parameter.append('declination')
    parameter.append(declination)
    parameter.append('right_ascension')
    parameter.append(right_ascension)
    parameter.append('polarization')
    parameter.append(polarization)
    parameter = np.array(parameter)
    
    np.savez(data_dir+'waveform_data_ID_'+str(data_num), a=strain, b=parameter)
    
    
    if 0:
        #pylab.plot(_signal_h1.sample_times,_signal_h1/np.max(np.array(_signal_h1)),label='before whiten')

        pylab.plot(strain[:,0],strain[:,1]/np.max(strain[:,1]),label='H1')
        #pylab.plot(strain[:,0],strain[:,2],label='L1')
        #pylab.plot(strain[:,0],strain[:,3],label='V1')
        #pylab.plot(strain[:,0],strain[:,4],label='K1')

        #pylab.xlim(end_time-0.7,end_time+0.3)
        pylab.ylim(-1,1)
        pylab.ylabel('Strain')
        pylab.xlabel('Time (s)')
        pylab.legend()
        pylab.show()
    
    print(end_time)

