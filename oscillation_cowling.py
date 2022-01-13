#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 10:07:12 2022

@author: Tianqi Zhao
"""
from scipy.integrate import ode
from scipy import interpolate
from scipy.constants import c,G,e
from scipy.misc import derivative
from numpy import pi
import numpy as np
from astropy.constants import M_sun
import scipy.optimize as opt
from joblib import Parallel, delayed
from multiprocessing import cpu_count


#1.EOS block
dlnx_cs2=1e-10
class EOS_interpolation(object):
    def __init__(self,baryon_density_s,eos_array,s_k=[0,2],eos_array_adiabatic=[],eos_array_Ylep=[]): #defalt s=0,k=3 equal quadratic 1d intepolation
        self.s,self.k=s_k
        self.eos_array=eos_array
        n_array,energy_array,pressure_array=self.eos_array
        self.eosPressure_frombaryon = interpolate.UnivariateSpline(n_array,pressure_array, k=self.k,s=self.s)
        self.eosDensity  = interpolate.UnivariateSpline(pressure_array,energy_array, k=self.k,s=self.s)
        self.eosBaryonDensity = interpolate.UnivariateSpline(pressure_array,n_array, k=self.k,s=self.s)
        if(len(eos_array_adiabatic)==4):
            self.eos_array_adiabatic=eos_array_adiabatic
            self.eosCs2_adiabatic_int = interpolate.UnivariateSpline(self.eos_array_adiabatic[2],self.eos_array_adiabatic[3], k=max(self.k-1,1),s=self.s)
            self.has_cs2_adiabatic=True
        else:
            self.has_cs2_adiabatic=False
        if(len(eos_array_Ylep)==4):
            self.eos_array_Ylep=eos_array_Ylep
            self.eosYlep_int = interpolate.UnivariateSpline(self.eos_array_Ylep[2],self.eos_array_Ylep[3], k=max(self.k-1,1),s=self.s)
            self.has_Ylep=True
        else:
            self.has_Ylep=False
        self.chempo_surface=(pressure_array[0]+energy_array[0])/n_array[0]
        self.baryon_density_s=baryon_density_s
        self.pressure_s=self.eosPressure_frombaryon(self.baryon_density_s)
        self.density_s=self.eosDensity(self.pressure_s)
        self.unit_mass=c**4/(G**3*self.density_s*1e51*e)**0.5
        self.unit_radius=c**2/(G*self.density_s*1e51*e)**0.5
        self.unit_N=self.unit_radius**3*self.baryon_density_s*1e45
    def __getstate__(self):
        state = self.__dict__.copy()
        for dict_intepolation in ['eosPressure_frombaryon','eosDensity','eosBaryonDensity']:
            del state[dict_intepolation]
        if(self.has_cs2_adiabatic):
            del state['eosCs2_adiabatic_int']
        if(self.has_Ylep):
            del state['eosYlep_int']
        return state
    def __setstate__(self, state):
        self.__dict__.update(state)
        n_array,energy_array,pressure_array=self.eos_array
        self.eosPressure_frombaryon = interpolate.UnivariateSpline(n_array,pressure_array, k=self.k,s=self.s)
        self.eosDensity  = interpolate.UnivariateSpline(pressure_array,energy_array, k=self.k,s=self.s)
        self.eosBaryonDensity = interpolate.UnivariateSpline(pressure_array,n_array, k=self.k,s=self.s)
        if(self.has_cs2_adiabatic):
            self.eosCs2_adiabatic_int = interpolate.UnivariateSpline(self.eos_array_adiabatic[2],self.eos_array_adiabatic[3], k=max(self.k-1,1),s=self.s)
        if(self.has_Ylep):
            self.eosYlep_int = interpolate.UnivariateSpline(self.eos_array_Ylep[2],self.eos_array_Ylep[3], k=max(self.k-1,1),s=self.s)
    def eosChempo(self,pressure):
        return (pressure+self.eosDensity(pressure))/self.eosBaryonDensity(pressure)
    def eosCs2(self,pressure):
        return 1.0/derivative(self.eosDensity,pressure,dx=self.eosDensity(pressure)*dlnx_cs2)
    def eosCs2_adiabatic(self,pressure):
        return np.where(np.logical_and(self.eos_array_adiabatic[2].min()<=pressure,pressure<=self.eos_array_adiabatic[2].max()),self.eosCs2_adiabatic_int(pressure),self.eosCs2(pressure))

class EOS_SLY4(EOS_interpolation):
    def __init__(self,s_k=[0,1]):
        path='./'
        eos_array=np.loadtxt(path+'sly4.txt',skiprows=0)
        EOS_interpolation.__init__(self,0.159,eos_array,s_k=s_k)


#2.ODE solver block
def lsoda_ode(function,Preset_rtol,y0,x0,xf,para,method='lsoda'):
    r = ode(function).set_integrator(method,rtol=Preset_rtol,nsteps=1000)
    r.set_initial_value(y0, x0).set_f_params(para)
    r.integrate(xf)
    i=0
    while(i<5 and not r.successful()):
        r.set_initial_value(r.y, r.t).set_f_params(para)
        r.integrate(xf)
        i+=1
    return r

def lsoda_ode_array(function,Preset_rtol,y0,x0,xf_array,para,method='lsoda'):
    r = ode(function).set_integrator(method,rtol=Preset_rtol,nsteps=1000)
    r.set_initial_value(y0, x0).set_f_params(para)
    y_array=[]
    for xf in xf_array:
        r.integrate(xf)
        r.set_initial_value(r.y, r.t).set_f_params(para)
        i=0
        while(i<5 and not r.successful()):
            r.set_initial_value(r.y, r.t).set_f_params(para)
            r.integrate(xf)
            i+=1
        y_array.append(r.y)
    return np.array(y_array)

#3.TOV Equation block
def f_phi(x, y, eos):
    p=np.exp(-x)
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    baryondensity=eos.eosBaryonDensity(p_dimentionful)/eos.baryon_density_s
    dphidx=p/(eps+p)
    if(y[1]==0):
        den=dphidx/(eps/3.0+p)
        return np.array([0,0.5/pi*den,dphidx,0])
    else:
        r=y[1]**0.5
        r4=y[1]**2
        den=dphidx/(y[0]+4*pi*y[1]*r*p)
        rel=1-2*y[0]/r
        return np.array([4*pi*eps*r4*rel*den,2*y[1]*r*rel*den,dphidx,4*pi*baryondensity*r4*np.sqrt(rel)*den])

def MassRadiusPhi_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos):
    x0 = -np.log(pressure_center/eos.density_s)
    xf = x0-np.log(Preset_Pressure_final)
    xf_array = np.linspace(x0,xf,N)
    y_array = lsoda_ode_array(f_phi,Preset_rtol,[0.,0.,0.,0.],x0,xf_array,eos)
    # runtime warning due to zeros at star center
    M_array=y_array[:,0]*eos.unit_mass/M_sun.value
    r_array=y_array[:,1]**0.5*eos.unit_radius
    beta_array=np.concatenate(([0],y_array[1:,0]/(r_array[1:])*eos.unit_radius))
    Phi_array=(y_array[:,2]-y_array[-1,2])+0.5*np.log(1-2*beta_array[-1])
    a_array=y_array[:,3]*eos.unit_N#/y_array[-1,3]
    return  [y_array,xf_array,M_array,r_array,beta_array,Phi_array,a_array]


#4.Perturbation ODE block
def f_Cowling_lnr(lnr,y,eos_omega_l):
    eos,omega,l=eos_omega_l
    r2=np.exp(2*lnr)
    omega2=omega**2
    beta,p,phi,W,U=y
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    exp_lam=1/(1-2*beta)
    if eos.has_cs2_adiabatic:
        cs2=eos.eosCs2_adiabatic(p_dimentionful)
    else:
        cs2=eos.eosCs2(p_dimentionful)
    cs2_eq=eos.eosCs2(p_dimentionful)
    Delta_c2=(cs2-cs2_eq)/(cs2_eq*cs2) #!!!12/02/2021 note that small interpolation error cause serious problem in the crust where cs2 is small!!!
    exp_nu=np.exp(2*phi)
    exp_lam_sqr=exp_lam**0.5
    beta_over_r2=np.where(r2==0,4*pi*eps/3,beta/np.where(r2==0,np.infty,r2))
    dphidr_over_r=exp_lam*(beta_over_r2+4*pi*p)
    dphidr_time_r=dphidr_over_r*r2
    dpdr_time_r=-(eps+p)*dphidr_time_r
    dbetadr_time_r=(4*pi*eps-beta_over_r2)*r2
    
    part1=W-l*exp_nu*exp_lam_sqr*U
    part2=W-l*exp_nu/exp_lam_sqr*U #the_missing_term=4*np.pi*(eps+p)*exp_nu/omega2
    part3=U-dphidr_over_r/(omega2*exp_lam_sqr)*W
    dUdr_time_r=-(l+1)*part1-exp_lam_sqr*omega2*r2/cs2*part3
    dVdr_time_r=exp_lam_sqr/exp_nu*part2+Delta_c2*dphidr_time_r*part3
    return np.array([dbetadr_time_r,dpdr_time_r,dphidr_time_r,dUdr_time_r,dVdr_time_r])

def f_Cowling_lnp(x, y, eos_omega_l): 
    eos,omega,l=eos_omega_l
    p=np.exp(-x)
    beta,r2,phi,W,U=y
    omega2=omega**2
    p_dimentionful=p*eos.density_s
    eps=eos.eosDensity(p_dimentionful)/eos.density_s
    if eos.has_cs2_adiabatic:
        cs2=eos.eosCs2_adiabatic(p_dimentionful)
    else:
        cs2=eos.eosCs2(p_dimentionful)
    cs2_eq=eos.eosCs2(p_dimentionful)
    Delta_c2=(cs2-cs2_eq)/(cs2_eq*cs2) #!!!12/02/2021 note that small interpolation error cause serious problem in the crust where cs2 is small!!!
    exp_nu=np.exp(2*phi)
    exp_lam=1/(1-2*beta)
    exp_lam_sqr=exp_lam**0.5
    beta_over_r2=np.where(r2==0,4*pi*eps/3,beta/np.where(r2==0,np.infty,r2))
    dphidr_over_r=exp_lam*(beta_over_r2+4*pi*p)
    dpdr_over_r=-(eps+p)*dphidr_over_r
    dbetadr_over_r=4*pi*eps-beta_over_r2
    dr2dr_over_r=2
    drdx_time_r=-p/(dpdr_over_r)
    
    part1=W-l*exp_nu*exp_lam_sqr*U
    part2=W-l*exp_nu/exp_lam_sqr*U #the_missing_term=4*np.pi*(eps+p)*exp_nu/omega2
    part3=U-dphidr_over_r/(omega2*exp_lam_sqr)*W
    dUdr_over_r=-(l+1)*part1/r2-exp_lam_sqr*omega2/cs2*part3
    dVdr_over_r=exp_lam_sqr/exp_nu*part2/r2+Delta_c2*dphidr_over_r*part3
    dydr_over_r=np.array([dbetadr_over_r,dr2dr_over_r,dphidr_over_r,dUdr_over_r,dVdr_over_r])
    return dydr_over_r*drdx_time_r

def MassRadius_WU_profile(pressure_center,Preset_Pressure_final,Preset_Radius_range,Preset_rtol,N,eos,omega_dimentionful=2000,Phi_cen=None,lnr_sur=None,l=2):
    if(Phi_cen==None or lnr_sur==None):
        mr_result=MassRadiusPhi_profile(pressure_center,Preset_Pressure_final,Preset_rtol,N,eos)
        Phi_cen=mr_result[5][0]
        lnr_sur=np.log(mr_result[0][-1,1])/2
    method='lsoda'
    x0 = -np.log(pressure_center/eos.density_s)
    omega=omega_dimentionful*eos.unit_radius/c
    eos_omega_l=[eos,omega,l]
    
    #GR vs Cowling profile plot 01/11/2022
    #Preset_Radius_range = [-15,-2] causing a bump in beta around center
    #Use Preset_Radius_range = [-10,-2] to decrease bump in beta around center
    lnr0=lnr_sur+Preset_Radius_range[0]
    lnrt=lnr_sur+Preset_Radius_range[1]
    lnr_array=np.linspace(lnr0,lnrt,int(N/10)+2)
    eps_cen=eos.eosDensity(pressure_center)/eos.density_s
    beta_cen=4*pi*eps_cen/3*np.exp(2*lnr0)
    
    init=[beta_cen,pressure_center/eos.density_s,Phi_cen,1,np.exp(-2*Phi_cen)/l]
    y_array1=lsoda_ode_array(f_Cowling_lnr,Preset_rtol,init,lnr0,lnr_array,eos_omega_l,method=method)
    xt = -np.log(y_array1[-1,1])
    xf = x0-np.log(Preset_Pressure_final)
    xf_array = np.linspace(xt,xf,N)
    init_t=np.copy(y_array1[-1])
    init_t[1]=np.exp(2*lnrt)
    y_array2=lsoda_ode_array(f_Cowling_lnp,Preset_rtol,init_t,xt,xf_array,eos_omega_l,method=method)
    cowling_BC=y_array2[-1][3]-y_array2[-1][4]*y_array2[-1][1]*omega**2*(1-2*y_array2[-1][0])**0.5/y_array2[-1][0]
    if(np.abs(np.exp(2*y_array2[-1,2])-(1-2*y_array2[-1,0]))>1e-5):
        print('phi0 ERROR at surface. y_surface=',y_array2[-1])
        print(np.exp(2*y_array2[-1,2]),(1-2*y_array2[-1,0]))
    xf_array=np.concatenate((-np.log(y_array1[:,1]),xf_array))
    y_array1[:,1]=np.exp(2*lnr_array)
    y_array=np.concatenate((y_array1,y_array2),axis=0)
    V=-y_array[:,4]*np.exp(2*y_array[:,2])
    eps_plus_p=np.exp(-xf_array)+eos.eosDensity(np.exp(-xf_array)*eos.density_s)/eos.density_s
    X=eps_plus_p*(V*omega**2*np.exp(-y_array[:,2])+(y_array[:,0]/y_array[:,1]+4*np.pi*np.exp(-xf_array))/(1-2*y_array[:,0])**0.5*np.exp(y_array[:,2])*y_array[:,3])
    return  np.concatenate((y_array,V[:,np.newaxis],X[:,np.newaxis]),axis=1),cowling_BC,xf_array

#5.eigenvalue root finding block
def omega_for_root_Cowling(omega,pressure_center,Phi_cen,lnr_sur,Preset_Pressure_final,Preset_Radius_range,Preset_rtol,N,eos,WU_function):
    result=MassRadius_WU_profile(pressure_center,Preset_Pressure_final,Preset_Radius_range,Preset_rtol,N,eos,omega_dimentionful=omega,Phi_cen=Phi_cen,lnr_sur=lnr_sur,l=2)
    return result[1]

def omega_Cowling(pc,Preset_Pressure_final,Preset_Radius_range,Preset_rtol,N,eos,omega_tol,init='f'):
    mrp_function,WU_function=[MassRadiusPhi_profile,MassRadius_WU_profile]
    mrp=mrp_function(pc,Preset_Pressure_final,Preset_rtol,N,eos)
    phi0=mrp[5][0]
    lnr_sur=0.5*np.log(mrp[0][-1,1])
    print(phi0,lnr_sur)
    omega_for_root_args=(pc,phi0,lnr_sur,Preset_Pressure_final,Preset_Radius_range,Preset_rtol,N,eos,WU_function)
    if(len(init)==2):
        print('init=',init)
        result1=opt.root_scalar(omega_for_root_Cowling,args=omega_for_root_args,xtol=omega_tol,x0=init[0],x1=init[1]).root
        print('omega=',result1)
        print('===================')
        if(not ('flag' in locals())):flag=False
        return result1

#6.parallel block
def main_parallel_unsave(Calculation_i,parameter_list,other_args=[],verbose=1):
    num_cores = cpu_count()-1
    Output=Parallel(n_jobs=num_cores,verbose=verbose)(delayed(Calculation_i)(parameter_i,other_args) for parameter_i in parameter_list)
    return np.array(Output)



#tested on 01/13/2020
#initialize an EOS from table
eos=EOS_SLY4()

#set solver parameters
N=10
Preset_Pressure_final,Preset_rtol=1e-8,1e-6
Preset_Radius_range=[-10,-2]
omega_tol=0.1
def Calculation_frequency_Cowling(pc_omega,other_args):
    try:
        pc,omega_init=pc_omega
        eos=other_args
        omega_result=omega_Cowling(pc,Preset_Pressure_final,Preset_Radius_range,Preset_rtol,N,eos,omega_tol,init=[omega_init,1.01*omega_init])
    except RuntimeWarning:
        print('Runtimewarning happens at calculating max mass:')
        print(eos.args)
    return np.array([pc,omega_result])


#calculate for a single pc
pc_trial=85
omega_trial=15000
result=Calculation_frequency_Cowling([pc_trial,omega_trial],eos)
print(result)

#calculate for multiple pc parallely
pc_trial=50*2**np.linspace(0,4,30)
omega_trial=np.linspace(14000,20000,30)
result=main_parallel_unsave(Calculation_frequency_Cowling,zip(pc_trial,omega_trial),other_args=eos,verbose=10)
print(result)

import matplotlib.pyplot as plt
plt.plot(result[:,0],result[:,1])
plt.xlabel('$p_c$ (MeV fm$^{-3}$)')
plt.ylabel('$\omega_f$ (s$^{-1}$)')


