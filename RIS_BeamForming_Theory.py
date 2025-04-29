from pickle import TRUE
import scipy
from scipy.interpolate import RBFInterpolator, InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
import skrf as rf
import numpy as np
import time
import serial

def main():

    rf.stylely()

    f0 = 11.78
    L0 = (scipy.constants.c*1e-6)/f0
    
    W = 180
    L = 230

    dX = 10
    Nx = 18
    
    dY = 10
    Ny = 1
    # bf = BeamForming(-20, 0, 0, 0)
    # angle = [-10, -10, -10]
    # #---------------------------------------------------------------------#
    # # Unit Cell with different Phase Span
    # min_amplitude = 0.3*np.array([1,1,1, 1,1,1])
    # max_phase = 325*np.array([1,1,1, 1,1,1])
    # min_amplitude = [0.1, 0.1, 0.1]
    # max_phase = np.linspace(90, 360, int((360-90)/30)+1)
    max_phase = np.array([325])

    N = len(max_phase)

    min_amplitude   = 0.35*np.ones(N)
    angle           = -20*np.ones(N)
    
    uc = []
    bf = []
    AF = []
    RIS = []
    for i in range(N):
        # Create the Unit Cell
        uc.insert(i, UnitCell(min_amplitude[i], max_phase[i], f0))

        # Create BeamForming class
        bf.insert(i, BeamForming(angle[i], 0, 0, 0))

        # Initialize the Antenna Factor
        AF.insert(i,  AntennaFactor(bf[i]))

        # Create the RIS configuration using given UnitCell for BeamForming
        RIS.insert(i,  RIS_plate(uc[i], bf[i], AF[i], Nx, dX, Ny, dY))

        # Estimate Antenna Factor
        AF[i].get_Balanis_3D_AntennaFactor(RIS[i])

        # Get Side Lobes Level
        AF[i].getSideLobesLevel(RIS[i])

    plt.figure(1)
    plt.subplot(1,2,1)
    for i in range(N):
        plot_label = "MinAmplitude = {0:.2f}".format(uc[i].minAmpl)
        plt.plot(uc[i].V, uc[i].ampl, label=plot_label)
    plt.legend()
    plt.xlabel("Voltage [V]")
    plt.ylabel("Amplitude [1]")
    plt.ylim([0, 1])
    plt.subplot(1,2,2)
    for i in range(N):
        plot_label = "MaxPhase = {0}".format(uc[i].phaseSpan)
        plt.plot(uc[i].V, uc[i].phase, label=plot_label)
    plt.legend()
    plt.xlabel("Voltage [V]")
    plt.ylabel("Phase [deg]")
    plt.ylim([0, 400])

    plt.figure(4)
    for i in range(N):
        plot_label = "MinAmplitude = {0:.2f}, MaxPhase = {1}".format(uc[i].minAmpl, uc[i].phaseSpan)
        plt.plot(AF[i].theta_deg, AF[i].limit_dB, label=plot_label)
        #plt.axhline(AF[i].ml_mag)
        #plt.axhline(AF[i].sll_mag+AF[i].ml_mag)
    #plt.axvline(AF[1].bf.thetaR_deg, color="black", linestyle="--", label="Reflected Beam={0}deg".format(AF[1].bf.thetaR_deg))
    plt.legend()
    plt.xlabel("Azimuth [deg]")
    plt.ylabel("Magnitude [dB]")
    plt.ylim([-60, 0])
    print("")
    for i in range(N):
        print("")
        print("Angle={0}deg, MinAmplitude={1}, MaxPhase={2}".format(angle[i], min_amplitude[i], max_phase[i]))
        print("ML [{:0.2f}, {:0.2f}]".format(AF[i].ml_azim, AF[i].ml_mag))
        print("SL [{:0.2f}, {:0.2f}]".format(AF[i].sll_azim, AF[i].sll_mag))
        print("SLL = {:0.2f}dB".format(AF[i].sll_mag-AF[i].ml_mag))

    plt.show()

    print("")






def rad2deg(rad):
    return rad*180/np.pi

def deg2rad(deg):
    return deg*np.pi/180

class AntennaFactor:
    def __init__(self, bf):
        self.bf = bf
        self.PhiR_limit_deg = []
        self.PhiR_ideal_deg = []
        self.PhiR_limit_rad = []
        self.PhiR_ideal_rad = []
        self.sll_azim       = []
        self.sll_mag        = []
    
    def get_Balanis_3D_AntennaFactor(self, RIS):
        self.theta_deg = np.linspace(-90, 90, 1801)
        # self.phy_deg   = np.linspace(-180, 180, ((180*2)+1))
        self.phy_deg   = np.linspace(0, 0, 1)
        self.theta_rad = deg2rad(self.theta_deg)
        self.phy_rad   = deg2rad(self.phy_deg)

        self.N_theta = len(self.theta_deg)
        self.N_phy   = len(self.phy_deg)

        self.ideal = np.zeros((self.N_theta, self.N_phy), dtype=complex)
        self.limit = np.zeros((self.N_theta, self.N_phy), dtype=complex)

        max_theta = self.theta_deg[np.argmax(self.theta_deg)]

        for i in range(self.N_theta):
            print("Theta={0:.2f}\tTheta_max={1:.2f}".format(self.theta_deg[i], max_theta), end='\r')
            for j in range(self.N_phy):

                for y in range(RIS.Ny):
                    for x in range(RIS.Nx):
                        # A = np.exp(-1*1j*self.PhiI_rad[y][x]);
                        # B = np.exp(-1*1j*RIS.phase_ideal_deg[y][x]);

                        C = 1*RIS.k0*RIS.dX*(x*np.sin(self.theta_rad[i])*np.cos(self.phy_rad[j]) + y*np.sin(self.theta_rad[i])*np.sin(self.phy_rad[j]));

                        self.ideal[i][j] = self.ideal[i][j] +           1*np.exp(-1*1j*(self.PhiR_ideal_rad[y][x]+C))
                        # self.limit[i][j] = self.limit[i][j] +           1*np.exp(-1*1j*(self.PhiR_limit_rad[y][x]+C))
                        self.limit[i][j] = self.limit[i][j] + RIS.ampl[x]*np.exp(-1*1j*(self.PhiR_limit_rad[y][x]+C))

                self.ideal[i][j] = self.ideal[i][j]/(RIS.Nx*RIS.Ny)
                self.limit[i][j] = self.limit[i][j]/(RIS.Nx*RIS.Ny)

        self.limit_dB = 20*np.log10(abs(self.limit))
        self.ideal_dB = 20*np.log10(abs(self.ideal))

    def getSideLobesLevel(self, RIS):
        for j in range(self.N_phy):
            limit = self.limit_dB[:,j]
            peaks, _ = scipy.signal.find_peaks(limit, height=-20)
            peaksX = np.take(self.theta_deg, peaks)
            peaksY = np.take(limit, peaks)

            index = np.argmax(peaksY)
            self.ml_azim = peaksX[index]
            self.ml_mag  = peaksY[index]
            peaksY[index] = -30
            index = np.argmax(peaksY)
            self.sll_azim = peaksX[index]
            self.sll_mag  = peaksY[index]



    def getMainLobeWidth(self):
        index = np.argmax(self.limit_dB)
        self.mainLobe = self.limit_dB[index]
        diff = np.abs(self.limit_dB - self.mainLobe - (-3))
        index1 = np.argmin(diff)
        self.mainLobeLeft = self.theta_deg[index1]
        diff[index1] = 100
        index2 = np.argmin(diff)
        self.mainLobeRight = self.theta_deg[index2]
        self.mainLobeWidth = abs(self.theta_deg[index2] - self.theta_deg[index1])


    def plotAntennaFactor(self):
        plt.plot(self.theta_deg, self.limit_dB, label="AF limited")
        # plt.plot(self.theta_deg, self.ideal_dB, label="AF ideal")
        plt.axvline(self.bf.thetaR_deg, color="black", linestyle="--", label="Reflected Beam={0}deg".format(self.bf.thetaR_deg))
        plt.legend()
        plt.xlabel("Azimuth [deg]")
        plt.ylabel("Magnitude [dB]")
        plt.ylim([-60, 0])

class RIS_plate:
    def __init__(self, uc, bf, AF, Nx, dX, Ny, dY):
        self.uc = uc
        self.bf = bf
        self.AF = AF
        self.F = self.uc.F
        self.Nx = Nx
        self.dX = dX
        self.Ny = Ny
        self.dY = dY
        self.L0 = (scipy.constants.c*1e-6)/self.F
        #self.dX = self.L0/3
        self.k0 = (2*np.pi)/self.L0
        self.MaxPhase_deg = uc.phaseSpan

        self.get_dPhy()
        self.getPhaseProfile()
        self.getVoltageAtUnitCell()

    def get_dPhy(self):
        if(self.bf.thetaI_deg == 0):
            dL = self.dX*np.sin(self.bf.thetaR_rad); 
            self.dPhy_deg = -1*360*dL/self.L0;
            self.dPhy_rad = deg2rad(self.dPhy_deg)
        else:
            R = self.bf.nR*np.sin(self.bf.thetaI_rad);
            I = self.bf.nI*np.sin(self.bf.thetaR_rad);
            self.dPhy_rad = self.dX*self.k0*(R-I);
            self.dPhy_deg = rad2deg(self.dPhy_rad)

    def getPhaseProfile(self):

        self.phase_limit_deg = np.zeros((self.Ny, self.Nx))
        self.phase_ideal_deg = np.zeros((self.Ny, self.Nx))

        self.PhiR_rad = np.zeros((self.Ny, self.Nx))
        self.PhiI_rad = np.zeros((self.Ny, self.Nx))

        self.PhiR_deg = np.zeros((self.Ny, self.Nx))
        self.PhiI_deg = np.zeros((self.Ny, self.Nx))

        self.AF.PhiR_limit_deg = np.zeros((self.Ny, self.Nx))
        self.AF.PhiR_ideal_deg = np.zeros((self.Ny, self.Nx))

        self.midVal = (360-self.MaxPhase_deg)/2;

        for y in range(self.Ny):
            for x in range(self.Nx):

                self.PhiR_rad[y][x] = -1*self.k0*self.dX*(x*np.sin(self.bf.thetaR_rad)*np.cos(self.bf.phyR_rad) + y*np.sin(self.bf.thetaR_rad)*np.sin(self.bf.phyR_rad))
                self.PhiI_rad[y][x] = +1*self.k0*self.dX*(x*np.sin(self.bf.thetaI_rad)*np.cos(self.bf.phyI_rad) + y*np.sin(self.bf.thetaI_rad)*np.sin(self.bf.phyI_rad))

                self.PhiR_deg[y][x] = rad2deg(self.PhiR_rad[y][x])
                self.PhiI_deg[y][x] = rad2deg(self.PhiI_rad[y][x])

                # Y = (self.PhiR_deg[y][x] - self.PhiI_deg[y][x]) - self.dPhy_deg;
                Y = (self.PhiR_deg[y][x] - self.PhiI_deg[y][x])

                while Y>=360:
                    Y = Y-360

                if (Y>self.MaxPhase_deg) and (Y<360) and ((Y-self.MaxPhase_deg)<self.midVal) :
                    self.phase_limit_deg[y][x] = self.MaxPhase_deg
                    self.phase_ideal_deg[y][x] = Y
                elif (Y>self.MaxPhase_deg) and (Y<360) and ((360-Y)<=self.midVal) :
                    self.phase_limit_deg[y][x] = 0
                    self.phase_ideal_deg[y][x] = Y
                else:
                    self.phase_limit_deg[y][x] = Y
                    self.phase_ideal_deg[y][x] = Y

                self.AF.PhiR_ideal_deg[y][x] = self.phase_ideal_deg[y][x] + self.PhiI_deg[y][x]
                self.AF.PhiR_limit_deg[y][x] = self.phase_limit_deg[y][x] + self.PhiI_deg[y][x]

        self.phase_limit_rad = deg2rad(self.phase_limit_deg)
        self.phase_ideal_rad = deg2rad(self.phase_ideal_deg)
        self.AF.PhiR_limit_rad = deg2rad(self.AF.PhiR_limit_deg)
        self.AF.PhiR_ideal_rad = deg2rad(self.AF.PhiR_ideal_deg)

    def getVoltageAtUnitCell(self):
        self.voltage = np.empty(self.Nx)
        self.voltage_index = np.empty(self.Nx)
        self.ampl = np.empty(self.Nx)
        for i in range(self.Nx):
            diff = abs(self.uc.phase - self.phase_limit_deg[0][i]);
            index = np.argmin(diff);
            self.voltage_index[i] = index;
            self.voltage[i] = self.uc.V[index];
            self.ampl[i] = self.uc.ampl[index];

class BeamForming:
    def __init__(self, thetaR_deg, thetaI_deg, phyR_deg, phyI_deg):
        self.thetaR_deg = thetaR_deg
        self.thetaI_deg = thetaI_deg
        self.phyR_deg   = phyR_deg
        self.phyI_deg   = phyI_deg

        self.thetaR_rad = deg2rad(self.thetaR_deg)
        self.thetaI_rad = deg2rad(self.thetaI_deg)
        self.phyR_rad   = deg2rad(self.phyR_deg)
        self.phyI_rad   = deg2rad(self.phyI_deg)

        self.nI = 1
        self.nR = 1

class UnitCell:
    def __init__(self, minAmpl, phaseSpan, F):
        self.phaseSpan = phaseSpan
        self.F = F
        
        self.L = (1/(2*np.pi)) * (1/10)
        self.C = (1/(2*np.pi)) * (1/10)
        self.R = 2
        self.minAmpl = minAmpl
        self.fres = 1 / (2*np.pi*np.sqrt(self.L*self.C))

        self.getZ()
        self.ampl  = -1*np.abs(self.Z)/self.R + 1
        self.ampl = (self.ampl*(1-self.minAmpl)) + self.minAmpl
        self.phase = -2*rad2deg(np.angle(self.Z))
        self.phase = self.phase - self.phase[np.argmin(self.phase)]
        self.phase = self.phase* (self.phaseSpan/self.phase[np.argmax(self.phase)])

    def getZ(self):
        self.Nv = 2001
        self.V = np.linspace(0, 20, self.Nv)
        self.Z = np.zeros(self.Nv, dtype=complex)
        for i in range(self.Nv):
            self.Z[i] = self.resonance(self.V[i])
    def resonance(self, f):
        if(f == 0): 
            f = 1e-9
        ZL = 1j*2*np.pi*f*self.L
        ZC = 1 / (1j*2*np.pi*f*self.C)
        G = (1/self.R) + (1/ZL) + (1/ZC)
        Z = 1/G
        return Z



if __name__ == "__main__":
    main()



