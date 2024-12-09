#!/usr/bin/env python3
#-----------------------------------------------------------------------------#
#           Group on Data Assimilation Development - GDAD/CPTEC/INPE          #
#-----------------------------------------------------------------------------#
#BOP
#
# !SCRIPT: gsiDiag.py
#
# !DESCRIPTION: Class to read and plot GSI diagnostics files.
#
# !CALLING SEQUENCE:
#
# !REVISION HISTORY: 
# 09 out 2017 - J. G. de Mattos - Initial Version
# 08 may 2024 - L. F. Saoycci - Radiance initial version
#
# !REMARKS:
#   This version can be used not only with conventional diganostics files by radiance too.
#
#EOP
#-----------------------------------------------------------------------------#
#BOC

"""
This module defines the majority of gsidiag functions, including all plot types
"""
from diag2python import diag2python as d2p
from .datasources import getVarInfo
import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cartopy import crs as ccrs
import gc
import sys
from textwrap import wrap
import matplotlib as mpl
import matplotlib.ticker as mticker
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
import re

def help():
    print('Esta é uma ajudada')

def getColor(minVal, maxVal, value, hex=False, cmapName=None):

    try:
       import matplotlib.cm as cm
       from matplotlib.colors import Normalize
       from matplotlib.colors import rgb2hex
    except ImportError:
       pass # module doesn't exist, deal with it.

    if cmapName is None:
        cmapName='Paired'

    # Get a color map
    cmap = cm.get_cmap(cmapName)

    # Get normalize function (takes data in range [vmin, vmax] -> [0, 1])
    norm = Normalize(vmin=minVal, vmax=maxVal)
    
    if hasattr(value,'__iter__'):

       color = []
       for i in range(len(value)):
          if hex is True:
              color.append(rgb2hex(cmap(norm(value[i]))))
          else:
              color.append(cmap(norm(value[i]),bytes=True))

    else:

        if hex is True:
            color = rgb2hex(cmap(norm(value)))
        else:
            color = cmap(norm(value),bytes=True)

    return color

def geoMap(area=None,**kwargs):
    
    if 'ax' not in kwargs:
        fig  = plt.figure(figsize=(12, 6))
        ax   = fig.add_subplot(1, 1, 1)#, projection=ccrs.PlateCarree())
    else:
        ax = kwargs['ax']
        del kwargs['ax']

    
    path=gpd.datasets.get_path('naturalearth_lowres')
    
    world = gpd.read_file(path)
    gdp_max = world['gdp_md_est'].max()
    gdp_min = world['gdp_md_est'].min()
    
    ax = world.plot(ax=ax, facecolor='lightgrey', edgecolor='k')#,**kwargs)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    # set axis range
    if area:
        ax.set_xlim([area[0],area[2]])
        ax.set_ylim([area[1],area[3]])
    else:
        ax.set_xlim([-180,180])
        ax.set_ylim([ -90, 90])

    return ax
    
class setcolor:
    HEADER    = '\033[95m'
    OKBLUE    = '\033[94m'
    OKGREEN   = '\033[92m'
    WARNING   = '\033[93m'
    FAIL      = '\033[91m'
    ENDC      = '\033[0m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'

class read_diag(object):

    """
    read a diagnostic file from gsi. Return an array with
    some information.
    """
    #@profile(precision=8)

    def __init__(self, diagFile, diagFileAnl=None, isisList=None, zlevs=None, zchan=None):

        print(' ')
        print('>>> GSI DIAG <<<')
        print(' ')

        self._diagFile     = diagFile
        self._diagFileAnl  = diagFileAnl

        if diagFileAnl == None:
            extraInfo = False
        else:
            extraInfo = True

        convIndex =['lat','lon', 'elev', 'prs', 'hgt', 'press', 'time', 'idqc', 'iuse', 'iusev', 
                   'wpbqc', 'inp_err', 'adj_err', 'inverr', 'oer', 'obs', 'omf', 'oma', 'imp', 'dfs']

        radIndex  = ['lat','lon','elev','nchan','time','iuse','idqc','inverr','oer','obs',
                     'omf','omf_nobc','emiss','oma','oma_nobc','imp','dfs']
                     
        if isisList is None:
            isis = np.array(['None'],dtype='c').T
        else:
            # put all string with same length
            s=len(max(isisList,key=len))
            l=[]
            for i in isisList:
                l.append(i.ljust(s,' '))
            isis = np.array(l,dtype='c').T
            
        self._FNumber    = d2p.open(self._diagFile, self._diagFileAnl, isis)
        if (self._FNumber <= -1):
            self._FNumber = None
            print('Some was was wrong during reading files ...')
            return

        self._FileType   = d2p.getFileType(self._FNumber)
        if (self._FileType == -1):
            print('Some wrong was happening!')
            return
        
        self._undef = d2p.getUndef(self._FNumber)
        
        # set default levels to obtain data information
        if zlevs is None:
           self.zlevs = [1000.0,900.0,800.0,700.0,600.0,500.0,400.0,300.0,250.0,200.0,150.0,100.0,50.0,0.0]
        else:
           self.zlevs = zlevs
        
        #
        # Get extra informations
        #

        self._nVars     = d2p.getnvars(self._FNumber)
        vnames,nTypes   = d2p.getObsVarInfo(self._FNumber,self._nVars);
        self.varNames   = []
        self.obsInfo    = {}
        for i, name in enumerate(vnames):
            obsName = name.tostring().decode('UTF-8').strip()
            self.varNames.append(obsName)
            vTypes, svTypes = d2p.getVarTypes(self._FNumber,obsName, nTypes[i])
            sTypes = svTypes.tostring().decode('UTF-8').strip().split()
            df = {}
            if self._FileType == 1:
            # for convetional data
               for i, vType in enumerate(vTypes):
                   nObs = d2p.getobs(self._FNumber, obsName, vType, 'None', self.zlevs, len(self.zlevs))
                   if extraInfo is True:
                       d = pd.DataFrame(d2p.array2d.copy().T,index=convIndex).T
                       d2p.array2d = None
                   else:
                       d = pd.DataFrame(d2p.array2d.copy().T,index=convIndex[:17]).T
                       d2p.array2d = None

                   # convert all undef to NaN
                   d.replace(to_replace = self._undef,
                             value      = np.nan,
                             inplace    = True)

                   lon = (d.lon + 180) % 360 - 180
                   lat = d.lat
                   df[vType] = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(lon,lat))
                
            elif self._FileType == 2:
            # for satellite data
               for i, sType in enumerate(sTypes):
                   nObs = d2p.getobs(self._FNumber, obsName, 0, sType, self.zlevs, len(self.zlevs))
                   if extraInfo is True:
                       d   = pd.DataFrame(d2p.array2d.copy().T,index=radIndex).T
                       d2p.array2d = None
                   else:
                       d = pd.DataFrame(d2p.array2d.copy().T,index=radIndex[:13]).T
                       d2p.array2d = None

                   # convert all undef to NaN
                   d.replace(to_replace = self._undef,
                             value      = np.nan,
                             inplace    = True)

                   lon = (d.lon + 180) % 360 - 180
                   lat = d.lat
                   df[sType] = gpd.GeoDataFrame(d, geometry=gpd.points_from_xy(lon,lat))


            if self._FileType == 1:
                self.obsInfo[obsName] = pd.concat(df.values(),keys=df.keys(), names=['kx','points'])
            elif self._FileType == 2:
                self.obsInfo[obsName] = pd.concat(df.values(),keys=df.keys(), names=['SatId','points'])
            
        self.obs  = pd.concat(self.obsInfo, sort=False).reset_index(level=2, drop=True)

    def overview(self):

        """
        Creates a dictionary of the existing variables and types. Returns a Python dictionary.

        Usage: overview()
        """

        variablesList = {}
        for var in self.varNames:
            variablesTypes = []
            for kx in self.obsInfo[var].index.levels[0]:
                variablesTypes.append(kx)
            variablesList.update({var:variablesTypes})
        return variablesList

    def pfileinfo(self):

        """
        Prints a fancy list of the existing variables and types.

        Usage: pfileinfo()
        """

        for name in self.varNames:
            print('Variable Name :',name)
            print('              └── kx => ', end='', flush=True)
            for kx in self.obsInfo[name].index.levels[0]:
               print(kx,' ', end='', flush=True)
            print()

            print()
       
    def close(self):

        """
        Closes a previous openned file. Returns an integer status value.

        Usage: close()
        """

        iret = d2p.close(self._FNumber)
        self._FileName = None # File name
        self._FNumber  = None # File unit number to be closed
        self._nVars    = None # Total of variables
        self.varNames  = None # Name of variables
        self.obsInfo   = None 
        self.nObs      = None # Number of observations for vName
        del self
        gc.collect()
        
        return iret
    @staticmethod
    def tocsv(self, varName=None, varType=None, dateIni=None, dateFin=None, nHour="06", Level=None, Lay = None, SingleL=None):
        
        '''
        The function tocsv is similar to the time_series funcion, however, it outputs a CSV file instead of figures. 
        Refers to the time_series description below for more information.

        '''

        delta = nHour
        omflag = "OmF"
        omflaga = "OmA"

        Laydef = 50

        separator = " ====================================================================================================="

        print()
        print(separator)
        #print(" Reading dataset in " + data_path)
        varInfo = getVarInfo(varType, varName, 'instrument')
        if varInfo is not None:
            print(" Analyzing data of variable: " + varName + "  ||  type: " + str(varType) + "  ||  " + varInfo + "  ||  check: " + omflag)
        else:
            print(" Analyzing data of variable: " + varName + "  ||  type: " + str(varType) + "  ||  Unknown instrument  ||  check: " + omflag)
        print(separator)
        print()

        zlevs_def = list(map(int,self[0].zlevs))


        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        levs_tmp, DayHour_tmp = [], []
        info_check = {}
        f = 0
        while (date <= datef):
            
            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            dataDict = self[f].obsInfo[varName].loc[varType]
            info_check.update({date.strftime("%d%H"):True})

            if 'prs' in dataDict and (Level == None or Level == "Zlevs"):
                if(Level == None):
                    levs_tmp.extend(list(set(map(int,dataDict['prs']))))
                else:
                    levs_tmp = zlevs_def[::-1]
                info_check.update({date.strftime("%d%H"):True})
                print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"))
                print(' Levels: ', sorted(levs_tmp), end='\n')
                print("")
                f = f + 1
            else:
                if (Level != None and Level != "Zlevs") and info_check[date.strftime("%d%H")] == True:
                    levs_tmp.extend([Level])
                    print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"), ' - Level: ', Level , end='\n')
                    f = f + 1
                else:
                    info_check.update({date.strftime("%d%H"):False})
                    print(date.strftime(setcolor.WARNING + ' Preparing data for: ' + "%Y-%m-%d:%H"), ' - No information on this date ' + setcolor.ENDC, end='\n')

            del(dataDict)
            
            date = date + timedelta(hours=int(delta))
        

        if(len(DayHour_tmp) > 4):
            DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
        else:
            DayHour = DayHour_tmp

        zlevs = [z if z in zlevs_def else "" for z in sorted(set(levs_tmp+zlevs_def))]

        print()
        print(separator)
        print()

        list_meanByLevs, list_stdByLevs, list_countByLevs = [], [], []
        list_meanByLevsa, list_stdByLevsa, list_countByLevsa = [], [], []
        date = datei
        levs = sorted(list(set(levs_tmp)))
        levs_tmp.clear()
        del(levs_tmp[:])

        head_levs = ['datetime']
        for lev in levs:
            head_levs.append('mean'+str(lev))
            head_levs.append('std'+str(lev))
            head_levs.append('count'+str(lev))

        dset = []
        dseta = []
        f = 0
        while (date <= datef):

            print(date.strftime(' Calculating for ' + "%Y-%m-%d:%H"))
            datefmt = date.strftime("%Y%m%d%H")

            try:
                dataByLevs, mean_dataByLevs, std_dataByLevs, count_dataByLevs = {}, {}, {}, {}
                dataByLevsa, mean_dataByLevsa, std_dataByLevsa, count_dataByLevsa = {}, {}, {}, {}
                
                if info_check[date.strftime("%d%H")] == True:
                    dataDict = self[f].obsInfo[varName].loc[varType]

                    [dataByLevs.update({int(lvl): []}) for lvl in levs]
                    [dataByLevsa.update({int(lvl): []}) for lvl in levs]
                    if Level != None and Level != "Zlevs":
                        if SingleL == None:
                            [ dataByLevs[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].omf) if int(p) == Level ]
                            [ dataByLevsa[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].oma) if int(p) == Level ]
                            forplot = ' Level='+str(Level) +'hPa'
                            forplotname = 'level_'+str(Level) +'hPa'
                        else:
                            if SingleL == "All":
                                [ dataByLevs[Level].append(v) for v in self[f].obsInfo[varName].loc[varType].omf ]
                                [ dataByLevsa[Level].append(v) for v in self[f].obsInfo[varName].loc[varType].oma ]
                                forplot = ' Layer=Entire Atmosphere'
                                forplotname = 'layer_allAtm'
                            else:
                                if SingleL == "OneL":
                                    if Lay == None:
                                        print("")
                                        print(" Variable Lay is None, resetting it to its default value: "+str(Laydef)+" hPa.")
                                        print("")
                                        Lay = Laydef
                                    [ dataByLevs[int(Level)].append(v) for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].omf) if int(p) >=Level-Lay and int(p) <Level+Lay ]
                                    [ dataByLevsa[int(Level)].append(v) for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].oma) if int(p) >=Level-Lay and int(p) <Level+Lay ]
                                    forplot = ' Layer='+str(Level+Lay)+'-'+str(Level-Lay)+'hPa'
                                    forplotname = 'layer_'+str(Level+Lay)+'-'+str(Level-Lay)+'hPa'
                                else:
                                    print(" Wrong value for variable SingleL. Please, check it and rerun the script.")    
                    else:
                        if Level == None:
                            [ dataByLevs[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].omf) ]
                            [ dataByLevsa[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].oma) ]
                            forplotname = 'all_levels_byLevels'
                        else:
                            for ll in range(len(levs)):
                                if Lay == None:
                                    lv = levs[ll]
                                    if ll == 0:
                                        Llayi = 0
                                    else:
                                        Llayi = (levs[ll] - levs[ll-1]) / 2.0
                                    if ll == len(levs)-1:
                                        Llayf = Llayi
                                    else:
                                        Llayf = (levs[ll+1] - levs[ll]) / 2.0
                                    cutlevs = [ v for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].omf) if int(p) >=lv-Llayi and int(p) <lv+Llayf ]
                                    cutlevsa = [ v for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].oma) if int(p) >=lv-Llayi and int(p) <lv+Llayf ]
                                    forplotname = 'all_levels_filledLayers'
                                else:
                                    cutlevs = [ v for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].omf) if int(p) >=lv-Lay and int(p) <lv+Lay ]
                                    cutlevsa = [ v for p,v in zip(self[f].obsInfo[varName].loc[varType].prs,self[f].obsInfo[varName].loc[varType].oma) if int(p) >=lv-Lay and int(p) <lv+Lay ]
                                    forplotname = 'all_levels_bylayers'
                                [ dataByLevs[lv].append(il) for il in cutlevs ]
                                [ dataByLevsa[lv].append(il) for il in cutlevsa ]
                    f = f + 1
                for lv in levs:
                    if len(dataByLevs[lv]) != 0 and info_check[date.strftime("%d%H")] == True:
                        mean_dataByLevs.update({int(lv): np.mean(np.array(dataByLevs[lv]))})
                        std_dataByLevs.update({int(lv): np.std(np.array(dataByLevs[lv]))})
                        count_dataByLevs.update({int(lv): len(np.array(dataByLevs[lv]))})
                        mean_dataByLevsa.update({int(lv): np.mean(np.array(dataByLevsa[lv]))})
                        std_dataByLevsa.update({int(lv): np.std(np.array(dataByLevsa[lv]))})
                        count_dataByLevsa.update({int(lv): len(np.array(dataByLevsa[lv]))})
                    else:
                        mean_dataByLevs.update({int(lv): -99})
                        std_dataByLevs.update({int(lv): -99})
                        count_dataByLevs.update({int(lv): -99})
                        mean_dataByLevsa.update({int(lv): -99})
                        std_dataByLevsa.update({int(lv): -99})
                        count_dataByLevsa.update({int(lv): -99})
            
            except:
                if info_check[date.strftime("%d%H")] == True:
                    print("ERROR in time_series function.")
                else:
                    print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)

                for lv in levs:
                    mean_dataByLevs.update({int(lv): -99})
                    std_dataByLevs.update({int(lv): -99})
                    count_dataByLevs.update({int(lv): -99})
                    mean_dataByLevsa.update({int(lv): -99})
                    std_dataByLevsa.update({int(lv): -99})
                    count_dataByLevsa.update({int(lv): -99})

            if Level == None or Level == "Zlevs":
                list_meanByLevs.append(list(mean_dataByLevs.values()))
                list_stdByLevs.append(list(std_dataByLevs.values()))
                list_countByLevs.append(list(count_dataByLevs.values()))
                list_meanByLevsa.append(list(mean_dataByLevsa.values()))
                list_stdByLevsa.append(list(std_dataByLevsa.values()))
                list_countByLevsa.append(list(count_dataByLevsa.values()))
            else:
                list_meanByLevs.append(mean_dataByLevs[int(Level)])
                list_stdByLevs.append(std_dataByLevs[int(Level)])
                list_countByLevs.append(count_dataByLevs[int(Level)])
                list_meanByLevsa.append(mean_dataByLevsa[int(Level)])
                list_stdByLevsa.append(std_dataByLevsa[int(Level)])
                list_countByLevsa.append(count_dataByLevsa[int(Level)])
                
            dataByLevs.clear()
            mean_dataByLevs.clear()
            std_dataByLevs.clear()
            count_dataByLevs.clear()
            dataByLevsa.clear()
            mean_dataByLevsa.clear()
            std_dataByLevsa.clear()
            count_dataByLevsa.clear()

            date_finale = date
            date = date + timedelta(hours=int(delta))

            values_levs = [datefmt]
            values_levsa = [datefmt]
            for me,sd,nd in zip(list_meanByLevs[-1][:],list_stdByLevs[-1][:],list_countByLevs[-1][:]):
                values_levs.append(me)
                values_levs.append(sd)
                values_levs.append(nd)
            for me,sd,nd in zip(list_meanByLevsa[-1][:],list_stdByLevsa[-1][:],list_countByLevsa[-1][:]):
                values_levsa.append(me)
                values_levsa.append(sd)
                values_levsa.append(nd)
            dset.append(values_levs)
            dseta.append(values_levsa)

        print()
        print(separator)
        print()


        # ==============================================================================================================
        # Save dataset into CSV File ===================================================================================

        print("\n Saving Dataset in CSV File...  ")

        dataout_file  = 'dataout_' + str(varName) + '_' + str(varType) + '_' + omflag  + '.csv'
        dataout_filea = 'dataout_' + str(varName) + '_' + str(varType) + '_' + omflaga + '.csv'
        df = pd.DataFrame.from_records(dset, columns=head_levs)
        df.to_csv(dataout_file, index=False)
        del(df)
        df = pd.DataFrame.from_records(dseta, columns=head_levs)
        df.to_csv(dataout_filea, index=False)
        del(df)
        print(" Done \n")

        return

# #-------------- Nova função: ler arquivo fort.220 --------------------#
    def fort_220_cost_gradient(DIRdiag, dates):
        """ 
        read files fort.220 from gsi. Return an list of tables with convergence information of the minimization process.
        
        1. The values of cost function:
        
        -> Jb: background term;          -> Jc: dry pressure constraint term;
        -> Jo: observations term;        -> Jl: negative and excess moisture term.
        
        2. The value of the cost function and norm of the gradient:
        
        -> cost: the cost function values;
        -> grad: inner product of gradients (norm of the gradient (Y*X));
        -> step: stepsize;
        -> b: parameter to estimate the new search direction
        
        This function return a list of the tables for first e second outer loops.
        The columns identify flags and lines identify inner loop.
        
        """
        
        print('DIRdiag = ',DIRdiag)
        print('')
        
        
        names_columns = ['InnerLoop', 'Jb', 'Jo', 'Jc', 'Jl', 'cost', 'grad', 'step', 'b', 'step?']
        
        print('Names columns = ',names_columns)
        print('')
        
        pathsf, self = [], []
        [pathsf.append(DIRdiag + '/' + dt + '/fort.220') for dt in dates]
        
        print(pathsf)
        print('')
        
        tidx = 0
        for path in pathsf:
            print('Reading ' + path)
            print('')
            
            date = datetime.strptime(str(dates[tidx]), "%Y%m%d%H")
            print(date.strftime(' Preparing data for: fort.220 ---> ' + "%Y-%m-%d:%H"))
            print('')
            
            with open(path, "r") as file:
                lines = file.readlines()
                
                List_data = []
                MinItera  = []
                data1 = []
                
                for line in lines:
                    
                    #---------------- Iterations -----------------#
                    if re.search('Minimization iteration', line):
                        itera = None
                        itera = line
                        itera = re.findall(r'\d+', itera)
                        
                        MinItera.append(int(itera[0]))
                        inner = int(itera[0])
                    
                    #---------------- costterms -----------------#
                    if re.search('costterms Jb,Jo,Jc,Jl  =', line):
                        data1 = []
                        data1 = line
                        data1 = re.findall(r'[-+]?\d+\.\d+\D+\d+', data1)
                        
                        for lv in range(0,len(data1),1):
                            data1[lv] = float(data1[lv])
                               
                    #---------------- cost,grad,step,b,step? -----------------#
                    if re.search('cost,grad,step,b,step', line):
                        data2 = []
                        data2 = line
                        data2 = re.findall(r'[-+]?\d+\.\d+\D+\d+', data2)
                        if re.search('good', line):
                            aux2 = 'good'
                        else:
                            aux2 = None   ## Verificar ????!!!
                            
                        for lv in range(0,len(data2),1):
                            data2[lv] = float(data2[lv])
                            
                        
                        if (len(data1)>0 or len(data2)>0):
                            data = []
                            if len(data1) > 0:
                                data = data1
                            else:
                                data = [None, None, None, None]
                                
                            for lv in data2:
                                data.append(lv)
                        
                            data.append(aux2)    # inserted the word good in finish position
                            data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        
                            List_data.append(data)
                        
                        else:
                            print("++++++++++++++++++++++++++ ERROR: file reading --> fort_220_cost_gradient ++++++++++++++++++++++++++")
                            print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                        
                
            file.close()
            print('')
            
            InnerLoop1, InnerLoop2 = [], []
            
            for it in range(0, len(MinItera), 1):
                if MinItera[it] > MinItera[it+1]:
                    InnerLoop1.append(MinItera[it])
                    break
                else:
                    InnerLoop1.append(MinItera[it])
            
            
            InnerLoop2 = MinItera[it+1:]
            
            
            index1 = InnerLoop1
            index2 = InnerLoop2
            
            nloop_1 = InnerLoop1[-1]
            
            df1 = pd.DataFrame(List_data[0:nloop_1+1], columns=names_columns, index=index1)
            df2 = pd.DataFrame(List_data[nloop_1+1:], columns=names_columns, index=index2)
            
            self.append([df1, df2])
            tidx = tidx + 1
            
            
        return self
    

# #-------------- Nova função: ler arquivo fort.220 --------------------#
    def fort_220_Flags_V3_4(DIRdiag, dates):
        """ 
        read files fort.220 from gsi (V 3.4). Return an list of tables with detailed convergence information of the minimization process.
        
        1. Labels detailed:
        
        -> J: cost function;                       ->  c: c term for estimate stepsize;
        -> b: b term for estimate stepsize;        -> EJ: estimate terms in penalty.
        
        
        ! Contributions to penalty for various terms.
        !
        ! Linear terms:
        !
        !  Flag | Observation types
        !  +----+-----------------------------------------------------------
        !    1  | contribution from background, sat rad bias, and precip bias
        !    2  | placeholder for future linear linear term
        !    3  | contribution from dry pressure constraint term (Jc)
        !
        ! Nonlinear terms:
        !
        !  Flag | Observation types
        !  +----+-----------------------------------------------------------
        !    4  | contribution from negative moisture constraint term (Jl/Jq)
        !    5  | contribution from excess moisture term (Jl/Jq)
        !    6  | contribution from negative gust constraint term (Jo)
        !    7  | contribution from negative vis constraint term (Jo)
        !    8  | contribution from negative pblh constraint term (Jo)
        !
        !-----------------------------------------------------------------------------------------------------------------------
        ! The list below is different from the list in the Advanced User's guide version 3.4
        !-----------------------------------------------------------------------------------------------------------------------
        !
        ! The list below is new information addeded in addition to the list described in the Advanced User's guide version 3.4
        !
        !  Flag | Observation types
        !  +----+-----------------------------------------------------------
        !    9  | contribution from negative wspd10m constraint term (Jo)
        !    10 | contribution from negative howv constraint term (Jo)
        !    11 | contribution from negative lcbas constraint term (Jo)
        !
        !------------------------------------------------------------------------------------------
        !
        ! The list below reffers to the list from 9-32 of the Advanced User's guide version 3.4
        !
        !  Flag | Observation types                                       |Flag | Observation types
        !  +----+---------------------------------------------------------+-----+-----------------------------------------------
        !    12 | contribution from ps observation  term (Jo)             |  25 | contribution from gps observation  term (Jo)
        !    13 | contribution from t observation  term (Jo)              |  26 | contribution from rad observation  term (Jo)
        !    14 | contribution from w observation  term (Jo)              |  27 | contribution from tcp observation  term (Jo)
        !    15 | contribution from q observation  term (Jo)              |  28 | contribution from lag observation  term (Jo)
        !    16 | contribution from spd observation  term (Jo)            |  29 | contribution from colvk observation  term (Jo)
        !    17 | contribution from srw observation  term (Jo)            |  30 | contribution from aero observation  term (Jo)
        !    18 | contribution from rw observation  term (Jo)             |  31 | contribution from aerol observation  term (Jo)
        !    19 | contribution from dw observation  term (Jo)             |  32 | contribution from pm2_5 observation  term (Jo)
        !    20 | contribution from sst observation  term (Jo)            |  33 | contribution from gust observation  term (Jo)
        !    21 | contribution from pw observation  term (Jo)             |  34 | contribution from vis observation  term (Jo)
        !    22 | contribution from pcp observation  term (Jo)            |  35 | contribution from pblh observation  term (Jo)
        !    23 | contribution from oz observation  term (Jo)             |
        !    24 | contribution from o3l observation  term (Jo)(not used)  |
        !
        !-----------------------------------------------------------------------------------------
        ! The list below is new information addeded in addition to the list described in the Advanced User's guide version 3.4
        !
        !  Flag | Observation types                                       |Flag | Observation types
        !  +----+---------------------------------------------------------+-----+-----------------------------------------------
        !    36 | contribution from wspd10m observation  term (Jo)        |  40 | contribution from pmsl observation  term (Jo)
        !    37 | contribution from td2m observation  term (Jo)           |  41 | contribution from howv observation  term (Jo)
        !    38 | contribution from mxtm observation  term (Jo)           |  42 | contribution from tcamt observation  term (Jo)
        !    39 | contribution from mitm observation  term (Jo)           |  43 | contribution from lcbas observation  term (Jo)
        !

        
        This function return a list of the tables for first e second outer loops. 
        The columns identify flags and lines identify the tuples (inner loop, labels).
        
        """
        
        print('DIRdiag = ',DIRdiag)
        print('')
        
        
        Labels = ['J', 'b', 'c', 'EJ']
        temp_Labels = [None, None, None, None]
        flags =[]
        n_flag = 43   # Qntdd de flags
        Li = 15       # Para uma iteração e label: qntdd de linhas sequenciais com as valores de contribuicao de cada flag
        [flags.append(str(i)) for i in range(1,n_flag+1,1)]
        
        print('Labels = ',Labels)
        print('')
        
        print('Flags = ',flags)
        print('')
        
        flags.insert(0,'InnerLoop') # inserted inner loop in first column
        print('Columns: ', flags)
        print('')
        
        pathsf, self = [], []
        [pathsf.append(DIRdiag + '/' + dt + '/fort.220') for dt in dates]
        
        print(pathsf)
        print('')
        
        tidx = 0
        for path in pathsf:
            print('Reading ' + path)
            print('')
            
            date = datetime.strptime(str(dates[tidx]), "%Y%m%d%H")
            print(date.strftime(' Preparing data for: fort.220 ---> ' + "%Y-%m-%d:%H"))
            print('')
            
            with open(path, "r") as file:
                
                lines = file.readlines()
                nlines = len(lines)
                
                print('Nº total de linhas =', nlines)
                
                List_data = []
                MinItera  = []
                
                nl = 0
                while ( nl < nlines ):
                    line = lines[nl]
                    
                    #---------------- Iterations -----------------#
                    if re.search('Minimization iteration', line):
                        itera = None
                        itera = line
                        itera = re.findall(r'\d+', itera)
                        
                        MinItera.append(int(itera[0]))
                        inner = int(itera[0])
                    
                    #---------------- Label J -----------------#
                    if re.search(' J=', line):
                        data = []
                        data = line
                        data = re.findall(r'[-+]?\d+\.\d+\D+\d+', data)
                        for ll in range(1,Li,1):
                            aux = lines[lines.index(line) + ll]
                            aux = re.findall(r'[-+]?\d+\.\d+\D+\d+', aux)
                            [data.append(i) for i in aux]
                        
                        for lv in range(0,len(data),1):
                            data[lv] = float(data[lv])
                            
                        
                        if (len(data)>n_flag):
                            print('len data J =', len(data), 'index line =', lines.index(line))
                            print('data J =',data)
                            print('')
                        
                        data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        List_data.append(data)
                        temp_Labels[0] = 'J'
                        
                        # para encontrar b soma 15 linhas (as matrizes são escritas em sequências de 15 linhas)
                        nl = nl + Li
                        line = lines[nl]
                        
                    #---------------- Label b -----------------#
                    if ( re.search('b=', line) and re.search(' J=', lines[nl-Li]) ):
                        data = []
                        data = line
                        data = re.findall(r'[-+]?\d+\.\d+\D+\d+', data)
                        for ll in range(1,Li,1):
                            aux = lines[lines.index(line) + ll]
                            aux = re.findall(r'[-+]?\d+\.\d+\D+\d+', aux)
                            [data.append(i) for i in aux]
                            
                        for lv in range(0,len(data),1):
                            data[lv] = float(data[lv])
                            
                        
                        if (len(data)>n_flag):
                            print('len data b =', len(data), 'index line =', lines.index(line))
                            print('data b =',data)
                            print('')
                            
                        data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        List_data.append(data)
                        temp_Labels[1] = 'b'
                        
                        nl = nl + Li
                        line = lines[nl]
                        
                    #---------------- Label c -----------------#
                    if re.search('c=', line):
                        data = []
                        data = line
                        data = re.findall(r'[-+]?\d+\.\d+\D+\d+', data)
                        for ll in range(1,Li,1):
                            aux = lines[lines.index(line) + ll]
                            aux = re.findall(r'[-+]?\d+\.\d+\D+\d+', aux)
                            [data.append(i) for i in aux]
                            
                        for lv in range(0,len(data),1):
                            data[lv] = float(data[lv])
                            
                        
                        if (len(data)>n_flag):
                            print('len data c =', len(data), 'index line =', lines.index(line))
                            print('data c =',data)
                            print('')
                            
                        data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        List_data.append(data)
                        temp_Labels[2] = 'c'
                        
                        nl = nl + Li
                        line = lines[nl]
                        
                    #---------------- Label EJ -----------------#
                    if re.search('EJ=', line):
                        data = []
                        data = line
                        data = re.findall(r'[-+]?\d+\.\d+\D+\d+', data)
                        for ll in range(1,Li,1):
                            aux = lines[lines.index(line) + ll]
                            aux = re.findall(r'[-+]?\d+\.\d+\D+\d+', aux)
                            [data.append(i) for i in aux]
                            
                        for lv in range(0,len(data),1):
                            data[lv] = float(data[lv])
                            
                        
                        if (len(data)>n_flag):
                            print('len data EJ =', len(data), 'index line =', lines.index(line))
                            print('data EJ =',data)
                            print('')
                            
                        data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        List_data.append(data)
                        temp_Labels[3] = ('EJ')
                        
                        nl = nl + Li
                        line = lines[nl]
                    
                    # próxima linha
                    nl = nl + 1
                    
                print('nl final =', nl)
            
            file.close()
            print('')
            
            
            InnerLoop1, InnerLoop2 = [], []
            
            for it in range(0, len(MinItera), 1):
                if MinItera[it] > MinItera[it+1]:
                    InnerLoop1.append(MinItera[it])
                    break
                else:
                    InnerLoop1.append(MinItera[it])

            
            InnerLoop2 = MinItera[it+1:]
            
            check_Labels = []
            [check_Labels.append(x) for x in Labels if x in temp_Labels]
            
            tuples1 = [(Il, lb) for Il in InnerLoop1 for lb in Labels if lb in temp_Labels ]
            tuples2 = [(Il, lb) for Il in InnerLoop2 for lb in Labels if lb in temp_Labels ]
            
            index1 = pd.MultiIndex.from_tuples(tuples1,names=['Inner loop', 'Label'])
            index2 = pd.MultiIndex.from_tuples(tuples2,names=['Inner loop', 'Label'])
            
            
            nloop_1 = InnerLoop1[-1]
            nlabels = len(check_Labels)
            
            df1 = pd.DataFrame(List_data[0:nlabels*(nloop_1+1)], columns=flags, index=index1)
            df2 = pd.DataFrame(List_data[nlabels*(nloop_1+1):], columns=flags, index=index2)
            
            df1 = df1.rename_axis("Flags", axis="columns")
            df2 = df2.rename_axis("Flags", axis="columns")
            
            self.append([df1, df2])
            tidx = tidx + 1
            
            
        return self
    
# #-------------- Nova função: ler arquivo fort.220 --------------------#
    def fort_220_Flags_V3_7(DIRdiag, dates):
        """ 
        read files fort.220 from gsi (V 3.7). Return an list of tables with detailed convergence information of the minimization process.
        
        1. Labels detailed:
        
        -> J: cost function;                       ->  c: c term for estimate stepsize;
        -> b: b term for estimate stepsize;        -> EJ: estimate terms in penalty.
        
        
        ! Contributions to penalty for various terms.
        !
        ! Linear terms:
        !
        !  Flag | Observation types
        !  +----+-----------------------------------------------------------
        !    1  | contribution from background, sat rad bias, and precip bias
        !    2  | placeholder for future linear linear term
        !    3  | contribution from dry pressure constraint term (Jc)
        !
        ! Nonlinear terms:
        !
        !  Flag | Observation types
        !  +----+-----------------------------------------------------------
        !    4  | contribution from negative moisture constraint term (Jl/Jq)
        !    5  | contribution from excess moisture term (Jl/Jq)
        !    6  | contribution from negative gust constraint term (Jo)
        !    7  | contribution from negative vis constraint term (Jo)
        !    8  | contribution from negative pblh constraint term (Jo)
        !    9  | contribution from negative wspd10m constraint term (Jo)
        !    10 | contribution from negative howv constraint term (Jo)
        !    11 | contribution from negative lcbas constraint term (Jo)
        !    12 | contribution from negative cldch constraint term (Jo)
        !    13 | contribution from negative ql constraint term (Jl/Jg)
        !    14 | contribution from negative qi constraint term (Jl/Jg)
        !    15 | contribution from negative qr constraint term (Jl/Jg)
        !    16 | contribution from negative qs constraint term (Jl/Jg)
        !    17 | contribution from negative qg constraint term (Jl/Jg)
        !
        !-----------------------------------------------------------------------------------------------------------------
        !    Under polymorphism the following is the contents of pbs:
        !    linear terms => pbcjo(*,n0+1:n0+nobs_type),
        !       pbc  (*,n0+j) := pbcjo(*,j); for j=1,nobs_type
        !    where,
        !       pbcjo(*,   j) := sum( pbcjoi(*,j,1:nobs_bins) )
        !-----------------------------------------------------------------------------------------------------------------
        !    The original (wired) implementation of obs-types has
        !    the extra contents of pbc defined as:
        !-----------------------------------------------------------------------------------------------------------------
        !  Flag | Observation types                                       |Flag | Observation types
        !  +----+---------------------------------------------------------+-----+-----------------------------------------------
        !    18 | contribution from ps observation  term (Jo)             |  31 | contribution from gps refractivity  observation  term (Jo)
        !    19 | contribution from t observation  term (Jo)              |  32 | contribution from rad observation  term (Jo)
        !    20 | contribution from w observation  term (Jo)              |  33 | contribution from tcp observation  term (Jo)
        !    21 | contribution from q observation  term (Jo)              |  34 | contribution from lag observation  term (Jo)
        !    22 | contribution from spd observation  term (Jo)            |  35 | contribution from colvk observation  term (Jo)
        !    23 | contribution from rw observation  term (Jo)             |  36 | contribution from aero observation  term (Jo)
        !    24 | contribution from dw observation  term (Jo)             |  37 | contribution from aerol observation  term (Jo)
        !    25 | contribution from sst observation  term (Jo)            |  38 | contribution from pm2_5 observation  term (Jo)
        !    26 | contribution from pw observation  term (Jo)             |  39 | contribution from gust observation  term (Jo)
        !    27 | contribution from pcp observation  term (Jo)            |  40 | contribution from vis observation  term (Jo)
        !    28 | contribution from oz observation  term (Jo)             |  41 | contribution from pblh observation  term (Jo)
        !    29 | contribution from o3l observation  term (Jo)(not used)  |
        !    30 | contribution from gps bending angle observation  term (Jo)
        !
        !---------------------------------------------------------------------------------------------------------------------------------
        !  Flag | Observation types                                       |Flag | Observation types
        !  +----+---------------------------------------------------------+-----+----------------------------------------------------------
        !    42 | contribution from wspd10m observation  term (Jo)        |  48 | contribution from tcamt observation  term (Jo)
        !    43 | contribution from td2m observation  term (Jo)           |  49 | contribution from lcbas observation  term (Jo)
        !    44 | contribution from mxtm observation  term (Jo)           |  50 | contribution from pm10 observation  term (Jo)
        !    45 | contribution from mitm observation  term (Jo)           |  51 | contribution from cldch observation  term (Jo)
        !    46 | contribution from pmsl observation  term (Jo)           |  52 | contribution from uwnd10m observation  term (Jo)
        !    47 | contribution from howv observation  term (Jo)           |  53 | contribution from vwnd10m observation  term (Jo)
        !
        !-----------------------------------------------------------------------------------------
        !    Users should be awared that under polymorphism, obOper types are defined on
        !    the fly.  Such that the second index of pbc(*,:) listed above for n0:1 and
        !    above, is no longer reflecting their actual location in arrays, e.g. pbc,
        !    pj, etc..  The actual indices for all obOper types are defined as
        !    enumerators in module gsi_obOperTypeManager, for any given build.  These
        !    indices are referenceable as public iobOper_xxx integer parameters from
        !    there, if one has to know or to reference them explicitly.
        !
        
        This function return a list of the tables for first e second outer loops. 
        The columns identify flags and lines identify the tuples (inner loop, labels).
        
        """
        
        print('DIRdiag = ',DIRdiag)
        print('')
        
        
        Labels = ['J', 'b', 'c', 'EJ']
        temp_Labels = [None, None, None, None]
        flags =[]
        n_flag = 56   # Qntdd de flags
        Li = 19       # Para uma iteração e label: qntdd de linhas sequenciais com as valores de contribuicao de cada flag
        [flags.append(str(i)) for i in range(1,n_flag+1,1)]
        
        print('Labels = ',Labels)
        print('')
        
        print('Flags = ',flags)
        print('')
        
        flags.insert(0,'InnerLoop') # inserted inner loop in first column
        print('Columns: ', flags)
        print('')
        
        pathsf, self = [], []
        [pathsf.append(DIRdiag + '/' + dt + '/fort.220') for dt in dates]
        
        print(pathsf)
        print('')
        
        tidx = 0
        for path in pathsf:
            print('Reading ' + path)
            print('')
            
            date = datetime.strptime(str(dates[tidx]), "%Y%m%d%H")
            print(date.strftime(' Preparing data for: fort.220 ---> ' + "%Y-%m-%d:%H"))
            print('')
            
            with open(path, "r") as file:
                
                lines = file.readlines()
                nlines = len(lines)
                
                print('Nº total de linhas =', nlines)
                
                List_data = []
                MinItera  = []
                
                nl = 0
                while ( nl < nlines ):
                    line = lines[nl]
                    
                    #---------------- Iterations -----------------#
                    if re.search('Minimization iteration', line):
                        itera = None
                        itera = line
                        itera = re.findall(r'\d+', itera)
                        
                        MinItera.append(int(itera[0]))
                        inner = int(itera[0])
                    
                    #---------------- Label J -----------------#
                    if re.search(' J=', line):
                        data = []
                        data = line
                        data = re.findall(r'[-+]?\d+\.\d+\D+\d+', data)
                        for ll in range(1,Li,1):
                            aux = lines[lines.index(line) + ll]
                            aux = re.findall(r'[-+]?\d+\.\d+\D+\d+', aux)
                            [data.append(i) for i in aux]
                        
                        for lv in range(0,len(data),1):
                            data[lv] = float(data[lv])
                            
                        
                        if (len(data)>n_flag):
                            print('len data J =', len(data), 'index line =', lines.index(line))
                            print('data J =',data)
                            print('')
                        
                        data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        List_data.append(data)
                        temp_Labels[0] = 'J'
                        
                        # para encontrar b soma Li linhas (as matrizes são escritas em sequências de Li linhas)
                        nl = nl + Li
                        line = lines[nl]
                        
                    #---------------- Label b -----------------#
                    if ( re.search('b=', line) and re.search(' J=', lines[nl-Li]) ):
                        data = []
                        data = line
                        data = re.findall(r'[-+]?\d+\.\d+\D+\d+', data)
                        for ll in range(1,Li,1):
                            aux = lines[lines.index(line) + ll]
                            aux = re.findall(r'[-+]?\d+\.\d+\D+\d+', aux)
                            [data.append(i) for i in aux]
                            
                        for lv in range(0,len(data),1):
                            data[lv] = float(data[lv])
                            
                        
                        if (len(data)>n_flag):
                            print('len data b =', len(data), 'index line =', lines.index(line))
                            print('data b =',data)
                            print('')
                            
                        data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        List_data.append(data)
                        temp_Labels[1] = 'b'
                        
                        nl = nl + Li
                        line = lines[nl]
                        
                    #---------------- Label c -----------------#
                    if re.search('c=', line):
                        data = []
                        data = line
                        data = re.findall(r'[-+]?\d+\.\d+\D+\d+', data)
                        for ll in range(1,Li,1):
                            aux = lines[lines.index(line) + ll]
                            aux = re.findall(r'[-+]?\d+\.\d+\D+\d+', aux)
                            [data.append(i) for i in aux]
                            
                        for lv in range(0,len(data),1):
                            data[lv] = float(data[lv])
                            
                        
                        if (len(data)>n_flag):
                            print('len data c =', len(data), 'index line =', lines.index(line))
                            print('data c =',data)
                            print('')
                            
                        data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        List_data.append(data)
                        temp_Labels[2] = 'c'
                        
                        nl = nl + Li
                        line = lines[nl]
                        
                    #---------------- Label EJ -----------------#
                    if re.search('EJ=', line):
                        data = []
                        data = line
                        data = re.findall(r'[-+]?\d+\.\d+\D+\d+', data)
                        for ll in range(1,Li,1):
                            aux = lines[lines.index(line) + ll]
                            aux = re.findall(r'[-+]?\d+\.\d+\D+\d+', aux)
                            [data.append(i) for i in aux]
                            
                        for lv in range(0,len(data),1):
                            data[lv] = float(data[lv])
                            
                        
                        if (len(data)>n_flag):
                            print('len data EJ =', len(data), 'index line =', lines.index(line))
                            print('data EJ =',data)
                            print('')
                            
                        data.insert(0,inner) # inserted the minimization iteration (InnerLoop) in begging position
                        List_data.append(data)
                        temp_Labels[3] = ('EJ')
                        
                        nl = nl + Li
                        line = lines[nl]
                    
                    # próxima linha
                    nl = nl + 1
                    
                print('nl final =', nl)
            
            file.close()
            print('')
            
            InnerLoop1, InnerLoop2 = [], []
            
            for it in range(0, len(MinItera), 1):
                if MinItera[it] > MinItera[it+1]:
                    InnerLoop1.append(MinItera[it])
                    break
                else:
                    InnerLoop1.append(MinItera[it])
            
            InnerLoop2 = MinItera[it+1:]
            
            check_Labels = []
            [check_Labels.append(x) for x in Labels if x in temp_Labels]
            
            tuples1 = [(Il, lb) for Il in InnerLoop1 for lb in Labels if lb in temp_Labels ]
            tuples2 = [(Il, lb) for Il in InnerLoop2 for lb in Labels if lb in temp_Labels ]
            
            index1 = pd.MultiIndex.from_tuples(tuples1,names=['Inner loop', 'Label'])
            index2 = pd.MultiIndex.from_tuples(tuples2,names=['Inner loop', 'Label'])
            
            nloop_1 = InnerLoop1[-1]
            nlabels = len(check_Labels)
            
            df1 = pd.DataFrame(List_data[0:nlabels*(nloop_1+1)], columns=flags, index=index1)
            df2 = pd.DataFrame(List_data[nlabels*(nloop_1+1):], columns=flags, index=index2)
            
            df1 = df1.rename_axis("Flags", axis="columns")
            df2 = df2.rename_axis("Flags", axis="columns")
            
            self.append([df1, df2])
            tidx = tidx + 1
            
            
        return self
    
    
    
# #-------------- Nova função: ler arquivo fort.207 --------------------#
    def fort_207_read(DIRdiag, dates):
        """ 
        Read files fort.207 from gsi. Return an list of tables with radiance data analysis. 
        The tables providing detailed statistics about the data in stages before the 1st outer loop (it=1), 
        between the 1st and 2nd outer loops (it=2), and after the 2nd outer loop (it=3).
        
        TABLE A: Summaries for various statistics as a function of observation type
        >> Columns:
           'it', 'sat', 'type', 'penalty', 'nobs', 'iland', 'isnoice', 'icoast', 'ireduce', 'ivarl', 'nlgross', 
           'qcpenalty', 'qc1', 'qc2', 'qc3', 'qc4', 'qc5', 'qc6', 'qc7'
        
        TABLE B: Summaries for various statistics as a function of channel
        >> Columns: 
           'it', 'SN satinfo', 'nchan', 'type', 'sat', 'nobsused', 'nobstossed', 'varCH', 'biasBC', 'biasAC', 'penaltyCH', 'sqrt', 'STD'
        
        TABLE C: Summary for each observation type
        >> Columns:
           'it', 'sat', 'type', 'read', 'keep', 'assim', 'penalty', 'qcpnlty', 'cpen', 'qccpen'
        
        This function return a list of the three tables.
        
        """
        
        print('DIRdiag = ',DIRdiag)
        print('')
        
        names_columnsA = ['it', 'sat', 'type', 'penalty', 'nobs', 'iland', 'isnoice', 'icoast', 'ireduce', 'ivarl', 'nlgross', 'qcpenalty', 'qc1', 'qc2', 'qc3', 'qc4', 'qc5', 'qc6', 'qc7']
        
        print('TABLE A: Names columns A = ',names_columnsA)
        print('')
        
        names_columnsB = ['it', 'SN satinfo', 'nchan', 'type', 'sat', 'nobsused', 'nobstossed', 'varCH', 'biasBC', 'biasAC', 'penaltyCH', 'sqrt', 'STD']
        
        print('TABLE B: Names columns B = ',names_columnsB)
        print('')
        
        names_columnsC = ['it', 'sat', 'type', 'read', 'keep', 'assim', 'penalty', 'qcpnlty', 'cpen', 'qccpen']
        
        print('TABLE C: Names columns C = ',names_columnsC)
        print('')
        
        pathsf, self = [], []
        [pathsf.append(DIRdiag + '/' + dt + '/fort.207') for dt in dates]
        
        print(pathsf)
        print('')
        
        tidx = 0
        for path in pathsf:
            print('Reading ' + path)
            print('')
            
            date = datetime.strptime(str(dates[tidx]), "%Y%m%d%H")
            print(date.strftime(' Preparing data for: fort.207 ---> ' + "%Y-%m-%d:%H"))
            print('')
            
            with open(path, "r") as file:
                
                lines = file.readlines()
                nlines = len(lines)
                
                print('Nº total de linhas =', nlines)
                
                List_dataA, List_dataB, List_dataC = [], [], []
                
                ll = 1
                it = ll
                #it = 'o-g 0'+str(ll)+' rad'
                
                nl = 0
                while ( nl < nlines ):
                    line = lines[nl]
                    
                    #---------------- table A -----------------#
                    if re.search('sat       type              penalty    nobs   iland isnoice  icoast ireduce   ivarl nlgross', line):
                        nl = nl + 1
                        data = []
                        data = lines[nl]
                        data = data.split()
                        data[2] = float(data[2])  # penalty is float
                        for lv in range(3,len(data),1):
                            aux = data[lv]
                            data[lv] = int(aux)
                        nl = nl + 1
                        line = lines[nl]
                        if re.search('                            qcpenalty     qc1     qc2     qc3     qc4     qc5     qc6     qc7', line):
                            nl = nl + 1
                            data2 = []
                            data2 = lines[nl]
                            data2 = data2.split()
                            data.append(float(data2[0]))  # qcpenalty is float
                            for lv in range(1,len(data2),1):
                                aux = data2[lv]
                                data.append(int(aux))
                        
                        data.insert(0,it) # inserted the stage number
                        List_dataA.append(data)
                        nl = nl + 2
                        #lineTest = lines[nl + 1]
                        #if re.search('  rad total   penalty_all=', lineTest):
                        #    ll += 1
                        #    it = ll
                        #    #it = 'o-g 0'+str(ll)+' rad'
                    
                    #---------------- table B -----------------#
                    if re.search('rad total failed nonlinqc=', line):
                        nl = nl + 1
                        Tab = True
                        while (Tab == True):
                            data = []
                            aux1 = []
                            data = lines[nl]
                            data = data.split()
                            data[0] = int(data[0])
                            data[1] = int(data[1])
                            data[3] = int(data[3])
                            data[4] = int(data[4])
                            for lv in range(5,len(data),1):
                                aux = data[lv]
                                data[lv] = (float(aux))
                            #separa a string 'type_sat' em dois elementos da lista data
                            aux1 = data[2].split('_')
                            data[2] = aux1[0]       #type
                            data.insert(3,aux1[1])  #sat
                            
                            data.insert(0,it) # inserted the stage number
                            List_dataB.append(data)
                            nl = nl + 1
                            line = lines[nl]
                            if re.search('    it      satellite instrument     # read     # keep    # assim  penalty      qcpnlty       cpen      qccpen', line):
                                Tab = False
                    
                    #---------------- table C -----------------#
                    if re.search('    it      satellite instrument     # read     # keep    # assim  penalty      qcpnlty       cpen      qccpen', line):
                        nl = nl + 1
                        Tab = True
                        while (Tab == True):
                            data = []
                            aux1 = []
                            aux1 = lines[nl]
                            aux1 = aux1.split()
                            data.append(aux1[0]+' '+aux1[1]+' '+aux1[2])
                            data.append(aux1[3])
                            data.append(aux1[4])
                            data.append(int(aux1[5]))
                            data.append(int(aux1[6]))
                            data.append(int(aux1[7]))
                            data.append(float(aux1[8]))
                            data.append(float(aux1[9]))
                            data.append(float(aux1[10]))
                            data.append(float(aux1[11]))
                            
                            List_dataC.append(data)
                            nl = nl + 1
                            if nl == nlines:
                                Tab = False
                                nl = nl - 1
                            else:
                                line = lines[nl]
                                if re.search('sat       type              penalty    nobs   iland isnoice  icoast ireduce   ivarl nlgross', line):
                                    Tab = False
                                    ll += 1
                                    it = ll
                                    #it = 'o-g 0'+str(ll)+' rad'
                                    nl = nl - 1
                                
                    # próxima linha
                    nl = nl + 1
                    
                print('nl final =', nl)
            
            file.close()
            print('')
            
            df1 = pd.DataFrame(List_dataA, columns=names_columnsA)
            df2 = pd.DataFrame(List_dataB, columns=names_columnsB)
            df3 = pd.DataFrame(List_dataC, columns=names_columnsC)
            
            self.append([df1, df2, df3])
            tidx = tidx + 1
            
        return self
    


class plot_diag(object):
    """
    plot diagnostic file from gsi. 
    """

    def plot(self, varName, varType, param, minVal=None, maxVal=None, mask=None, area=None, **kwargs):
        '''
        The plot function makes a plot for the selected observation by using information of the following columns available within the dataframe.
 
        Available columns to be used with the plot function:

        lat  : All latitudes from the selected kinds 
        lon  : All longitudes from the selected kinds
        prs  : Pressure level of the observation
        lev  : Pressure levels of the observation 
        time : Time of the observation (in minutes, relative to the analysis time)
        idqc : Quality control mark or event mark 
        iuse : Use flag (use = 1; monitoring = -1)
        iusev: Value of the flag used in the analysis
        obs  : Observation value
        
        Optional parameters: minVal and maxVal (float)
        vmin = minVal and vmax = maxVal define the data range that the colormap covers. By default (minVal=maxVal=None), the colormap covers the complete value range of the supplied data.

        Example:
        gd.plot('ps', 187, 'obs', mask='iuse == 1')
        
        In the above example, a plot will be made displaying by using the values of the used surface pressure observations of the kind 187 (ADPSFC).

        area = [Loni, Lati, Lonf, Latf]

        '''
        #
        # Parse options 
        #
        if 'style' in kwargs:
            plt.style.use(kwargs['style'])
            del kwargs['style']
        else:
            plt.style.use('seaborn-v0_8')
        
        if 'ax' not in kwargs:
            fig = plt.figure(figsize=(12, 6))
            ax  = fig.add_subplot(1, 1, 1)
        else:
            ax = kwargs['ax']
            del kwargs['ax']

        if kwargs.get('legend') is True:
            divider = make_axes_locatable(ax)
            cax     = divider.append_axes("right", size="5%", pad=0.1)
            kwargs['cax'] = cax

        if 'title' in kwargs:
            ax.set_title(kwargs['title'])

        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'jet'

        ax = geoMap(area=area,ax=ax)
        
        # try: For issues reading the file (file not found)
        # in the except statement an error message is printed and continues for other dates
        try:
            if mask is None:
                ax  = self.obsInfo[varName].loc[varType].plot(param, ax=ax, vmin=minVal, vmax=maxVal, **kwargs, legend_kwds={'shrink': 0.5})
            else:
                df = self.obsInfo[varName].loc[varType]
                ax = df.query(mask).plot(param, ax=ax, vmin=minVal, vmax=maxVal, **kwargs, legend_kwds={'shrink': 0.5})
                    
        except:
            ax = None
            print("++++++++++++++++++++++++++ ERROR: file reading --> plot ++++++++++++++++++++++++++")
            print(setcolor.WARNING + "    >>> No information on this date <<< " + setcolor.ENDC)   
            
        
        return ax

    def ptmap(self, varName, varType=None, mask=None, area=None, **kwargs):
        '''
        The ptmap function plots the selected observation for the selected kinds.

        Example:
        a.ptmap('uv', [290, 224, 223])
        
        In the above example, a plot for the wind (uv) for the kinds 290 (ASCATW), 224 (VADWND) and 223 (PROFLR) will be made.

        Note: If no kind is explicity informed, all kinds for that particular observation will be considered, which may clutter
        the plot.

        area = [Loni, Lati, Lonf, Latf]

        '''
        #
        # Parse options 
        #

        if 'style' in kwargs:
            plt.style.use(kwargs['style'])
            del kwargs['style']
        else:
            plt.style.use('seaborn-v0_8')

        if 'ax' not in kwargs:
            fig  = plt.figure(figsize=(12, 6))
            ax   = fig.add_subplot(1, 1, 1)
        else:
            ax = kwargs['ax']
            del kwargs['ax']

        if varType is None:
            varType = self.obsInfo[varName].index.levels[0].tolist()
            print('varType',varType)

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'marker' not in kwargs:
            kwargs['marker'] = '*'

        if 'markersize' not in kwargs:
            kwargs['markersize'] = 5

        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = 1

        if 'legend' not in kwargs:
            kwargs['legend'] = False
            legend = True
        else:
            legend = kwargs['legend']
            kwargs['legend'] = False
                
        
        ax = geoMap(area=area,ax=ax)

        # color range
        if type(varType) is list:
            cmin = 0
            cmax = len(varType)-1
        else:
            varType = [varType]
            cmin = 0
            cmax = 1

        legend_labels = []
        for i, kx in enumerate(varType):
            df    = self.obsInfo[varName].loc[kx]

            color = getColor(minVal=cmin, maxVal=cmax,
                             value=i,hex=True,cmapName='Paired')
            instr = getVarInfo(kx,varName,'instrument')
            
            label = '\n'.join(wrap(varName + '-' + str(kx) + ' | ' + instr,30))
            legend_labels.append(mpatches.Patch(color=color, 
                                 label=label)
                                )

            if mask is None:
               ax = df.plot(ax=ax,c=color, **kwargs)
            else:
               ax = df.query(mask).plot(ax=ax,c=color, **kwargs)
        
        if legend is True:
            plt.subplots_adjust(bottom=0.30)
            plt.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.08),
                       fancybox=False, shadow=False, frameon=False, numpoints=1, prop={"size": 9}, labelspacing=1.0, ncol=4)


        return ax

    def pvmap(self, varName=None, mask=None, area=None, **kwargs):
        '''
        The pvmap function plots the selected observations without specifying its kinds. It used the flag iuse instead. 

        Example:
        a.pvmap(['uv','ps','t','q'], mask='iuse==1')
        
        In the above example, a plot for the used (iuse=1) observations of wind (uv), surface pressure (ps), temperature (t) and moisture (q) will be made. 

        area = [Loni, Lati, Lonf, Latf]

        '''
        #
        # Parse options 
        #
        
        if 'style' in kwargs:
            plt.style.use(kwargs['style'])
            del kwargs['style']
        else:
            plt.style.use('seaborn-v0_8')
        
        if 'ax' not in kwargs:
            fig = plt.figure(figsize=(12, 6))
            ax  = fig.add_subplot(1, 1, 1)
        else:
            ax = kwargs['ax']
            del kwargs['ax']

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'marker' not in kwargs:
            kwargs['marker'] = '*'

        if 'markersize' not in kwargs:
            kwargs['markersize'] = 5

        if 'linewidth' not in kwargs:
            kwargs['linewidth'] = 1

        if 'legend' not in kwargs:
            kwargs['legend'] = False
            legend = True
        else:
            legend = kwargs['legend']
            kwargs['legend'] = False

        #
        # total by var
        #
        
        total = self.obs.groupby(level=0).size()

        #
        # parse options em kwargs

        if varName is None:
            varName = total.sort_values(ascending=False).keys()
        else:
            if type(varName) is list:
               varName = total[varName].sort_values(ascending=False).keys()
            else:
                varName = [varName]
        
        ax = geoMap(area=area,ax=ax)

        
        colors_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
        setColor = 0
        legend_labels = []
        for var in varName:
            df    = self.obsInfo[var]
            legend_labels.append(mpatches.Patch(color=colors_palette[setColor], label=var) )

            if mask is None:
               ax = df.plot(ax=ax,c=colors_palette[setColor], **kwargs)
            else:
               ax = df.query(mask).plot(ax=ax,c=colors_palette[setColor], **kwargs)
            setColor += 1

        if legend is True:
            plt.legend(handles=legend_labels, numpoints=1, loc='best', bbox_to_anchor=(1.1, 0.6), 
                       fancybox=False, shadow=False, frameon=False, ncol=1, prop={"size": 10})


        return ax

    def pcount(self,varName,**kwargs):

        """
        Plots a histogram of the desired variable and types.

        Usage: pcount(VarName)
        """

        try:
           import matplotlib.pyplot as plt
           import matplotlib.cm as cm
           from matplotlib.colors import Normalize

        except ImportError:
           pass # module doesn't exist, deal with it.
        #
        # Parse options 
        #

        if 'style' in kwargs:
            plt.style.use(kwargs['style'])
            del kwargs['style']
        else:
            plt.style.use('seaborn-v0_8')

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'rot' not in kwargs:
            kwargs['rot'] = 45

        if 'legend' not in kwargs:
            kwargs['legend'] = False


        df = self.obsInfo[varName].groupby(level=0).size()

        # Get a color map
        colors = getColor(minVal=df.min(),maxVal=df.max(),
                          value=df.values,hex=True,cmapName='Paired')

        df.plot.bar(color=colors,**kwargs)

        plt.ylabel('Number of Observations')
        plt.xlabel('KX')
        plt.title('Variable Name : '+varName)
 
    def vcount(self,**kwargs):

        """
        Plots a histogram of the total count of eath variable.

        Usage: pcount(**kwargs)
        """

        try:
           import matplotlib.pyplot as plt
           import matplotlib.cm as cm
           from matplotlib.colors import Normalize

        except ImportError:
           pass # module doesn't exist, deal with it.
        #
        # Parse options 
        #

        if 'style' in kwargs:
            plt.style.use(kwargs['style'])
            del kwargs['style']
        else:
            plt.style.use('seaborn-v0_8')

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'rot' not in kwargs:
            kwargs['rot'] = 0

        if 'legend' not in kwargs:
            kwargs['legend'] = False

        df = pd.DataFrame({key: len(value) for key, value in self.obsInfo.items()},index=['total']).T

        # Get a color map
        colors = getColor(minVal=df.min(),maxVal=df.max(),
                          value=df['total'].values,hex=True,cmapName='Paired')
         
        df.plot.bar(color=colors, **kwargs)

        plt.ylabel('Number of Observations')
        plt.xlabel('Variable Names')
        plt.title('Total Number of Observations')
 
    def kxcount(self,**kwargs):

        """
        Plots a histogram of the total count by KX.

        Usage: pcount(**kwargs)
        """

        try:
           import matplotlib.pyplot as plt
           from matplotlib.colors import Normalize

        except ImportError:
           pass # module doesn't exist, deal with it.
        #
        # Parse options 
        #

        if 'style' in kwargs:
            plt.style.use(kwargs['style'])
            del kwargs['style']
        else:
            plt.style.use('seaborn-v0_8')

        if 'alpha' not in kwargs:
            kwargs['alpha'] = 0.5

        if 'rot' not in kwargs:
            kwargs['rot'] = 90

        if 'legend' not in kwargs:
            kwargs['legend'] = False

        d  = pd.concat(self.obsInfo, sort=False).reset_index(level=2, drop=True)
        df = d.groupby(['kx']).size()

        # Get a color map
        colors = getColor(minVal=df.min(),maxVal=df.max(),
                          value=df.values,hex=True,cmapName='Paired')
          
        plt.style.use('seaborn-v0_8')
        df.plot.bar(color=colors, **kwargs)

        plt.ylabel('Number of Observations by KX')
        plt.xlabel('KX number')
        plt.title('Total Number of Observations')
 
    def time_series(self, varName=None, varType=None, mask=None, dateIni=None, dateFin=None, nHour="06", vminOMA=None, vmaxOMA=None, vminSTD=0.0, vmaxSTD=14.0, Level=None, Lay = None, SingleL=None, Clean=None):
        
        '''
        The time_series function plots a time series for different levels/layers or for a single level/layer considering
        OmF and OmA. 

        Example:

        vName = 'uv'          # Variable
        vType = 224           # Source Type
        mask  = None          # Mask the data by chosen used/not used data, ex: mask='iuse==1'
        dateIni = 2013010100  # Inicial Date
        dateFin = 2013010900  # Final Date
        nHour = "06"          # Time Interval
        vminOMA = -4.0        # Y-axis Minimum Value for OmF or OmA
        vmaxOMA = 4.0         # Y-axis Maximum Value for OmF or OmA
        vminSTD = 0.0         # Y-axis Minimum Value for Standard Deviation
        vmaxSTD = 14.0        # Y-axis Maximum Value for Standard Deviation
        Level = 1000          # Time Series Level, if any (None), all standard levels are plotted
        Lay = 15              # The size of half layer in hPa, if the plot type is sampled by layers.
        SingleL = "OneL"      # When level is fixed, ex: 1000 hPa, the plot can be exactly in this level (SingleL = None),
                              # on all levels as a single layer (SingleL = "All") or on a layer centered in Level and bounded by
                              # Level-Lay and Level+Lay (SingleL="OneL"). If Lay is not defined, it will be used a standard value of 50 hPa. 

        '''
        if Clean == None:
            Clean = True

        delta = nHour
        omflag = "OmF"
        omflaga = "OmA"

        Laydef = 50

        separator = " ====================================================================================================="

        print()
        print(separator)
        #print(" Reading dataset in " + data_path)
        varInfo = getVarInfo(varType, varName, 'instrument')
        if varInfo is not None:
            print(" Analyzing data of variable: " + varName + "  ||  type: " + str(varType) + "  ||  " + varInfo + "  ||  check: " + omflag)
        else:
            print(" Analyzing data of variable: " + varName + "  ||  type: " + str(varType) + "  ||  Unknown instrument  ||  check: " + omflag)

        print(separator)
        print()

        if mask == None:
            maski  = "iuse>-99999.9"
            cmaski = "iuse = All"
        else:
            maski  = mask
            cmaski = mask

        if type(Level) == list:
            zlevs_def = Level
            Level = "Zlevs"
        else:
            zlevs_def = list(map(int,self[0].zlevs))

        print(zlevs_def)

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        levs_tmp, DayHour_tmp = [], []
        info_check = {}
        f = 0
        while (date <= datef):
            
            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            dataDict = self[f].obsInfo[varName].query(maski).loc[varType]
            info_check.update({date.strftime("%d%H"):True})

            if 'prs' in dataDict and (Level == None or Level == "Zlevs"):
                if(Level == None):
                    levs_tmp.extend(list(set(map(int,dataDict['prs']))))
                else:
                    levs_tmp = zlevs_def[::-1]
                info_check.update({date.strftime("%d%H"):True})
                print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"))
                print(' Levels: ', sorted(levs_tmp), end='\n')
                print("")
                f = f + 1
            else:
                if (Level != None and Level != "Zlevs") and info_check[date.strftime("%d%H")] == True:
                    levs_tmp.extend([Level])
                    print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"), ' - Level: ', Level , end='\n')
                    f = f + 1
                else:
                    info_check.update({date.strftime("%d%H"):False})
                    print(date.strftime(setcolor.WARNING + ' Preparing data for: ' + "%Y-%m-%d:%H"), ' - No information on this date ' + setcolor.ENDC, end='\n')

            del(dataDict)
            
            date = date + timedelta(hours=int(delta))
            
        if(len(DayHour_tmp) > 4):
            DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
        else:
            DayHour = DayHour_tmp

        zlevs = [z if z in zlevs_def else "" for z in sorted(set(levs_tmp+zlevs_def))]

        print()
        print(separator)
        print()

        list_meanByLevs, list_stdByLevs, list_countByLevs = [], [], []
        list_meanByLevsa, list_stdByLevsa, list_countByLevsa = [], [], []
        date = datei
        levs = sorted(list(set(levs_tmp)))
        levs_tmp.clear()
        del(levs_tmp[:])

        f = 0
        while (date <= datef):

            print(date.strftime(' Calculating for ' + "%Y-%m-%d:%H"))
            datefmt = date.strftime("%Y%m%d%H")

            try: 
                if info_check[date.strftime("%d%H")] == True:
                    dataDict = self[f].obsInfo[varName].query(maski).loc[varType]
                    dataByLevs, mean_dataByLevs, std_dataByLevs, count_dataByLevs = {}, {}, {}, {}
                    dataByLevsa, mean_dataByLevsa, std_dataByLevsa, count_dataByLevsa = {}, {}, {}, {}
                    [dataByLevs.update({int(lvl): []}) for lvl in levs]
                    [dataByLevsa.update({int(lvl): []}) for lvl in levs]
                    if Level != None and Level != "Zlevs":
                        if SingleL == None:
                            [ dataByLevs[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) == Level ]
                            [ dataByLevsa[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) == Level ]
                            forplot = ' Level='+str(Level) +'hPa'
                            forplotname = 'level_'+str(Level) +'hPa'
                        else:
                            if SingleL == "All":
                                [ dataByLevs[Level].append(v) for v in self[f].obsInfo[varName].query(maski).loc[varType].omf ]
                                [ dataByLevsa[Level].append(v) for v in self[f].obsInfo[varName].query(maski).loc[varType].oma ]
                                forplot = ' Layer=Entire Atmosphere'
                                forplotname = 'layer_allAtm'
                            else:
                                if SingleL == "OneL":
                                    if Lay == None:
                                        print("")
                                        print(" Variable Lay is None, resetting it to its default value: "+str(Laydef)+" hPa.")
                                        print("")
                                        Lay = Laydef
                                    [ dataByLevs[int(Level)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) >=Level-Lay and int(p) <Level+Lay ]
                                    [ dataByLevsa[int(Level)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) >=Level-Lay and int(p) <Level+Lay ]
                                    forplot = ' Layer='+str(Level+Lay)+'-'+str(Level-Lay)+'hPa'
                                    forplotname = 'layer_'+str(Level+Lay)+'-'+str(Level-Lay)+'hPa'
                                else:
                                    print(" Wrong value for variable SingleL. Please, check it and rerun the script.")    
                    else:
                        if Level == None:
                            [ dataByLevs[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) ]
                            [ dataByLevsa[int(p)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) ]
                            forplotname = 'all_levels_byLevels'
                        else:
                            for ll in range(len(levs)):
                                lv = levs[ll]
                                if Lay == None:
                                    if ll == 0:
                                        Llayi = 0
                                    else:
                                        Llayi = (levs[ll] - levs[ll-1]) / 2.0
                                    if ll == len(levs)-1:
                                        Llayf = Llayi
                                    else:
                                        Llayf = (levs[ll+1] - levs[ll]) / 2.0
                                    cutlevs = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) >=lv-Llayi and int(p) <lv+Llayf ]
                                    cutlevsa = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) >=lv-Llayi and int(p) <lv+Llayf ]
                                    forplotname = 'all_levels_filledLayers'
                                else:
                                    cutlevs = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) >=lv-Lay and int(p) <lv+Lay ]
                                    cutlevsa = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].prs,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) >=lv-Lay and int(p) <lv+Lay ]
                                    forplotname = 'all_levels_bylayers_'+str(Lay)+"hPa"
                                [ dataByLevs[lv].append(il) for il in cutlevs ]
                                [ dataByLevsa[lv].append(il) for il in cutlevsa ]
                    f = f + 1
                for lv in levs:
                    if len(dataByLevs[lv]) != 0 and info_check[date.strftime("%d%H")] == True:
                        mean_dataByLevs.update({int(lv): np.mean(np.array(dataByLevs[lv]))})
                        std_dataByLevs.update({int(lv): np.std(np.array(dataByLevs[lv]))})
                        count_dataByLevs.update({int(lv): len(np.array(dataByLevs[lv]))})
                        mean_dataByLevsa.update({int(lv): np.mean(np.array(dataByLevsa[lv]))})
                        std_dataByLevsa.update({int(lv): np.std(np.array(dataByLevsa[lv]))})
                        count_dataByLevsa.update({int(lv): len(np.array(dataByLevsa[lv]))})
                    else:
                        mean_dataByLevs.update({int(lv): -99})
                        std_dataByLevs.update({int(lv): -99})
                        count_dataByLevs.update({int(lv): -99})
                        mean_dataByLevsa.update({int(lv): -99})
                        std_dataByLevsa.update({int(lv): -99})
                        count_dataByLevsa.update({int(lv): -99})
            
            except:
                if info_check[date.strftime("%d%H")] == True:
                    print("ERROR in time_series function.")
                else:
                    print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)

                for lv in levs:
                    mean_dataByLevs.update({int(lv): -99})
                    std_dataByLevs.update({int(lv): -99})
                    count_dataByLevs.update({int(lv): -99})
                    mean_dataByLevsa.update({int(lv): -99})
                    std_dataByLevsa.update({int(lv): -99})
                    count_dataByLevsa.update({int(lv): -99})

            if Level == None or Level == "Zlevs":
                list_meanByLevs.append(list(mean_dataByLevs.values()))
                list_stdByLevs.append(list(std_dataByLevs.values()))
                list_countByLevs.append(list(count_dataByLevs.values()))
                list_meanByLevsa.append(list(mean_dataByLevsa.values()))
                list_stdByLevsa.append(list(std_dataByLevsa.values()))
                list_countByLevsa.append(list(count_dataByLevsa.values()))
            else:
                list_meanByLevs.append(mean_dataByLevs[int(Level)])
                list_stdByLevs.append(std_dataByLevs[int(Level)])
                list_countByLevs.append(count_dataByLevs[int(Level)])
                list_meanByLevsa.append(mean_dataByLevsa[int(Level)])
                list_stdByLevsa.append(std_dataByLevsa[int(Level)])
                list_countByLevsa.append(count_dataByLevsa[int(Level)])

            dataByLevs.clear()
            mean_dataByLevs.clear()
            std_dataByLevs.clear()
            count_dataByLevs.clear()
            dataByLevsa.clear()
            mean_dataByLevsa.clear()
            std_dataByLevsa.clear()
            count_dataByLevsa.clear()

            date_finale = date
            date = date + timedelta(hours=int(delta))

        print()
        print(separator)
        print()

        print(' Making Graphics...')

        y_axis      = np.arange(0, len(zlevs), 1)
        x_axis      = np.arange(0, len(DayHour), 1)

        mean_final  = np.ma.masked_array(np.array(list_meanByLevs), np.array(list_meanByLevs) == -99)
        std_final   = np.ma.masked_array(np.array(list_stdByLevs), np.array(list_stdByLevs) == -99)
        count_final = np.ma.masked_array(np.array(list_countByLevs), np.array(list_countByLevs) == -99)
        mean_finala  = np.ma.masked_array(np.array(list_meanByLevsa), np.array(list_meanByLevsa) == -99)
        std_finala   = np.ma.masked_array(np.array(list_stdByLevsa), np.array(list_stdByLevsa) == -99)
        count_finala = np.ma.masked_array(np.array(list_countByLevsa), np.array(list_countByLevsa) == -99)

        OMF_inf = np.array(list_meanByLevs)-np.array(list_stdByLevs)
        OMF_sup = np.array(list_meanByLevs)+np.array(list_stdByLevs)
        OMA_inf = np.array(list_meanByLevsa)-np.array(list_stdByLevsa)
        OMA_sup = np.array(list_meanByLevsa)+np.array(list_stdByLevsa)

        mean_limit_inf = np.min(np.array([np.min(mean_final), np.min(mean_finala)]))
        mean_limit_sup = np.max(np.array([np.max(mean_final), np.max(mean_finala)]))

        std_limit_inf = np.min(np.array([np.min(std_final), np.min(std_finala)]))
        std_limit_sup = np.max(np.array([np.max(std_final), np.max(std_finala)]))

        omfoma_limit_inf =     (np.min(np.array([np.min(OMF_inf), np.min(OMA_inf)])))
        if omfoma_limit_inf > 0:
            omfoma_limit_inf = 0.9*omfoma_limit_inf
        else:
            omfoma_limit_inf = 1.1*omfoma_limit_inf  
        omfoma_limit_sup = 1.1*(np.max(np.array([np.max(OMF_sup), np.max(OMA_sup)])))

        if (vminOMA == None) and (vmaxOMA == None): vminOMA, vmaxOMA = mean_limit_inf, 1.1*mean_limit_sup
        if vminOMA > 0:
            vminOMA = 0.9*vminOMA
        else:
            vminOMA = 1.1*vminOMA 

        vmaxOMAabs = np.max([np.abs(vminOMA),np.abs(vminOMA)])

        if (vminSTD == None) and (vmaxSTD == None): vminSTD, vmaxSTD = std_limit_inf - 0.1*std_limit_inf,  1.1*std_limit_sup

        date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
        instrument_title = str(varName) + '-' + str(varType) + '  |  ' + getVarInfo(varType, varName, 'instrument')

        # Figure with more than one level - default levels: [600, 700, 800, 900, 1000]
        if Level == None or Level == "Zlevs":
            fig = plt.figure(figsize=(6, 9))
            plt.rcParams['axes.facecolor'] = 'None'
            plt.rcParams['hatch.linewidth'] = 0.3

            ##### OMF

            plt.subplot(3, 1, 1)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(mean_final.T), origin='lower', vmin=-vmaxOMAabs, vmax=vmaxOMAabs, cmap='seismic', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Mean ('+omflag+')', labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 2)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(std_final.T), origin='lower', vmin=vminSTD, vmax=vmaxSTD, cmap='Blues', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Standard Deviation ('+omflag+')', labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 3)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(count_final.T), origin='lower', vmin=0.0, vmax=np.max(count_final), cmap='gist_heat_r', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Total Observations'+" ("+cmaski+")", labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflag+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMA

            fig = plt.figure(figsize=(6, 9))
            plt.rcParams['axes.facecolor'] = 'None'
            plt.rcParams['hatch.linewidth'] = 0.3

            plt.subplot(3, 1, 1)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(mean_finala.T), origin='lower', vmin=-vmaxOMAabs, vmax=vmaxOMAabs, cmap='seismic', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Mean ('+omflaga+')', labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 2)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(std_finala.T), origin='lower', vmin=vminSTD, vmax=vmaxSTD, cmap='Blues', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Standard Deviation ('+omflaga+')', labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 3)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(count_finala.T), origin='lower', vmin=0.0, vmax=np.max(count_finala), cmap='gist_heat_r', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Vertical Levels (hPa)')
            plt.xlabel('Total Observations'+" ("+cmaski+")", labelpad=50)
            plt.yticks(y_axis, zlevs[::-1])
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflaga+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

        # Figure with only one level
        else:
        
            ##### OMF

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            ax1.plot(x_axis, list_meanByLevs, "b-", label="Mean ("+omflag+")")
            ax1.plot(x_axis, list_meanByLevs, "bo", label="Mean ("+omflag+")")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.set_ylabel('Mean ('+omflag+')', color='b', fontsize=10)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevs),ls='dotted',c='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(x_axis, std_final, "r-", label="Std. Deviation ("+omflag+")")
            ax2.plot(x_axis, std_final, "rs", label="Std. Deviation ("+omflag+")")
            ax2.set_ylim(vminSTD, vmaxSTD)
            ax2.set_ylabel('Std. Deviation ('+omflag+')', color='r', fontsize=10)
            ax2.tick_params('y', colors='r')
            major_ticks = np.arange(0, max(x_axis), len(DayHour)/len(list(filter(None, DayHour))))
            ax2.set_xticks(major_ticks)
            plt.axhline(y=np.mean(std_final),ls='dotted',c='red')

            ax3 = ax1.twinx()
            ax3.plot(x_axis, count_final, "g-", label="Total Observations"+" ("+cmaski+")")
            ax3.plot(x_axis, count_final, "g^", label="Total Observations"+" ("+cmaski+")")
            ax3.set_ylim(0, np.max(count_final) + (np.max(count_final)/8))
            ax3.set_ylabel('Total Observations'+" ("+cmaski+")", color='g', fontsize=10)
            ax3.tick_params('y', colors='g')
            ax3.spines["right"].set_position(("axes", 1.15))
            plt.yticks(rotation=90)
            plt.axhline(y=np.mean(count_final),ls='dotted',c='green')

            ax3.set_title(instrument_title, loc='left', fontsize=10)
            ax3.set_title(date_title, loc='right', fontsize=10)

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax3.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflag+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMA

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            ax1.plot(x_axis, list_meanByLevsa, "b-", label="Mean ("+omflaga+")")
            ax1.plot(x_axis, list_meanByLevsa, "bo", label="Mean ("+omflaga+")")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.set_ylabel('Mean ('+omflaga+')', color='b', fontsize=10)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevsa),ls='dotted',c='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(x_axis, std_finala, "r-", label="Std. Deviation ("+omflaga+")")
            ax2.plot(x_axis, std_finala, "rs", label="Std. Deviation ("+omflaga+")")
            ax2.set_ylim(vminSTD, vmaxSTD)
            ax2.set_ylabel('Std. Deviation ('+omflaga+')', color='r', fontsize=10)
            ax2.tick_params('y', colors='r')
            plt.axhline(y=np.mean(std_finala),ls='dotted',c='red')

            ax3 = ax1.twinx()
            ax3.plot(x_axis, count_finala, "g-", label="Total Observations"+" ("+cmaski+")")
            ax3.plot(x_axis, count_finala, "g^", label="Total Observations"+" ("+cmaski+")")
            ax3.set_ylim(0, 1.2*np.max(count_finala))
            ax3.set_ylabel('Total Observations'+" ("+cmaski+")", color='g', fontsize=10)
            ax3.tick_params('y', colors='g')
            ax3.spines["right"].set_position(("axes", 1.15))
            plt.yticks(rotation=90)
            plt.axhline(y=np.mean(count_finala),ls='dotted',c='green')

            ax3.set_title(instrument_title, loc='left', fontsize=10)
            ax3.set_title(date_title, loc='right', fontsize=10)

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax3.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflaga+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMF and OMA

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            ax1.plot(x_axis, list_meanByLevs, "b-", label="Mean ("+omflag+")")
            ax1.plot(x_axis, list_meanByLevs, "bo", label="")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevs),ls='dotted',c='blue')
            
            ax1.plot(x_axis, list_meanByLevsa, "r-", label="Mean ("+omflaga+")")
            ax1.plot(x_axis, list_meanByLevsa, "rs", label="")
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.tick_params('y', colors='black')
            plt.axhline(y=np.mean(list_meanByLevsa),ls='dotted',c='red')

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)

            ybox1 = TextArea('Mean ('+omflag+')' , textprops=dict(color="b", size=12,rotation=90,ha='left',va='bottom'))
            ybox2 = TextArea(' and '             , textprops=dict(color="k", size=12,rotation=90,ha='left',va='bottom'))
            ybox3 = TextArea('Mean ('+omflaga+')', textprops=dict(color="r", size=12,rotation=90,ha='left',va='bottom'))

            ybox = VPacker(children=[ybox3, ybox2, ybox1],align="bottom", pad=0, sep=5)

            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.12, 0.16), 
                                                bbox_transform=ax1.transAxes, borderpad=0.)

            ax1.add_artist(anchored_ybox)
            plt.legend()

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_OmFOmA_'+ forplotname +'.png', bbox_inches='tight', dpi=100)

            # OMF and OMA and StdDev

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')
            
            ax1.plot(x_axis, list_meanByLevs, lw=2, label='OmF Mean', color='blue', zorder=1)
            ax1.fill_between(x_axis, OMF_inf, OMF_sup, label='OmF Std Dev',  facecolor='blue', alpha=0.3, zorder=1)
            ax1.plot(x_axis, list_meanByLevsa, lw=2, label='OmA Mean', color='red', zorder=2)
            ax1.fill_between(x_axis, OMA_inf, OMA_sup, label='OmA Std Dev',  facecolor='red', alpha=0.3, zorder=2)
            ybox1 = TextArea(' OmF ' , textprops=dict(color="b", size=12,rotation=90,ha='left',va='bottom'))
            ybox2 = TextArea(' | '             , textprops=dict(color="k", size=12,rotation=90,ha='left',va='bottom'))
            ybox3 = TextArea(' OmA ', textprops=dict(color="r", size=12,rotation=90,ha='left',va='bottom'))

            ybox = VPacker(children=[ybox3, ybox2, ybox1],align="bottom", pad=0, sep=5)

            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.125, 0.42), 
                                                bbox_transform=ax1.transAxes, borderpad=0.)

            ax1.add_artist(anchored_ybox)
            ax1.set_xlabel('Date (DayHour)', fontsize=12)
            ax1.set_ylim(omfoma_limit_inf,omfoma_limit_sup)
            ax1.legend(bbox_to_anchor=(-0.11, -0.25),ncol=4,loc='lower left', fancybox=True, shadow=False, frameon=True, framealpha=1.0, fontsize='11', facecolor='white', edgecolor='lightgray')
            plt.grid(axis='y', color='lightgray', linestyle='-.', linewidth=0.5, zorder=0)

            ax2 = ax1.twinx()
            ax2.plot(x_axis, list_countByLevsa, lw=2, label='OmA', linestyle='--', color='green', zorder=3)
            ax2.plot(x_axis, list_countByLevs, lw=2, label='OmF', linestyle=':', color='purple', zorder=3)
            ax2.set_ylabel('Total Observations (OmF | OmA)'+"\n ("+cmaski+")", fontsize=12)
            ax2.set_ylim(0, (np.max(list_countByLevsa) + np.max(list_countByLevsa)/5))
            ax2.legend(loc='upper left', ncol=2, fancybox=True, shadow=False, frameon=True, framealpha=1.0, fontsize='11', facecolor='white', edgecolor='lightgray')
            
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax2.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
        
            t = plt.annotate(forplot, xy=(0.78, 0.995), xytext=(-9, -9), xycoords='axes fraction', textcoords='offset points', color='darkgray', fontweight='bold', fontsize='10',
                                horizontalalignment='center', verticalalignment='center')
            t.set_bbox(dict(facecolor='whitesmoke', alpha=1.0, edgecolor='whitesmoke', boxstyle="square,pad=0.3"))

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_OmFOmA_StdDev_'+ forplotname +'.png', bbox_inches='tight', dpi=100)

        # Cleaning up
        if Clean:
            plt.close('all')

        print(' Done!')
        print()
        
               

        return
        
# radiance inicio

    def time_series_radi(self, varName=None, varType=None, mask=None, dateIni=None, dateFin=None, nHour="06", vminOMA=None, vmaxOMA=None, vminSTD=0.0, vmaxSTD=14.0, channel=None, Clean=None):
        
        '''
        The time_series_radi function plots a time series for radiance data in different chanell OmF and OmA. This function is different from time_series because the level are not defined by radiance dada.

        Example:

        vName = 'amsua'       # Radiance Sensor 
        vType = 'n15'         # Source Type satellite
        mask  = None          # Mask the data by chosen used/not used data, ex: mask='iuse==1 & idqc==0'
        dateIni = 2013010100  # Inicial Date
        dateFin = 2013010900  # Final Date
        nHour = "06"          # Time Interval
        vminOMA = -4.0        # Y-axis Minimum Value for OmF or OmA
        vmaxOMA = 4.0         # Y-axis Maximum Value for OmF or OmA
        vminSTD = 0.0         # Y-axis Minimum Value for Standard Deviation
        vmaxSTD = 14.0        # Y-axis Maximum Value for Standard Deviation
        channel = 1           # Time Series channel, if any (None), all nchan are plotted
        
        '''
        if Clean == None:
            Clean = True

        delta = nHour
        omflag = "OmF"
        omflaga = "OmA"

        separator = " ============================================================================================================="

        print()
        print(separator)
        #print(" Reading dataset in " + data_path)
        varInfo = getVarInfo(varType, varName, 'instrument')
        if varInfo is not None:
            print(" Variable: " + varName + "  ||  type: " + str(varType) + "  ||  " + varInfo + "  ||  check: " + omflag)
        else:
            print(" Variable: " + varName + "  ||  type: " + str(varType) + "  ||  Unknown instrument  ||  check: " + omflag)

        print(separator)
        print()
        
        if varName == 'amsua':
            zchans_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] #list of all channels of the amsua sensor
        elif varName == 'hirs4':
            zchans_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] #list of all channels of the hirs/4 sensor
        else:
            zchans_all = list(map(int,self[0].obsInfo[varName].loc[varType].nchan.unique()))

        if mask == None:
            maski  = "iuse>-99999.9"
            cmaski = "iuse = All"
        else:
            maski  = mask
            cmaski = mask

        if type(channel) == list:
            zchan = channel
            chanList = 1
            zchans_def = zchan
        elif channel == None:
            zchan = zchans_all       #list of all channels of the sensor 
            chanList = 0
            zchans_def = zchan
        else:
            zchan = channel
            chanList = 0 
            zchans_def = zchans_all  #list of all channels of the sensor 

            
#         print(zchan,chanList)

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        levs_tmp, DayHour_tmp = [], []
        info_check = {}
        f = 0
        
        while (date <= datef):
            
            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            # Try: For issues reading the file (file not found), 
            # in the except statement an error message is printed and continues for other dates
            try:
                dataDict = self[f].obsInfo[varName].query(maski).loc[varType]
                info_check.update({date.strftime("%d%H"):True})

                if 'nchan' in dataDict and (channel == None or chanList == 1):
                    levs_tmp = zchans_def[::-1]
                    info_check.update({date.strftime("%d%H"):True})
                    print(date.strftime(' Preparing data for: Canais de radiancia' + "%Y-%m-%d:%H"))
                    print(' Channels: ', sorted(levs_tmp), end='\n')
                    print("")
                    f = f + 1
                else:
                    if (channel != None and chanList != 1) and info_check[date.strftime("%d%H")] == True:
                        levs_tmp.extend([zchan])
                        info_check.update({date.strftime("%d%H"):True})
                        print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"), ' - Channel de radiancia: ', zchan , end='\n')
                        f = f + 1
                    else:
                        info_check.update({date.strftime("%d%H"):False})
                        print(date.strftime(setcolor.WARNING + ' Preparing data for: ' + "%Y-%m-%d:%H"), ' - No information on this date ' + setcolor.ENDC, end='\n')
                
                del(dataDict)
                
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> time_series_radi ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                info_check.update({date.strftime("%d%H"):False})
                print(" Update info --> info_check[date.strftime(%d%H)]",info_check[date.strftime("%d%H")])
                print("")
                f = f + 1
            
            date = date + timedelta(hours=int(delta))
            
        if(len(DayHour_tmp) > 4):
            DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
        else:
            DayHour = DayHour_tmp
        
        zlevs = [z if z in zchans_def else "" for z in sorted(set(levs_tmp+zchans_def))]

        print()
        print(separator)
        print()

        list_meanByLevs, list_stdByLevs, list_countByLevs = [], [], []
        list_meanByLevsa, list_stdByLevsa, list_countByLevsa = [], [], []
        date = datei
        levs = sorted(list(set(levs_tmp)))
        levs_tmp.clear()
        del(levs_tmp[:])
        
        print('channels = ',levs)

        
        f = 0
        while (date <= datef):

            print(date.strftime(' Calculating for ' + "%Y-%m-%d:%H"))
            datefmt = date.strftime("%Y%m%d%H")


            try: 
                if info_check[date.strftime("%d%H")] == True:
                    dataDict = self[f].obsInfo[varName].query(maski).loc[varType]
                    dataByLevs, mean_dataByLevs, std_dataByLevs, count_dataByLevs = {}, {}, {}, {}
                    dataByLevsa, mean_dataByLevsa, std_dataByLevsa, count_dataByLevsa = {}, {}, {}, {}
                    [dataByLevs.update({int(lvl): []}) for lvl in levs]
                    [dataByLevsa.update({int(lvl): []}) for lvl in levs]
                    if channel != None and chanList != 1: 
                        forplot = 'Channel ='+str(zchan)
                        forplotname = 'Channel_'+str(zchan)
                        [ dataByLevs[int(zchan)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].nchan,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) == zchan ]
                        [ dataByLevsa[int(zchan)].append(v) for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].nchan,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) == zchan ]   
                    else:
                        for ll in range(len(levs)):
                            lv = levs[ll]
                            cutlevs = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].nchan,self[f].obsInfo[varName].query(maski).loc[varType].omf) if int(p) == lv ]
                            cutlevsa = [ v for p,v in zip(self[f].obsInfo[varName].query(maski).loc[varType].nchan,self[f].obsInfo[varName].query(maski).loc[varType].oma) if int(p) == lv ]
                            forplotname = 'List_Channel'
                            [ dataByLevs[lv].append(il) for il in cutlevs ]
                            [ dataByLevsa[lv].append(il) for il in cutlevsa ]
                            cutlevs.clear()
                            cutlevsa.clear()
                    f = f + 1
                for lv in levs:
                    if len(dataByLevs[lv]) != 0 and info_check[date.strftime("%d%H")] == True:
                        mean_dataByLevs.update({int(lv): np.mean(np.array(dataByLevs[lv]))})
                        std_dataByLevs.update({int(lv): np.std(np.array(dataByLevs[lv]))})
                        count_dataByLevs.update({int(lv): len(np.array(dataByLevs[lv]))})
                        mean_dataByLevsa.update({int(lv): np.mean(np.array(dataByLevsa[lv]))})
                        std_dataByLevsa.update({int(lv): np.std(np.array(dataByLevsa[lv]))})
                        count_dataByLevsa.update({int(lv): len(np.array(dataByLevsa[lv]))})
                    else:
                        mean_dataByLevs.update({int(lv): -99})
                        std_dataByLevs.update({int(lv): -99})
                        count_dataByLevs.update({int(lv): -99})
                        mean_dataByLevsa.update({int(lv): -99})
                        std_dataByLevsa.update({int(lv): -99})
                        count_dataByLevsa.update({int(lv): -99})
            
            except:
                dataByLevs, mean_dataByLevs, std_dataByLevs, count_dataByLevs = {}, {}, {}, {}
                dataByLevsa, mean_dataByLevsa, std_dataByLevsa, count_dataByLevsa = {}, {}, {}, {}
                if info_check[date.strftime("%d%H")] == True:
                    print("ERROR in time_series function.")
                else:
                    f = f + 1 # Estava faltando: sem essa atualização o dataDict do próximo UTC não é concatenado corretamente
                    print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)

                for lv in levs:
                    mean_dataByLevs.update({int(lv): -99})
                    std_dataByLevs.update({int(lv): -99})
                    count_dataByLevs.update({int(lv): -99})
                    mean_dataByLevsa.update({int(lv): -99})
                    std_dataByLevsa.update({int(lv): -99})
                    count_dataByLevsa.update({int(lv): -99})

            
            if channel == None or chanList == 1:
                list_meanByLevs.append(list(reversed(mean_dataByLevs.values())))
                list_stdByLevs.append(list(reversed(std_dataByLevs.values())))
                list_countByLevs.append(list(reversed(count_dataByLevs.values())))
                list_meanByLevsa.append(list(reversed(mean_dataByLevsa.values())))
                list_stdByLevsa.append(list(reversed(std_dataByLevsa.values())))
                list_countByLevsa.append(list(reversed(count_dataByLevsa.values())))
            else:
                list_meanByLevs.append(mean_dataByLevs[int(zchan)])
                list_stdByLevs.append(std_dataByLevs[int(zchan)])
                list_countByLevs.append(count_dataByLevs[int(zchan)])
                list_meanByLevsa.append(mean_dataByLevsa[int(zchan)])
                list_stdByLevsa.append(std_dataByLevsa[int(zchan)])
                list_countByLevsa.append(count_dataByLevsa[int(zchan)])

            dataByLevs.clear()
            mean_dataByLevs.clear()
            std_dataByLevs.clear()
            count_dataByLevs.clear()
            dataByLevsa.clear()
            mean_dataByLevsa.clear()
            std_dataByLevsa.clear()
            count_dataByLevsa.clear()

            date_finale = date
            date = date + timedelta(hours=int(delta))

        
        print()
        print(separator)
        print()

        print(' Making Graphics...')

        y_axis      = np.arange(0, len(zlevs), 1)
        x_axis      = np.arange(0, len(DayHour), 1)

        mean_final  = np.ma.masked_array(np.array(list_meanByLevs), np.array(list_meanByLevs) == -99)
        std_final   = np.ma.masked_array(np.array(list_stdByLevs), np.array(list_stdByLevs) == -99)
        count_final = np.ma.masked_array(np.array(list_countByLevs), np.array(list_countByLevs) == -99)
        mean_finala  = np.ma.masked_array(np.array(list_meanByLevsa), np.array(list_meanByLevsa) == -99)
        std_finala   = np.ma.masked_array(np.array(list_stdByLevsa), np.array(list_stdByLevsa) == -99)
        count_finala = np.ma.masked_array(np.array(list_countByLevsa), np.array(list_countByLevsa) == -99)

        OMF_inf = np.array(list_meanByLevs)-np.array(list_stdByLevs)
        OMF_sup = np.array(list_meanByLevs)+np.array(list_stdByLevs)
        OMA_inf = np.array(list_meanByLevsa)-np.array(list_stdByLevsa)
        OMA_sup = np.array(list_meanByLevsa)+np.array(list_stdByLevsa)

        mean_limit_inf = np.min(np.array([np.min(mean_final), np.min(mean_finala)]))
        mean_limit_sup = np.max(np.array([np.max(mean_final), np.max(mean_finala)]))

        std_limit_inf = np.min(np.array([np.min(std_final), np.min(std_finala)]))
        std_limit_sup = np.max(np.array([np.max(std_final), np.max(std_finala)]))

        omfoma_limit_inf =     (np.min(np.array([np.min(OMF_inf), np.min(OMA_inf)])))
        if omfoma_limit_inf > 0:
            omfoma_limit_inf = 0.9*omfoma_limit_inf
        else:
            omfoma_limit_inf = 1.1*omfoma_limit_inf  
        omfoma_limit_sup = 1.1*(np.max(np.array([np.max(OMF_sup), np.max(OMA_sup)])))

        if (vminOMA == None) and (vmaxOMA == None): vminOMA, vmaxOMA = mean_limit_inf, 1.1*mean_limit_sup
        if vminOMA > 0:
            vminOMA = 0.9*vminOMA
        else:
            vminOMA = 1.1*vminOMA 

        vmaxOMAabs = np.max([np.abs(vminOMA),np.abs(vminOMA)])

        if (vminSTD == None) and (vmaxSTD == None): vminSTD, vmaxSTD = std_limit_inf - 0.1*std_limit_inf,  1.1*std_limit_sup

        date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
        instrument_title = str(varName) + '-' + str(varType) + '  |  ' + getVarInfo(varType, varName, 'instrument')

        # Figure with more than one channel - default all channels
        if channel == None or chanList == 1:
            fig = plt.figure(figsize=(6, 9))
            plt.rcParams['axes.facecolor'] = 'None'
            plt.rcParams['hatch.linewidth'] = 0.3

            ##### OMF

            plt.subplot(3, 1, 1)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(mean_final.T), origin='lower', vmin=-vmaxOMAabs, vmax=vmaxOMAabs, cmap='seismic', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Mean ('+omflag+')', labelpad=50)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 2)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(std_final.T), origin='lower', vmin=vminSTD, vmax=vmaxSTD, cmap='Blues', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Standard Deviation ('+omflag+')', labelpad=50)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 3)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(count_final.T), origin='lower', vmin=0.0, vmax=np.max(count_final), cmap='gist_heat_r', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Total Observations'+" ("+cmaski+")", labelpad=50)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.tight_layout()
            if chanList == 1:
                plt.savefig('hovmoller_'+str(varName) + '-' + str(varType)+'_'+omflag+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            else:
                plt.savefig('hovmoller_'+str(varName) + '-' + str(varType)+'_'+omflag+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMA

            fig = plt.figure(figsize=(6, 9))
            plt.rcParams['axes.facecolor'] = 'None'
            plt.rcParams['hatch.linewidth'] = 0.3

            plt.subplot(3, 1, 1)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(mean_finala.T), origin='lower', vmin=-vmaxOMAabs, vmax=vmaxOMAabs, cmap='seismic', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Mean ('+omflaga+')', labelpad=50)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 2)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(std_finala.T), origin='lower', vmin=vminSTD, vmax=vmaxSTD, cmap='Blues', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Standard Deviation ('+omflaga+')', labelpad=50)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.subplot(3, 1, 3)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(count_finala.T), origin='lower', vmin=0.0, vmax=np.max(count_finala), cmap='gist_heat_r', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Total Observations'+" ("+cmaski+")", labelpad=50)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.tight_layout()
            if chanList == 1:
                plt.savefig('hovmoller_'+str(varName) + '-' + str(varType)+'_'+omflaga+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            else:
                plt.savefig('hovmoller_'+str(varName) + '-' + str(varType)+'_'+omflaga+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

        # Figure with only one channel
        else:
        
            ##### OMF

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            ax1.plot(x_axis, list_meanByLevs, "b-", label="Mean ("+omflag+")")
            ax1.plot(x_axis, list_meanByLevs, "bo", label="Mean ("+omflag+")")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.set_ylabel('Mean ('+omflag+')', color='b', fontsize=10)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevs),ls='dotted',c='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(x_axis, std_final, "r-", label="Std. Deviation ("+omflag+")")
            ax2.plot(x_axis, std_final, "rs", label="Std. Deviation ("+omflag+")")
            ax2.set_ylim(vminSTD, vmaxSTD)
            ax2.set_ylabel('Std. Deviation ('+omflag+')', color='r', fontsize=10)
            ax2.tick_params('y', colors='r')
            major_ticks = np.arange(0, max(x_axis), len(DayHour)/len(list(filter(None, DayHour))))
            ax2.set_xticks(major_ticks)
            plt.axhline(y=np.mean(std_final),ls='dotted',c='red')

            ax3 = ax1.twinx()
            ax3.plot(x_axis, count_final, "g-", label="Total Observations"+" ("+cmaski+")")
            ax3.plot(x_axis, count_final, "g^", label="Total Observations"+" ("+cmaski+")")
            ax3.set_ylim(0, np.max(count_final) + (np.max(count_final)/8))
            ax3.set_ylabel('Total Observations'+" ("+cmaski+")", color='g', fontsize=10)
            ax3.tick_params('y', colors='g')
            ax3.spines["right"].set_position(("axes", 1.15))
            plt.yticks(rotation=90)
            plt.axhline(y=np.mean(count_final),ls='dotted',c='green')

            ax3.set_title(instrument_title, loc='left', fontsize=10)
            ax3.set_title(date_title, loc='right', fontsize=10)

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax3.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflag+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMA

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            ax1.plot(x_axis, list_meanByLevsa, "b-", label="Mean ("+omflaga+")")
            ax1.plot(x_axis, list_meanByLevsa, "bo", label="Mean ("+omflaga+")")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.set_ylabel('Mean ('+omflaga+')', color='b', fontsize=10)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevsa),ls='dotted',c='blue')
            
            ax2 = ax1.twinx()
            ax2.plot(x_axis, std_finala, "r-", label="Std. Deviation ("+omflaga+")")
            ax2.plot(x_axis, std_finala, "rs", label="Std. Deviation ("+omflaga+")")
            ax2.set_ylim(vminSTD, vmaxSTD)
            ax2.set_ylabel('Std. Deviation ('+omflaga+')', color='r', fontsize=10)
            ax2.tick_params('y', colors='r')
            plt.axhline(y=np.mean(std_finala),ls='dotted',c='red')

            ax3 = ax1.twinx()
            ax3.plot(x_axis, count_finala, "g-", label="Total Observations"+" ("+cmaski+")")
            ax3.plot(x_axis, count_finala, "g^", label="Total Observations"+" ("+cmaski+")")
            ax3.set_ylim(0, 1.2*np.max(count_finala))
            ax3.set_ylabel('Total Observations'+" ("+cmaski+")", color='g', fontsize=10)
            ax3.tick_params('y', colors='g')
            ax3.spines["right"].set_position(("axes", 1.15))
            plt.yticks(rotation=90)
            plt.axhline(y=np.mean(count_finala),ls='dotted',c='green')

            ax3.set_title(instrument_title, loc='left', fontsize=10)
            ax3.set_title(date_title, loc='right', fontsize=10)

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax3.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_'+omflaga+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()

            ##### OMF and OMA

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            ax1.plot(x_axis, list_meanByLevs, "b-", label="Mean ("+omflag+")")
            ax1.plot(x_axis, list_meanByLevs, "bo", label="")
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_meanByLevs),ls='dotted',c='blue')
            
            ax1.plot(x_axis, list_meanByLevsa, "r-", label="Mean ("+omflaga+")")
            ax1.plot(x_axis, list_meanByLevsa, "rs", label="")
            ax1.set_ylim(vminOMA, vmaxOMA)
            ax1.tick_params('y', colors='black')
            plt.axhline(y=np.mean(list_meanByLevsa),ls='dotted',c='red')

            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)

            ybox1 = TextArea('Mean ('+omflag+')' , textprops=dict(color="b", size=12,rotation=90,ha='left',va='bottom'))
            ybox2 = TextArea(' and '             , textprops=dict(color="k", size=12,rotation=90,ha='left',va='bottom'))
            ybox3 = TextArea('Mean ('+omflaga+')', textprops=dict(color="r", size=12,rotation=90,ha='left',va='bottom'))

            ybox = VPacker(children=[ybox3, ybox2, ybox1],align="bottom", pad=0, sep=5)

            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.12, 0.16), 
                                                bbox_transform=ax1.transAxes, borderpad=0.)

            ax1.add_artist(anchored_ybox)
            plt.legend()

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_OmFOmA_'+ forplotname +'.png', bbox_inches='tight', dpi=100)

            # OMF and OMA and StdDev

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')
            
            ax1.plot(x_axis, list_meanByLevs, lw=2, label='OmF Mean', color='blue', zorder=1)
            ax1.fill_between(x_axis, OMF_inf, OMF_sup, label='OmF Std Dev',  facecolor='blue', alpha=0.3, zorder=1)
            ax1.plot(x_axis, list_meanByLevsa, lw=2, label='OmA Mean', color='red', zorder=2)
            ax1.fill_between(x_axis, OMA_inf, OMA_sup, label='OmA Std Dev',  facecolor='red', alpha=0.3, zorder=2)
            ybox1 = TextArea(' OmF ' , textprops=dict(color="b", size=12,rotation=90,ha='left',va='bottom'))
            ybox2 = TextArea(' | '             , textprops=dict(color="k", size=12,rotation=90,ha='left',va='bottom'))
            ybox3 = TextArea(' OmA ', textprops=dict(color="r", size=12,rotation=90,ha='left',va='bottom'))

            ybox = VPacker(children=[ybox3, ybox2, ybox1],align="bottom", pad=0, sep=5)

            anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=0., frameon=False, bbox_to_anchor=(-0.125, 0.42), 
                                                bbox_transform=ax1.transAxes, borderpad=0.)

            ax1.add_artist(anchored_ybox)
            ax1.set_xlabel('Date (DayHour)', fontsize=12)
            ax1.set_ylim(omfoma_limit_inf,omfoma_limit_sup)
            ax1.legend(bbox_to_anchor=(-0.11, -0.25),ncol=4,loc='lower left', fancybox=True, shadow=False, frameon=True, framealpha=1.0, fontsize='11', facecolor='white', edgecolor='lightgray')
            plt.grid(axis='y', color='lightgray', linestyle='-.', linewidth=0.5, zorder=0)

            ax2 = ax1.twinx()
            ax2.plot(x_axis, list_countByLevsa, lw=2, label='OmA', linestyle='--', color='green', zorder=3)
            ax2.plot(x_axis, list_countByLevs, lw=2, label='OmF', linestyle=':', color='purple', zorder=3)
            ax2.set_ylabel('Total Observations (OmF | OmA)'+"\n ("+cmaski+")", fontsize=12)
            ax2.set_ylim(0, (np.max(list_countByLevsa) + np.max(list_countByLevsa)/5))
            ax2.legend(loc='upper left', ncol=2, fancybox=True, shadow=False, frameon=True, framealpha=1.0, fontsize='11', facecolor='white', edgecolor='lightgray')
            
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax2.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
        
            t = plt.annotate(forplot, xy=(0.78, 0.995), xytext=(-9, -9), xycoords='axes fraction', textcoords='offset points', color='darkgray', fontweight='bold', fontsize='10',
                                horizontalalignment='center', verticalalignment='center')
            t.set_bbox(dict(facecolor='whitesmoke', alpha=1.0, edgecolor='whitesmoke', boxstyle="square,pad=0.3"))

            plt.tight_layout()
            plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_OmFOmA_StdDev_'+ forplotname +'.png', bbox_inches='tight', dpi=100)

        # Cleaning up
        if Clean:
            plt.close('all')

        print(' Done!')
        print()
        
               

        return






# radiance final

    def statcount(self, varName=None, varType=None, noiqc=False, dateIni=None, dateFin=None, nHour="06", channel=None, figTS=False, figMap=False, **kwargs):

        '''
        The StatCount function plots a time series of assimilated, monitored and rejected data. 

        Example:

        varName = 'uv'           # Variable
        varType = 224            # Source Type
        noiqc = False            # noiqc GSI namelist parameter (OI QC - True or False)
        dateIni = 2013010100     # Inicial Date
        dateFin = 2013010900     # Final Date
        nHour = "06"             # Time Interval
        channel = None           # Radiance channel number (None for the conventional dataset)
        figTS = True             # Creates the time series plot
        figMap = False           # Creates the spatial plot for each time
        
        ! Case conventional dataset: channel = None
        ! The QC process creates a number indicating the data quality for each observation.
        ! These numbers are called QC markers in PrepBUFR files and are important as parts of
        ! the observation information. GSI uses QC markers to decide how to use the data. A 
        ! brief summary of the meaning of the QC markers is as follows:
        ! 
        !    +-----------------+-----------------------------------------------------------+
        !    | QC markes range | Data Process in GSI                                       |
        !    +-----------------+-----------------------------------------------------------+
        !    |  > 15 or        |GSI skips these observations during reading procedure. That|
        !    |  <= 0           |means these observations are tossed                        | 
        !    +-----------------+-----------------------------------------------------------+
        !    |  >= lim_qm      |These observations will be in monitoring status. That means|
        !    |  and            |these observations will be read in and be processed through|
        !    |  < = 15         |GSI QC process (gross check) and innovation calculation    | 
        !    |                 |stage but will not be used in inner iteration.             |
        !    +-----------------+-----------------------------------------------------------+
        !    |  > 0            |Observations will be used in further gross check (failure  |
        !    |  and            |observation will be list in rejection), innovation         |
        !    |  < lim_qm       |caalculation, and the analysis (inner iteration).          |
        !    +-----------------+-----------------------------------------------------------+

        !    +----------------------+---------------+---------------+
        !    |The value of namelist | lim_qm for Ps | lim_qm others |
        !    |option noiqc          |               |               |
        !    +----------------------+---------------+---------------+
        !    |True (without OI QC)  |       7       |       8       |
        !    +----------------------+---------------+---------------+
        !    |False (with OI QC)    |       4       |       4       |
        !    +----------------------+---------------+---------------+
        
        
        ! Case radiance dataset: channel = number
        ! There are three types of data classification: assimilated, monitored and rejected.
        ! Monitored data is organized into two groups: possibly assimilated and possibly rejected.
        !
        !    +------------------------+-------------+--------------------+
        !    |                        |   idqc      |        iuse        |
        !    +------------------------+-------------+--------------------+
        !    | Assimilated            |   == 0      |   >= 1             |
        !    +------------------------+-------------+--------------------+
        !    |            assimilated |   == 0      |   >= -1 and < 1    |
        !    | Monitored              |             |                    |
        !    |            rejected    |   != 0      |   >= -1 and < 1    |
        !    +------------------------+-------------+--------------------+
        !    | Rejected               |   != 0      |   >= 1             |
        !    +------------------------+-------------+--------------------+
        '''


        if(noiqc):
            lim_qm = 8
            if(varName == 'ps'):
                lim_qm = 7
        else:
            lim_qm = 4

        varInfo = getVarInfo(varType, varName, 'instrument')
        if varInfo is not None:
            instrument_title = str(varName) + '-' + str(varType) + '  |  ' + varInfo
        else:
            instrument_title = str(varName) + '-' + str(varType) + '  |  ' + 'Unknown instrument'

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        assi, reje, moni, DayHour_tmp = [], [], [], []
        moniAssi, moniReje = [], []
        assif, rejef, monif, moniAssif, moniRejef = [], [], [], [], []
        f = 0
        while (date <= datef):

            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            # try: For issues reading the file (file not found)
            # in the except statement an error message is printed and continues for other dates
            try:
                
                if(channel == None):  # Conventional
                    exp = "(iuse==1)"
                    assim = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(iuse==-1) & (idqc >= "+str(lim_qm)+" and idqc <= 15)"
                    monit = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(iuse==-1) & ((idqc > 15 or idqc <= 0) or (idqc > 0 and idqc < "+str(lim_qm)+"))"
                    rejei = self[f].obsInfo[varName].loc[varType].query(exp)
                
                    assi.append(len(assim))
                    moni.append(len(monit))
                    reje.append(len(rejei))
                
                    if (figMap):
                        df_list = [assim, monit, rejei]
                        name_list = ["Assimilated ["+str(len(assim))+"]","Monitored ["+str(len(monit))+"]","Rejected ["+str(len(rejei))+"]"]
                        marker_list = [".","x","*"]     
                        color_list = ["green","blue","red"]
                    
                        setColor = 0 
                        legend_labels = []
                    
                        fig = plt.figure(figsize=(12, 6))
                        ax  = fig.add_subplot(1, 1, 1)
                        ax = geoMap(area=None,ax=ax)
                        for dfi,namedf,mk,cl in zip(df_list,name_list,marker_list,color_list):
                            df    = dfi
                            legend_labels.append(mpatches.Patch(color=cl, label=namedf) )
                            ax = df.plot(ax=ax,legend=True, marker=mk, color=cl, **kwargs)
                            setColor += 1
                            plt.legend(handles=legend_labels, numpoints=1, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                                    fancybox=True, shadow=False, frameon=False, ncol=3, prop={"size": 10})
                    
                        date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                        plt.title(date_title, loc='right', fontsize=10)
                        plt.title(instrument_title, loc='left', fontsize=9)
                    
                        plt.tight_layout()
                        plt.savefig('TotalObs_'+str(varName) + '-' + str(varType)+'_'+datefmt+'.png', bbox_inches='tight', dpi=100)
                
                
                else:   # Radiance
                    exp = "(nchan=="+str(channel)+") & (iuse >= 1 & idqc==0.0)"
                    assim = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(nchan=="+str(channel)+") & ((iuse >= -1 and iuse < 1) & idqc==0.0)"
                    monitAssim = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(nchan=="+str(channel)+") & ((iuse >= -1 and iuse < 1) & idqc!=0.0)"
                    monitRejei = self[f].obsInfo[varName].loc[varType].query(exp)
                    exp = "(nchan=="+str(channel)+") & (iuse >= 1 & idqc!=0.0)"
                    rejei = self[f].obsInfo[varName].loc[varType].query(exp)
                
                    assi.append(len(assim))
                    moniAssi.append(len(monitAssim))
                    moniReje.append(len(monitRejei))
                    reje.append(len(rejei))
                    
                    forplot = 'Channel ='+str(channel)
                
                    # Radiance plots
                    if (figMap):
                        # Case: assimilated and rejected
                        if ((len(assim)) != 0 or (len(rejei)) != 0):
                            df_list = [assim, rejei]    
                            name_list = ["Assimilated ["+str(len(assim))+"]","Rejected ["+str(len(rejei))+"]"]
                            marker_list = ["^","v"]    
                            color_list = ["green","red"]
                    
                            setColor = 0 
                            legend_labels = []
                    
                            fig = plt.figure(figsize=(12, 6))
                            ax  = fig.add_subplot(1, 1, 1)
                            ax = geoMap(area=None,ax=ax)
                            for dfi,namedf,mk,cl in zip(df_list,name_list,marker_list,color_list):
                                df    = dfi
                                legend_labels.append(mpatches.Patch(color=cl, label=namedf) )
                                ax = df.plot(ax=ax,legend=True, marker=mk, color=cl, **kwargs) 
                                setColor += 1
                                plt.legend(handles=legend_labels, numpoints=1, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                                        fancybox=True, shadow=False, frameon=False, ncol=2, prop={"size": 10})
                        
                            date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                            plt.title(date_title, loc='right', fontsize=10)
                            plt.title(instrument_title, loc='left', fontsize=9)
                            plt.annotate(forplot, xy=(0.45, 1.015), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                                         color='gray', fontweight='bold', fontsize='10', horizontalalignment='left', 
                                         verticalalignment='center')
                    
                            plt.tight_layout()
                            plt.savefig('Assim-Rejei_'+str(varName) + '-' + str(varType)+'_'+ 'CH' + str(channel) + '_' +datefmt+'.png', 
                                        bbox_inches='tight', dpi=100)
                        else:
                            print("channel ",channel," not assimilated or rejected on the date -->",date.strftime("%Y-%m-%d:%H"))
                    
                        # Monitored cases: would be assimilated or rejected 
                        if ((len(monitAssim)) != 0 or (len(monitRejei)) != 0):
                            df_list = [monitAssim, monitRejei]
                            name_list = ["Monitored-Assimilated ["+str(len(monitAssim))+"]","Monitored-Rejected ["+str(len(monitRejei))+"]"]
                            marker_list = ["^","v"]   
                            color_list = ["teal","purple"]
                    
                            setColor = 0 
                            legend_labels = []
                    
                            fig = plt.figure(figsize=(12, 6))
                            ax  = fig.add_subplot(1, 1, 1)
                            ax = geoMap(area=None,ax=ax)
                            for dfi,namedf,mk,cl in zip(df_list,name_list,marker_list,color_list):
                                df    = dfi
                                legend_labels.append(mpatches.Patch(color=cl, label=namedf) )
                                ax = df.plot(ax=ax,legend=True, marker=mk, color=cl, **kwargs) 
                                setColor += 1
                                plt.legend(handles=legend_labels, numpoints=1, loc='lower center', bbox_to_anchor=(0.5, -0.02), 
                                        fancybox=True, shadow=False, frameon=False, ncol=2, prop={"size": 10})
                        
                            date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                            plt.title(date_title, loc='right', fontsize=10)
                            plt.title(instrument_title, loc='left', fontsize=9)
                            plt.annotate(forplot, xy=(0.45, 1.015), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                                         color='gray', fontweight='bold', fontsize='10', horizontalalignment='left', 
                                         verticalalignment='center')
                    
                            plt.tight_layout()
                            plt.savefig('Monitored_'+str(varName) + '-' + str(varType)+'_'+ 'CH' + str(channel) + '_'+datefmt+'.png', 
                                        bbox_inches='tight', dpi=100)
                        else:
                            print("channel ",channel," not monitored on the date -->",date.strftime("%Y-%m-%d:%H"))
                    
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> STATCOUNT ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                if(channel == None):
                    assi.append(None)
                    moni.append(None)
                    reje.append(None)
                else:
                    assi.append(None)
                    moniAssi.append(None)
                    moniReje.append(None)
                    reje.append(None)
                    
            f = f + 1
            date = date + timedelta(hours=int(nHour))
            date_finale = date
            


        if (figTS):
            if(channel == None):   # Conventional
                if(len(DayHour_tmp) > 4):
                    DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
                else:
                    DayHour = DayHour_tmp
                
                x_axis      = np.arange(0, len(DayHour), 1)
                date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
            
                fig = plt.figure(figsize=(6, 4))
                fig, ax1 = plt.subplots(1, 1)
                plt.style.use('seaborn-v0_8-ticks')

                plt.axhline(y=0.0,ls='solid',c='#d3d3d3')

                ax1.plot(x_axis, assi, "o", label="Assimilated \n["+str(sum(assi))+"]", color='green')
                ax1.plot(x_axis, moni, "o", label="Monitored \n["+str(sum(moni))+"]", color='blue')
                ax1.plot(x_axis, reje, "o", label="Rejected \n["+str(sum(reje))+"]", color='red')
                ax1.legend(fancybox=True, frameon=True, shadow=True, loc="upper center",ncol=3)
                ax1.set_xlabel('Date (DayHour)', fontsize=10)
                plt.title(date_title, loc='right', fontsize=10)
                plt.title(instrument_title, loc='left', fontsize=9)
                
                ax1.set_ylim(np.round(-0.05*np.max([assi,moni,reje])), np.round(1.25*np.max([assi,moni,reje])))
                ax1.set_ylabel('Total Observations', color='black', fontsize=10)
                ax1.tick_params('y', colors='black')
                plt.xticks(x_axis, DayHour)
                major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
                ax1.set_xticks(major_ticks)
                plt.axhline(y=np.mean(assi),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(moni),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(reje),ls='dotted',c='lightgray')
                plt.tight_layout()
                plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_TotalObs.png', bbox_inches='tight', dpi=100)
                
            else:   # Radiance
                if(len(DayHour_tmp) > 4):
                    DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
                else:
                    DayHour = DayHour_tmp
                
                x_axis      = np.arange(0, len(DayHour), 1)
                date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
            
                fig = plt.figure(figsize=(6, 4))
                fig, ax1 = plt.subplots(1, 1)
                plt.style.use('seaborn-v0_8-ticks')

                plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
                
                # List with value None: is removed to calculate sum, max and min
                # The lists below are only used to define the scale of the axes and the total sum of assi/rejei/monit data
                assif     = [x for x in assi if x != None]
                moniAssif = [x for x in moniAssi if x != None]
                moniRejef = [x for x in moniReje if x != None]
                rejef     = [x for x in reje if x != None]

                ax1.plot(x_axis, assi, "o", label="Assimilated \n["+str(sum(assif))+"]", color='green')
                ax1.plot(x_axis, moniAssi, "o", label="Monitored-Assim \n["+str(sum(moniAssif))+"]", color='teal')
                ax1.plot(x_axis, moniReje, "o", label="Monitored-Rejei \n["+str(sum(moniRejef))+"]", color='purple')
                ax1.plot(x_axis, reje, "o", label="Rejected \n["+str(sum(rejef))+"]", color='red')
                ax1.legend(fancybox=True, frameon=True, shadow=True, loc="best",ncol=1)
                ax1.set_xlabel('Date (DayHour)', fontsize=10)
                plt.title(date_title, loc='right', fontsize=10)
                plt.title(instrument_title, loc='left', fontsize=9)
                plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                             color='lightgray', fontweight='bold', fontsize='12', horizontalalignment='left', verticalalignment='center')
                
                ax1.set_ylim(np.round(-0.05*np.max([assif,moniAssif,moniRejef,rejef])),
                             np.round(1.25*np.max([assif,moniAssif,moniRejef,rejef])))
                ax1.set_ylabel('Total Observations', color='black', fontsize=10)
                ax1.tick_params('y', colors='black')
                plt.xticks(x_axis, DayHour)
                major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
                ax1.set_xticks(major_ticks)
                plt.axhline(y=np.mean(assif),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(moniAssif),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(moniRejef),ls='dotted',c='lightgray')
                plt.axhline(y=np.mean(rejef),ls='dotted',c='lightgray')
                plt.tight_layout()
                plt.savefig('time_series_'+str(varName) + '-' + str(varType) +'_'+ 'CH' + str(channel) + '_'+'_TotalObs.png',
                            bbox_inches='tight', dpi=100)



# Avaliação idqc radiancia
    def statcount_idqc(self, varName=None, varType=None, dateIni=None, dateFin=None, nHour="06", channel=None, figTS=False, figMap=False, **kwargs):

        '''
        The StatCount_idqc function plots a time series of the idqc flags. 

        Example:

        varName = 'amsua'        # Sensor
        varType = 'n19'          # Satellite
        dateIni = 2013010100     # Inicial Date
        dateFin = 2013010900     # Final Date
        nHour = "06"             # Time Interval
        channel = 10             # Radiance channel number
        figTS = True             # Creates the time series plot
        figMap = False           # Creates the spatial plot for each time        
        
        ! Table: classification idqc flags (atualizada com branch https://projetos.cptec.inpe.br/projects/gsi/repository/entry/branch/gsi_t11824/src/gsi/qcmod.f90 - Versão 3.7 GSI)
        !
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   idqc |   flag                                          |   idqc |   flag                                          |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   0    | Good Observations                               |                       SENSOR AMSUA                       |
        !    |        |                                                 |                        QC_AMSUA                          |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   1    | Reject due to flag in radinfo in setuprad       |   50   | Reject because 'factch6 > limit' in subroutine  |
        !    |        |                                                 |        | qc_amsua                                        |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   2    | Failure in CRTM in setuprad                     |   51   | Reject because 'factch4 > limit' in subroutine  |
        !    |        |                                                 |        | qc_amsua                                        |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   3    | Reject due to gross check failure in setuprad   |   52   | Reject because 'sval > limit' in subroutine     |
        !    |        |                                                 |        | qc_amsua over open water                        |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   4    | Reject due to interchannel check (if one channel|   53   | Reject because 'factch5 > limit' in subroutine  |
        !    |        | fails in group whole group thrown out)          |        | qc_amsua over open water                        |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   5    | Reject due to not using over this surface       |                       SENSOR HIRS/4                      |
        !    |        | in qc routine                                   |                         QC_IRSND                         |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   6    | Reject due to gross check in specific           |   50   | Reject because 'wavenumber > 2400'  in          |
        !    |        | qc routine                                      |        | subroutine qc_irsnd                             |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   7    | Reject due to 'cloud > limit' for channel       |   51   | Reject because 'wavenumber > 2000' in           |
        !    |        | in qc routine                                   |        | subroutine qc_irsnd                             |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   8    | Reject due to inaccurate emissivity/surface     |   52   | Reject because goes sounder and satellite       |
        !    |        | temperature estimate in qc routine              |        | 'zenith angle > 60' in subroutine qc_irsnd      |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   9    | Reject due to observations being out of range   |   53   | Reject because of surface emissivity/temperature|
        !    |        | in qc routine                                   |        | influence in subroutine qc_irsnd                |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   11   | Reject because outside the range of             |
        !    |        | lsingleradob                                    |
        !    +--------+-------------------------------------------------+
        !    |   12   | Reject due to cold-air outbreak area check      |
        !    |        | in setuprad                                     |
        !    +--------+-------------------------------------------------+


        Obs.: Failures specific to qc routine start at 50 and the numbers overlap
        '''


        varInfo = getVarInfo(varType, varName, 'instrument')
        if varInfo is not None:
            instrument_title = str(varName) + '-' + str(varType) + '  |  ' + varInfo
        else:
            instrument_title = str(varName) + '-' + str(varType) + '  |  ' + 'Unknown instrument'

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        nf = {}
        List_flags = []
        DayHour_tmp = []
        
        f = 0
        while (date <= datef):

            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            
            # try: For issues reading the file (file not found)
            # in the except statement an error message is printed and continues for other dates
            try:
                
                flags, aux0 = [], []
                
                if varName == 'amsua':
                    #flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51]
                    aux0 = list(map(round,self[f].obsInfo[varName].query("nchan=="+str(channel)).loc[varType].idqc))
                    flags = pd.unique(aux0)
                    flags.sort()
                elif varName == 'hirs4':
                    #flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51, 52, 53]
                    aux0 = list(map(int,self[f].obsInfo[varName].query("nchan=="+str(channel)).loc[varType].idqc))
                    flags = pd.unique(aux0)
                    flags.sort()
                else:
                    print(setcolor.WARNING + "    >>> Flags for sensor " + varName + " not set <<< " + setcolor.ENDC)
                    
                print('flags = ', flags)
                if len(flags) > 0:
                    List_flags.append(flags)
                
                
                soma_flags = 0
                setColor = 0 
                legend_labels = []
                if (figMap):
                    fig = plt.figure(figsize=(12, 6))
                    ax  = fig.add_subplot(1, 1, 1)
                    ax = geoMap(area=None,ax=ax)
                
                for ll in range(len(flags)):
                    fl = flags[ll]
                    print('ll = ', ll,'flag = ', fl)
                    aux1 = []
                    
                    #---------------- flags inteiras ------------#
                    exp  = "(nchan=="+str(channel)+") & (idqc=="+str(fl)+")"
                    
                    #---------------- lista com os dados da flag ------------#
                    aux1 = self[f].obsInfo[varName].loc[varType].query(exp)                   
                    
                    #---------------- soma qntdd de dados rejeitados (idqc!=0) ------------#
                    if (fl != 0.0):
                        soma_flags = soma_flags + len(aux1)
                    
                    #---------------- Comandos de plote ------------#
                    if ( len(aux1) > 0 and figMap == True ):
                        label = "flag "+str(fl)+" ["+str(len(aux1))+"]"
                        
                        color = getColor(minVal=0, maxVal=len(flags), 
                                         value=ll,hex=True,cmapName='tab20')
                        
                        
                        legend_labels.append(mpatches.Patch(color=color, label=label) )
                        ax = aux1.plot(ax=ax,legend=True, marker="o", color=color, **kwargs)
                        setColor += 1
                        plt.legend(handles=legend_labels, numpoints=1, loc='center left', bbox_to_anchor=(1.0, 0.5), 
                                   fancybox=True, shadow=False, frameon=False, ncol=1, prop={"size": 10})
                    
                    #---------------- FIM --> flags inteiras ------------#
                    
                    aux2 = []
                     
                    #---------------- teste flegs decimais com arredondamento ------------#
                    if (fl <= -1):
                        inf = fl - 1
                        exp  = "(nchan=="+str(channel)+") & (idqc>"+str(inf)+" & idqc<"+str(fl)+")"
                        
                        #---------------- lista com os dados da flag ------------#
                        aux2 = self[f].obsInfo[varName].loc[varType].query(exp)
                        
                        label_name = "flag btw ("+str(inf)+" "+str(fl)+") ["+str(len(aux2))+"]"
                        
                        #---------------- soma qntdd de dados rejeitados (idqc!=0) ------------#
                        if (fl != 0.0):
                            soma_flags = soma_flags + len(aux2)
                            
                        
                    elif (fl >= 1):
                        sup = fl + 1
                        exp  = "(nchan=="+str(channel)+") & (idqc>"+str(fl)+" & idqc<"+str(sup)+")"
                        
                        #---------------- lista com os dados da flag ------------#
                        aux2 = self[f].obsInfo[varName].loc[varType].query(exp)
                        
                        label_name = "flag btw ("+str(fl)+" "+str(sup)+") ["+str(len(aux2))+"]"
                        
                        #---------------- soma qntdd de dados rejeitados (idqc!=0) ------------#
                        if (fl != 0.0):
                            soma_flags = soma_flags + len(aux2)
                            
                                
                    else:
                        sup = fl + 1
                        inf = fl - 1
                        exp  = "(nchan=="+str(channel)+") & ((idqc != 0) and (idqc>"+str(inf)+" and idqc<"+str(sup)+"))"
                        
                        #---------------- lista com os dados da flag ------------#
                        aux2 = self[f].obsInfo[varName].loc[varType].query(exp)
                        
                        label_name = "flag btw ("+str(inf)+" 0)U(0 "+str(sup)+") ["+str(len(aux2))+"]"
                        
                        #---------------- soma qntdd de dados rejeitados (idqc!=0) ------------#
                        if (len(aux2) > 0):
                            soma_flags = soma_flags + len(aux2)
                            
                    #---------------- Comandos de plote ------------#
                    if ( len(aux2) > 0 and figMap == True ):
                        label = label_name
                        
                        color = getColor(minVal=0, maxVal=len(flags), 
                                         value=ll,hex=True,cmapName='Paired')
                        
                        legend_labels.append(mpatches.Patch(color=color, label=label) )
                        ax = aux2.plot(ax=ax,legend=True, marker="o", color=color, **kwargs)
                        setColor += 1
                        plt.legend(handles=legend_labels, numpoints=1, loc='center left', bbox_to_anchor=(1.0, 0.5), 
                                   fancybox=True, shadow=False, frameon=False, ncol=1, prop={"size": 10})
                        
                    #---------------- FIM --> flags decimas - intervalos ------------#
                    
                
                #---------------- comandos finais do plote e salva figura ------------#
                if (figMap):
                    forplot = 'Channel ='+str(channel)+' | Rejected = '+str(soma_flags)
                
                    date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                    plt.title(date_title, loc='right', fontsize=10)
                    plt.title(instrument_title, loc='left', fontsize=9)
                    plt.annotate(forplot, xy=(0.45, 1.015), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                                 color='gray', fontweight='bold', fontsize='10', horizontalalignment='left', verticalalignment='center')
                
                    plt.tight_layout()
                    plt.savefig('Flags-idqc_'+str(varName) + '-' + str(varType)+'_'+ 'CH' + str(channel) + '_' +datefmt+'.png', 
                                bbox_inches='tight', dpi=100)
                    
                
                    
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> STATCOUNT_idqc ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                
                
                    
            f = f + 1
            date = date + timedelta(hours=int(nHour))
            date_finale = date
            

        #figTS ----> Fazer a série temporal da quantidade de obs de cada flag idqc    

#-------------- Novas funções: ler e plotar arquivo fort.220 --------------------#
    def fort220_plot(self, dateIni=None, dateFin=None, nHour="06", Label=None, Flag=None, cost_gradient=False, vmin=None, vmax=None, Clean=None, **kwargs):
        """
        Function plot Label versus inner loop of the fort.220 file. A figure for each date in range [dateIni, dateFin].
        
        """
        
        idx = pd.IndexSlice
        
        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei
        
        DayHour_tmp = []
        
        f = 0
        while (date <= datef):

            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            
            # try: For issues reading the file (file not found)
            # in the except statement an error message is printed and continues for other dates
            try:
                
                InnerLoop1, InnerLoop2 = [], []
                if cost_gradient:
                    InnerLoop1 = self[f][0].loc[idx[:], idx['InnerLoop']]   # 1º outer loop
                    InnerLoop2 = self[f][1].loc[idx[:], idx['InnerLoop']]   # 2º outer loop
                    df1 = self[f][0].loc[idx[:], idx[Label]]  # 1º outer loop
                    df2 = self[f][1].loc[idx[:], idx[Label]]  # 2º outer loop
                else:
                    InnerLoop1 = self[f][0].loc[idx[:,[Label]], idx['InnerLoop']]   # 1º outer loop
                    InnerLoop2 = self[f][1].loc[idx[:,[Label]], idx['InnerLoop']]   # 2º outer loop
                    df1 = self[f][0].loc[idx[:,[Label]], idx[Flag]]  # 1º outer loop
                    df2 = self[f][1].loc[idx[:,[Label]], idx[Flag]]  # 2º outer loop
                
                xmin = min([min(InnerLoop1), min(InnerLoop2)])
                xmax = max([max(InnerLoop1), max(InnerLoop2)])
                
                #ymin = min([min(df1), min(df2)])
                #ymax = max([max(df1), max(df2)])
                
                nloop_1 = InnerLoop1.iloc[-1]
                
                #ymin = ymin - (ymax - ymin)/nloop_1
                #ymax = ymax + (ymax - ymin)/nloop_1
                
                ymin = vmin
                ymax = vmax
                
                fig = plt.figure(figsize=(6, 4))
                fig, ax1 = plt.subplots(1, 1)
                plt.style.use('seaborn-v0_8-darkgrid')
                
                date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                
                ax1.plot(InnerLoop1, df1, "-", label="1º outer loop", **kwargs)
                ax1.plot(InnerLoop2, df2, "--", label="2º outer loop", **kwargs)
                ax1.legend(fancybox=True, frameon=True, shadow=True, loc="best",ncol=1)
                ax1.grid(True)
                ax1.set_xlabel('Inner loop', color='black', fontsize=12)
                ax1.set_xlim(xmin, xmax)
                ax1.set_ylim(ymin, ymax)
                ax1.set_ylabel('('+Label+')', color='black', fontsize=14)
                plt.title(date_title, loc='right', fontsize=12)
                
                plt.tight_layout()
                plt.savefig('Fort220_'+str(Label) +'_'+datefmt+'.png', bbox_inches='tight', dpi=100)
                
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> fort220_plot_cost_gradient ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                
            f = f + 1
            date = date + timedelta(hours=int(nHour))
            date_finale = date
            
            
            
    def fort220_plot_lines(self, dateIni=None, dateFin=None, nHour="06", Label=None, Flag=None, cost_gradient=False, vmin=None, vmax=None, Clean=None, **kwargs):
        """
        Function plot Label versus inner loop of the fort.220 file. A figure for all date in range [dateIni, dateFin].
        
        """
        
        idx = pd.IndexSlice
        
        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei
        
        DayHour_tmp = []
        
        colors = mpl.colormaps['tab20'].colors
        
        fig = plt.figure(figsize=(10, 6))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-darkgrid')
        
        ymin = 990000000
        ymax = 0.000000
        f = 0
        while (date <= datef):

            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            
            # try: For issues reading the file (file not found)
            # in the except statement an error message is printed and continues for other dates
            try:
                
                InnerLoop1, InnerLoop2 = [], []
                if cost_gradient:
                    InnerLoop1 = self[f][0].loc[idx[:], idx['InnerLoop']]   # 1º outer loop
                    InnerLoop2 = self[f][1].loc[idx[:], idx['InnerLoop']]   # 2º outer loop
                    df1 = self[f][0].loc[idx[:], idx[Label]]  # 1º outer loop
                    df2 = self[f][1].loc[idx[:], idx[Label]]  # 2º outer loop
                else:
                    InnerLoop1 = self[f][0].loc[idx[:,[Label]], idx['InnerLoop']]   # 1º outer loop
                    InnerLoop2 = self[f][1].loc[idx[:,[Label]], idx['InnerLoop']]   # 2º outer loop
                    df1 = self[f][0].loc[idx[:,[Label]], idx[Flag]]  # 1º outer loop
                    df2 = self[f][1].loc[idx[:,[Label]], idx[Flag]]  # 2º outer loop
                
                xmin = min([min(InnerLoop1), min(InnerLoop2)])
                xmax = max([max(InnerLoop1), max(InnerLoop2)])
                
                ymin = min([min(df1), min(df2), ymin])
                ymax = max([max(df1), max(df2), ymax])
                
                nloop_1 = InnerLoop1.iloc[-1]
                
                ymin = ymin - (ymax - ymin)/nloop_1
                ymax = ymax + (ymax - ymin)/nloop_1
                
                color = colors[f]
                
                ax1.plot(InnerLoop1, df1, "-", c=color, label="1º out. "+str(date.strftime("%Y-%m-%d:%H")), **kwargs)
                ax1.plot(InnerLoop2, df2, "--", c=color, label="2º out. "+str(date.strftime("%Y-%m-%d:%H")), **kwargs)
                
                
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> fort220_plot_cost_gradient ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                
            f = f + 1
            date = date + timedelta(hours=int(nHour))
            date_finale = date
            
        
        date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
        
        ax1.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   fancybox=True, shadow=False, frameon=False, ncol=4, prop={"size": 10})
        ax1.grid(True)
        ax1.set_xlabel('Inner loop', color='black', fontsize=12)
        ax1.set_xlim(xmin, xmax)
        ax1.set_ylim(ymin, ymax)
        ax1.set_ylabel('('+Label+')', color='black', fontsize=14)
        plt.title(date_title, loc='right', fontsize=12)
            
        plt.tight_layout()
        plt.savefig('Fort220_lines_time_'+str(Label) +'.png', bbox_inches='tight', dpi=100)

            
#-------------- Novas funções: ler e plotar arquivo fort.220 --------------------#
    def fort220_time(self, dateIni=None, dateFin=None, nHour="06", Label=None, Flag=None, cost_gradient=False, Clean=None):
        """
        Function plot Label versus date of the fort.220 file in first and last inner loops.
        
        """
        
        idx = pd.IndexSlice
        
        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei
        
        DayHour_tmp = []
        df1_first, df2_first, df1_last, df2_last = [], [], [], []
        
        f = 0
        while (date <= datef):

            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            
            # try: For issues reading the file (file not found)
            # in the except statement an error message is printed and continues for other dates
            try:
                nloop1 = 0
                nloop2 = 0
                
                if cost_gradient:
                    nloop1 = self[f][0].iloc[-1, 0]  # ultimo valor na coluna iteração
                    nloop2 = self[f][1].iloc[-1, 0]  # ultimo valor na coluna iteração
                    df1_first.append(self[f][0].loc[idx[0], idx[Label]])  # 1º outer loop
                    df2_first.append(self[f][1].loc[idx[0], idx[Label]])  # 2º outer loop
                    df1_last.append(self[f][0].loc[idx[nloop1], idx[Label]])  # 1º outer loop
                    df2_last.append(self[f][1].loc[idx[nloop2], idx[Label]])  # 2º outer loop
                else:
                    nloop1 = self[f][0].iloc[-1, 0]  # ultimo valor na coluna iteração
                    nloop2 = self[f][1].iloc[-1, 0]  # ultimo valor na coluna iteração
                    df1_first.append(self[f][0].loc[idx[0,[Label]], idx[Flag]])  # 1º outer loop
                    df2_first.append(self[f][1].loc[idx[0,[Label]], idx[Flag]])  # 2º outer loop
                    df1_last.append(self[f][0].loc[idx[nloop1,[Label]], idx[Flag]])  # 1º outer loop
                    df2_last.append(self[f][1].loc[idx[nloop2,[Label]], idx[Flag]])  # 2º outer loop
                    
                    
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> fort220_plot_cost_gradient ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                
                
            f = f + 1
            date = date + timedelta(hours=int(nHour))
            date_finale = date
                    
                    
        if(len(DayHour_tmp) > 4):
            DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
        else:
            DayHour = DayHour_tmp
                
        x_axis      = np.arange(0, len(DayHour), 1)
        date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
            
        fig = plt.figure(figsize=(10, 4))
        fig, ax1 = plt.subplots(1, 1)
        plt.style.use('seaborn-v0_8-darkgrid')

        plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
        
        ax1.plot(x_axis, df1_first, "-o", label="1º Outer loop: Inner [0]")
        ax1.plot(x_axis, df1_last, "-o", label="1º Outer loop: Inner ["+str(nloop1)+"]")
        ax1.plot(x_axis, df2_first, "-o", label="2º Outer loop: Inner [0]")
        ax1.plot(x_axis, df2_last, "-o", label="2º Outer loop: Inner ["+str(nloop2)+"]")
        ax1.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   fancybox=True, shadow=True, frameon=True, ncol=2, prop={"size": 10})
        ax1.set_xlabel('Date (DayHour)', fontsize=10)
        plt.title(date_title, loc='right', fontsize=10)
                
        ax1.set_ylim(np.round(-0.05*np.max([df1_first,df1_last,df2_first,df2_last])), 
                     np.round(1.05*np.max([df1_first,df1_last,df2_first,df2_last])))
        ax1.set_ylabel(Label, color='black', fontsize=12)
        ax1.tick_params('y', colors='black')
        plt.xticks(x_axis, DayHour)
        major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
        ax1.set_xticks(major_ticks)
        plt.axhline(y=np.mean(df1_first),ls='dotted',c='lightgray')
        plt.axhline(y=np.mean(df1_last),ls='dotted',c='lightgray')
        plt.axhline(y=np.mean(df2_first),ls='dotted',c='lightgray')
        plt.axhline(y=np.mean(df2_last),ls='dotted',c='lightgray')
        plt.tight_layout()
        plt.savefig('time_series_'+str(Label) + 'Flag-' + str(Flag) + '_'+'Innerloop.png', 
                    bbox_inches='tight', dpi=100)
        
        
        
#-------------- Novas funções: ler e plotar arquivo fort.220 --------------------#
    def fort220_time_Obscontrib(self1, self2, dateIni=None, dateFin=None, nHour="06", Label=None, LabelObs=None, Flag=None, Clean=None):
        """
        The total cost function (cost) is sum of the list J (list of contribuition of the each observation). This Function plot the rate (DQ) of the observation contribuition LabelObs in respect the Label variable. The quocient DQ is calculate for each outer loop:
        
        DQ = ( J_last - J_first )/( cost_last - cost_first )
        
        last --> value in last inner loop
        firs --> value in first inner loop
        
        If Flag is None are ploted bar colunms for each flag and a figure for each date is returned. If Flag is a number, two time siries plotes are returnes: 
        
        1º ) Q = (J/cost)*100  in first and last inner loop of the each outer loops.
        2º ) DQ of the each outer loop.
        
        """
        
        idx = pd.IndexSlice
        
        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei
        
        DayHour_tmp = []
        Q1_first, Q2_first, Q1_last, Q2_last = [], [], [], []
        
        DQ1, DQ2 = [], []
        
        f = 0
        while (date <= datef):

            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            
            # try: For issues reading the file (file not found)
            # in the except statement an error message is printed and continues for other dates
            if Flag == None:
                
                Q1_first, Q2_first, Q1_last, Q2_last = [], [], [], []
                DQ1, DQ2 = [], []
                col_flags, xFlags1, xFlags2 = [], [], []
                
                col_flags = self2[f][0].loc[idx[:,idx[:]], idx['1':]].columns.tolist()
                
                for Flg in col_flags:
                    
                    try:
                        nloop1 = 0
                        nloop2 = 0

                        df1_first, df2_first, df1_last, df2_last = 0, 0, 0, 0
                        dfo1_first, dfo2_first, dfo1_last, dfo2_last = 0, 0, 0, 0


                        nloop1 = self1[f][0].iloc[-1, 0]  # ultimo valor na coluna iteração
                        nloop2 = self1[f][1].iloc[-1, 0]  # ultimo valor na coluna iteração

                        df1_first = self1[f][0].loc[idx[0], idx[Label]]  # 1º outer loop
                        df2_first = self1[f][1].loc[idx[0], idx[Label]]  # 2º outer loop
                        df1_last  = self1[f][0].loc[idx[nloop1], idx[Label]]  # 1º outer loop
                        df2_last  = self1[f][1].loc[idx[nloop2], idx[Label]]  # 2º outer loop

                        dfo1_first = self2[f][0].loc[idx[0,[LabelObs]], idx[Flg]].values  # 1º outer loop
                        dfo2_first = self2[f][1].loc[idx[0,[LabelObs]], idx[Flg]].values  # 2º outer loop
                        dfo1_last  = self2[f][0].loc[idx[nloop1,[LabelObs]], idx[Flg]].values  # 1º outer loop
                        dfo2_last  = self2[f][1].loc[idx[nloop2,[LabelObs]], idx[Flg]].values  # 2º outer loop
                        
                        if abs(dfo1_last) != 0:
                            xFlags1.append(float(Flg)) 
                            #if abs(df1_last - df1_first) > 10e-06:
                            DQ1.append( ( (dfo1_last[0] - dfo1_first[0]) / (df1_last - df1_first) ) )
                            #else:
                            #    DQ1.append(-99)
                                
                        if abs(dfo2_last) != 0:
                            xFlags2.append(float(Flg)) 
                            #if abs(df2_last - df2_first) > 10e-06:
                            DQ2.append( ( (dfo2_last[0] - dfo2_first[0]) / (df2_last - df2_first) ) )
                            #else:
                            #    DQ1.append(-99)


                    except:
                        print("++++++++++++++++++++++++++ ERROR: file reading --> fort220_plot_cost_gradient ++++++++++++++++++++++++++")
                        print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                
                if len(DQ1)>0 or len(DQ2)>0:
                    x_axis1 = xFlags1
                    x_axis2 = xFlags2
                    x_Flags = sorted(list(set(xFlags1 + xFlags2)))
                    print('x_Flags = ',x_Flags)
                    x_axis  = np.arange(0, len(x_Flags), 1)
                    date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                    forplot = 'Flag = '+ str(Flag)
                    
                    plt.rcParams['errorbar.capsize'] = 5

                    fig = plt.figure(figsize=(8, 6))
                    fig, ax1 = plt.subplots(figsize=(8, 6))
                    plt.style.use('seaborn-v0_8-darkgrid')

                    plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
                    width = 0.2
                    ax1.bar(x_axis-width, DQ1, width=2*width, label="1º Outer loop")#, color='navy')
                    ax1.bar(x_axis+width, DQ2, width=2*width, label="2º Outer loop")#, color='salmon')
                    ax1.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                               fancybox=True, shadow=True, frameon=True, ncol=2, prop={"size": 10})
                    ax1.set_xlabel('Flags', fontsize=10)
                    plt.title(date_title, loc='right', fontsize=10)
                    #plt.title(instrument_title, loc='left', fontsize=9)
                    plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                                 color='lightgray', fontweight='bold', fontsize='12', horizontalalignment='left', verticalalignment='center')
                    
                    ax1.set_ylabel('DQ = $\Delta$('+LabelObs+')/$\Delta$('+Label+')', color='black', fontsize=12)
                    ax1.tick_params('y', colors='black')
                    plt.xticks(x_axis, x_Flags)
                    plt.tight_layout()
                    plt.savefig('DQ_'+str(LabelObs) +'_'+str(Label) + 'Flag_' + str(Flag) + '_' + datefmt + '.png', 
                                bbox_inches='tight', dpi=100)
                
            else:
                try:
                    nloop1 = 0
                    nloop2 = 0

                    df1_first, df2_first, df1_last, df2_last = 0, 0, 0, 0
                    dfo1_first, dfo2_first, dfo1_last, dfo2_last = 0, 0, 0, 0


                    nloop1 = self1[f][0].iloc[-1, 0]  # ultimo valor na coluna iteração
                    nloop2 = self1[f][1].iloc[-1, 0]  # ultimo valor na coluna iteração

                    df1_first = self1[f][0].loc[idx[0], idx[Label]]  # 1º outer loop
                    df2_first = self1[f][1].loc[idx[0], idx[Label]]  # 2º outer loop
                    df1_last  = self1[f][0].loc[idx[nloop1], idx[Label]]  # 1º outer loop
                    df2_last  = self1[f][1].loc[idx[nloop2], idx[Label]]  # 2º outer loop

                    dfo1_first = self2[f][0].loc[idx[0,[LabelObs]], idx[Flag]].values  # 1º outer loop
                    dfo2_first = self2[f][1].loc[idx[0,[LabelObs]], idx[Flag]].values  # 2º outer loop
                    dfo1_last  = self2[f][0].loc[idx[nloop1,[LabelObs]], idx[Flag]].values  # 1º outer loop
                    dfo2_last  = self2[f][1].loc[idx[nloop2,[LabelObs]], idx[Flag]].values  # 2º outer loop



                    Q1_first.append( (dfo1_first[0]/df1_first)*100 )
                    Q2_first.append( (dfo2_first[0]/df2_first)*100 )

                    Q1_last.append( (dfo1_last[0]/df1_last)*100 )
                    Q2_last.append( (dfo2_last[0]/df2_last)*100 )

                    if abs(df1_last - df1_first) > 10e-06:
                        DQ1.append( ( (dfo1_last[0] - dfo1_first[0]) / (df1_last - df1_first) ) )
                    else:
                        DQ1.append(-99)

                    if abs(df2_last - df2_first) > 10e-06:
                        DQ2.append( ( (dfo2_last[0] - dfo2_first[0]) / (df2_last - df2_first) ) )
                    else:
                        DQ1.append(-99)


                except:
                    print("++++++++++++++++++++++++++ ERROR: file reading --> fort220_plot_cost_gradient ++++++++++++++++++++++++++")
                    print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                
                
            f = f + 1
            date = date + timedelta(hours=int(nHour))
            date_finale = date
        
        
        if Flag != None:
            
            if(len(DayHour_tmp) > 4):
                DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
            else:
                DayHour = DayHour_tmp

            x_axis      = np.arange(0, len(DayHour), 1)
            date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))

            #-------------------- 1ª figura ------------------------------------------#
            fig = plt.figure(figsize=(10, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-darkgrid')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            forplot = 'Flag = '+Flag

            ax1.plot(x_axis, Q1_first, "-o", label="1º Outer loop: Inner [0]")
            ax1.plot(x_axis, Q1_last, "-o", label="1º Outer loop: Inner ["+str(nloop1)+"]")
            ax1.plot(x_axis, Q2_first, "-o", label="2º Outer loop: Inner [0]")
            ax1.plot(x_axis, Q2_last, "-o", label="2º Outer loop: Inner ["+str(nloop2)+"]")
            ax1.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                       fancybox=True, shadow=True, frameon=True, ncol=2, prop={"size": 10})
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                         color='lightgray', fontweight='bold', fontsize='12', horizontalalignment='left', verticalalignment='center')

            ax1.set_ylim(np.round(-0.05*np.max([Q1_first,Q1_last,Q2_first,Q2_last])), 
                         np.round(1.05*np.max([Q1_first,Q1_last,Q2_first,Q2_last])))
            ax1.set_ylabel('Q = '+LabelObs+'/'+Label+' (%)', color='black', fontsize=12)
            ax1.tick_params('y', colors='black')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(Q1_first),ls='dotted',c='lightgray')
            plt.axhline(y=np.mean(Q1_last),ls='dotted',c='lightgray')
            plt.axhline(y=np.mean(Q2_first),ls='dotted',c='lightgray')
            plt.axhline(y=np.mean(Q2_last),ls='dotted',c='lightgray')
            plt.tight_layout()
            plt.savefig('time_series_Q_'+str(LabelObs) +'_'+str(Label) + 'Flag_' + str(Flag) + '_'+'Innerloop.png', 
                        bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()


            #-------------------- 2ª figura ------------------------------------------#
            fig = plt.figure(figsize=(10, 4))
            #fig = plt.figure(figsize=(12, 6))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-darkgrid')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            forplot = 'Flag = '+Flag

            ax1.plot(x_axis, DQ1, "-o", label="1º Outer loop")
            ax1.plot(x_axis, DQ2, "-o", label="2º Outer loop")
            ax1.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                       fancybox=True, shadow=True, frameon=True, ncol=2, prop={"size": 10})
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                         color='lightgray', fontweight='bold', fontsize='12', horizontalalignment='left', verticalalignment='center')
            
            ax1.set_ylabel('DQ = $\Delta$('+LabelObs+')/$\Delta$('+Label+')', color='black', fontsize=12)
            ax1.tick_params('y', colors='black')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(DQ1),ls='dotted',c='lightgray')
            plt.axhline(y=np.mean(DQ2),ls='dotted',c='lightgray')
            plt.tight_layout()
            plt.savefig('time_series_DQ_'+str(LabelObs) +'_'+str(Label) + 'Flag_' + str(Flag) + '.png', 
                        bbox_inches='tight', dpi=100)
        
        
        

    def time_series_fort220(self, OuterLoop=1, Label=None, Flag=None, dateIni=None, dateFin=None, nHour="06", Clean=None):
        
        '''
        The time_series fort220 function plots a time series for fort.220 data in different flag of the contribution term. A Hovmoller diagram is return if Flag=None or Flag is a list.

        '''
        if Clean == None:
            Clean = True

        delta = nHour
        
        idx = pd.IndexSlice

        separator = " ============================================================================================================="

        print()
        print(separator)
            
        zflags_all = []
        n_flag = len(self[0][OuterLoop-1].loc[idx[0,[Label]], idx[:]].columns) - 1   # Qntdd de flags
        [zflags_all.append(int(i)) for i in range(1,n_flag+1,1)]

        if type(Flag) == list:
            zflag = Flag
            flagList = 1
            zflags_def = zflag
        elif Flag == None:
            zflag = zflags_all       #list of all flags 
            flagList = 0
            zflags_def = zflag
        else:
            zflag = Flag
            flagList = 0 
            zflags_def = zflags_all  #list of all flags 

            
        print(zflag,flagList)
        print('')
        print('Outer Loop: ',OuterLoop,' Label: ',Label)

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        levs_tmp, DayHour_tmp = [], []
        info_check = {}
        f = 0
        
        while (date <= datef):
            
            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            # Try: For issues reading the file (file not found), 
            # in the except statement an error message is printed and continues for other dates
            try:
                nloop = self[f][OuterLoop-1].iloc[-1, 0]
                dataDict = self[f][OuterLoop-1].loc[idx[nloop,[Label]], idx[:]]
                print('nloop = ',nloop)
                
                if (Flag == None or flagList == 1):
                    levs_tmp = zflags_def[::-1]
                    print(date.strftime(' Preparing data for: observation flags' + "%Y-%m-%d:%H"))
                    print(' Flags: ', sorted(levs_tmp), end='\n')
                    print("")
                    f = f + 1
                else:
                    if (Flag != None and flagList != 1):
                        levs_tmp.extend([zflag])
                        print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"), ' - Flag: ', zflag , end='\n')
                        f = f + 1
                    else:
                        print(date.strftime(setcolor.WARNING + ' Preparing data for: ' + "%Y-%m-%d:%H"), ' - No information on this date ' + setcolor.ENDC, end='\n')
                
                del(dataDict)
                
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> time_series_radi ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                print("")
                f = f + 1
            
            date = date + timedelta(hours=int(delta))
            
        if(len(DayHour_tmp) > 4):
            DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
        else:
            DayHour = DayHour_tmp
        
        zlevs = [z if z in zflags_def else "" for z in sorted(set(levs_tmp+zflags_def))]

        print()
        print(separator)
        print()

        list_dataByLevs = []
        date = datei
        levs = sorted(list(set(levs_tmp)))
        levs_tmp.clear()
        del(levs_tmp[:])
        
        print('flags = ',levs)

        
        f = 0
        while (date <= datef):

            print(date.strftime(' Calculating for ' + "%Y-%m-%d:%H"))
            datefmt = date.strftime("%Y%m%d%H")


            try:
                nloop = self[f][OuterLoop-1].iloc[-1, 0]
                dataDict = self[f][OuterLoop-1].loc[idx[nloop,[Label]], idx[:]]
                dataByLevs, value_dataByLevs = {}, {}
                [dataByLevs.update({int(lvl): []}) for lvl in levs]
                if Flag != None and flagList != 1: 
                    forplot = 'Flag ='+str(zflag)
                    forplotname = 'Flag_'+str(zflag)
                    [ dataByLevs[int(zflag)].append(v) for v in self[f][OuterLoop-1].loc[idx[nloop,[Label]], idx[str(zflag)]] ]   
                else:
                    for ll in range(len(levs)):
                        lv = levs[ll]
                        cutlevs = [ v for v in self[f][OuterLoop-1].loc[idx[nloop,[Label]], idx[str(lv)]] if v != 0.0 ]
                        forplotname = 'List_Flags'
                        [ dataByLevs[lv].append(il) for il in cutlevs ]
                        cutlevs.clear()
                f = f + 1
                for lv in levs:
                    if len(dataByLevs[lv]) != 0:
                        value_dataByLevs.update({int(lv): dataByLevs[lv][0]})
                    else:
                        value_dataByLevs.update({int(lv): -99})
            
            except:
                dataByLevs, value_dataByLevs, mean_dataByLevs, std_dataByLevs, count_dataByLevs = {}, {}, {}, {}, {}
                f = f + 1 # Estava faltando: sem essa atualização o dataDict do próximo UTC não é concatenado corretamente
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)

                for lv in levs:
                    value_dataByLevs.update({int(lv): -99})
            
            if Flag == None or flagList == 1:
                list_dataByLevs.append(list(reversed(value_dataByLevs.values())))
            else:
                list_dataByLevs.append(value_dataByLevs[int(zflag)])
            
            dataByLevs.clear()
            value_dataByLevs.clear()

            date_finale = date
            date = date + timedelta(hours=int(delta))

        
        print()
        print(separator)
        print()

        print(' Making Graphics...')

        y_axis      = np.arange(0, len(zlevs), 1)
        x_axis      = np.arange(0, len(DayHour), 1)

        data_final  = np.ma.masked_array(np.array(list_dataByLevs), np.array(list_dataByLevs) == -99)

        date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
        instrument_title = str(OuterLoop) + 'º outer loop'

        # Figure with more than one Flag - default all Flags
        if Flag == None or flagList == 1:
            fig = plt.figure(figsize=(6, 10))
            plt.rcParams['axes.facecolor'] = 'None'
            plt.rcParams['hatch.linewidth'] = 0.3

            plt.subplot(3, 1, 1)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(data_final.T), origin='lower', cmap='nipy_spectral', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Flags')
            plt.xlabel(Label+' (Inner loop '+str(nloop)+')', labelpad=60)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)

            plt.tight_layout()
            if flagList == 1:
                plt.savefig('hovmoller_Label_'+Label +'_'+'OuterL'+str(OuterLoop)+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            else:
                plt.savefig('hovmoller_Label_'+Label + '_'+'OuterL'+str(OuterLoop)+'_all.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()
                
            
        # Figure with only one flag
        else:

            fig = plt.figure(figsize=(6, 4))
            fig, ax1 = plt.subplots(1, 1)
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', color='lightgray', fontweight='bold', fontsize='12',
            horizontalalignment='left', verticalalignment='center')

            
            ax1.plot(x_axis, list_dataByLevs, "b-", label=Label+' (Inner loop '+str(nloop)+')')
            ax1.plot(x_axis, list_dataByLevs, "bo", label=Label+' (Inner loop '+str(nloop)+')')
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylabel(Label, color='b', fontsize=10)
            ax1.tick_params('y', colors='b')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_dataByLevs),ls='dotted',c='blue')
            
            ax1.set_title(date_title, loc='right', fontsize=10)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_Label_'+Label +'_'+'OuterL'+str(OuterLoop)+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()
                
            
        # Cleaning up
        if Clean:
            plt.close('all')

        print(' Done!')
        print()
        
               

        return


    def Mean_std_fort220(self, OuterLoop=None, cost_gradient=True, Label=None, Flag=None, dateIni=None, dateFin=None, nHour="06", Clean=None):
        
        '''
        This function plots a graph of the mean values and standard deviations of the variable `Label` calculated over the time interval and for each inner loop. At the end the function returns a figure where the points are the mean values and the standard deviation is represented in the form of a vertical bar, indicating for each inner loop the variability of the data over the time interval.

        '''
        if Clean == None:
            Clean = True

        delta = nHour
        
        idx = pd.IndexSlice

        separator = " ============================================================================================================="

        print()
        print(separator)
        
            
        zloops_all = []
        nloop_all = self[0][1].iloc[-1, 0]   # Qntdd de inner loops
        [zloops_all.append(int(i)) for i in range(0,nloop_all+1,1)]
        zloops_def = zloops_all    
        
        print('zloops_def = ',zloops_def)
        print('')
        print('Outer Loop: ',OuterLoop,', Label: ',Label,', Flag: ',Flag)

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        levs_tmp, DayHour_tmp = [], []
        InnerLoop1, InnerLoop2 = [], []
        
        f = 0
        if OuterLoop == None:
            
            if cost_gradient:
                InnerLoop1 = self[f][0].loc[idx[:], idx['InnerLoop']]
                InnerLoop2 = self[f][1].loc[idx[:], idx['InnerLoop']]
            else:
                InnerLoop1 = self[f][0].loc[idx[:,[Label]], idx['InnerLoop']]
                InnerLoop2 = self[f][1].loc[idx[:,[Label]], idx['InnerLoop']]

            nloop1 = self[f][0].iloc[-1, 0]
            nloop2 = self[f][1].iloc[-1, 0]
        
            print('')
            print('OuterLoop = ',OuterLoop,': nloop1 = ',nloop1, ' nloop2 = ',nloop2)
            
        else:
            if cost_gradient:
                InnerLoop1 = self[f][OuterLoop-1].loc[idx[:], idx['InnerLoop']]
            else:
                InnerLoop1 = self[f][OuterLoop-1].loc[idx[:,[Label]], idx['InnerLoop']]

            nloop1 = self[f][OuterLoop-1].iloc[-1, 0]
            nloop2 = 0
        
            print('')
            print('OuterLoop = ',OuterLoop,': nloop = ',nloop1, ' nloop2 = ',nloop2)
        
        if OuterLoop == None:
            OutL = 1
        else:
            OutL = OuterLoop
        
        
        while (date <= datef):
            
            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            # Try: For issues reading the file (file not found), 
            # in the except statement an error message is printed and continues for other dates
            try:
                if cost_gradient:
                    dataDict = self[f][OutL-1].loc[idx[:], idx[:]]
                    if 'InnerLoop' in dataDict and Label in dataDict:
                        print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"), end='\n')
                        f = f + 1
                    else:
                        print(date.strftime(setcolor.WARNING + ' Preparing data for: ' + "%Y-%m-%d:%H"), ' - No information on this date ' + setcolor.ENDC, end='\n')
                    
                else:
                    dataDict = self[f][OutL-1].loc[idx[:,idx[:]], idx[:]]
                    if 'InnerLoop' in dataDict and Flag in dataDict:
                        print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"), end='\n')
                        f = f + 1
                    else:
                        print(date.strftime(setcolor.WARNING + ' Preparing data for: ' + "%Y-%m-%d:%H"), ' - No information on this date ' + setcolor.ENDC, end='\n')
                
                del(dataDict)
                
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> time_series_radi ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                print("")
                f = f + 1
            
            date_finale = date
            date = date + timedelta(hours=int(delta))
        
        DayHour = DayHour_tmp
        
        list_dataByLevs, list_meanByLevs, list_stdByLevs = [], [], []
        date = datei
        levs = DayHour
        
        print()
        print(separator)
        print('levs DayHour = ',levs)
        print('len levs DayHour = ',len(levs))
        print()
        print(separator)

        if OuterLoop == None:
            forplot = 'nloop1_'+str(nloop1)+'_nloop2_'+str(nloop2)
            OutL = 1
        else:
            forplot = 'nloop_'+str(nloop1)
            OutL = OuterLoop
        
        inl = 0
        while (inl <= nloop1):

            print('Outer Loop '+ str(OutL) +' Calculating for inner loop ' + str(inl))

            try: 
                
                value_dataByLevs, mean_dataByLevs, std_dataByLevs = {}, {}, {}
                dataByLevs = []
                
                if len(levs)> 1:
                    
                    if cost_gradient:
                        for ll in range(len(levs)):
                            v = self[ll][OutL-1].loc[idx[inl], idx[Label]]
                            dataByLevs.append(v)
                            forplotname = Label

                    else:
                        for ll in range(len(levs)):
                            v = self[ll][OutL-1].loc[idx[inl,[Label]], idx[Flag]].values
                            dataByLevs.append(v)
                            forplotname = Label + '_Flag' + Flag
                    
                else:
                    print(setcolor.WARNING + ' len dates is <= 1 ' + setcolor.ENDC, end='\n')
                
                if len(dataByLevs) != 0:
                    mean_dataByLevs.update({int(inl): np.mean(np.array(dataByLevs))})
                    std_dataByLevs.update({int(inl): np.std(np.array(dataByLevs))})
                else:
                    mean_dataByLevs.update({int(inl): -99})
                    std_dataByLevs.update({int(inl):-99})
                
            
            except:
                dataByLevs, mean_dataByLevs, std_dataByLevs = {}, {}, {}
                print(setcolor.WARNING + "    >>> No information on this inner loop (" + str(inl) +") <<< " + setcolor.ENDC)
                
                mean_dataByLevs.update({int(inl): -99})
                std_dataByLevs.update({int(inl):-99})
            
            
            list_meanByLevs.append(mean_dataByLevs[int(inl)])
            list_stdByLevs.append(std_dataByLevs[int(inl)])
            
            dataByLevs.clear()
            mean_dataByLevs.clear()
            std_dataByLevs.clear()

            
            inl = inl + 1

        
        print()
        print(separator)
        print()
        
        
        if OuterLoop == None:
            OutL = 2
            list2_dataByLevs, list2_meanByLevs, list2_stdByLevs = [], [], []
            
            inl = 0
            while (inl <= nloop2):

                print('Outer Loop '+ str(OutL) +' Calculating for inner loop ' + str(inl))

                try: 

                    value_dataByLevs, mean_dataByLevs, std_dataByLevs = {}, {}, {}
                    dataByLevs = []

                    if len(levs)> 1:

                        if cost_gradient:
                            for ll in range(len(levs)):
                                v = self[ll][OutL-1].loc[idx[inl], idx[Label]]
                                dataByLevs.append(v)
                                forplotname = Label

                        else:
                            for ll in range(len(levs)):
                                v = self[ll][OutL-1].loc[idx[inl,[Label]], idx[Flag]].values
                                dataByLevs.append(v)
                                forplotname = Label + '_Flag' + Flag
                    else:
                        print(setcolor.WARNING + ' len dates is <= 1 ' + setcolor.ENDC, end='\n')

                    if len(dataByLevs) != 0:
                        mean_dataByLevs.update({int(inl): np.mean(np.array(dataByLevs))})
                        std_dataByLevs.update({int(inl): np.std(np.array(dataByLevs))})
                    else:
                        mean_dataByLevs.update({int(inl): -99})
                        std_dataByLevs.update({int(inl):-99})


                except:
                    dataByLevs, mean_dataByLevs, std_dataByLevs = {}, {}, {}
                    print(setcolor.WARNING + "    >>> No information on this inner loop (" + str(inl) +") <<< " + setcolor.ENDC)

                    mean_dataByLevs.update({int(inl): -99})
                    std_dataByLevs.update({int(inl):-99})


                list2_meanByLevs.append(mean_dataByLevs[int(inl)])
                list2_stdByLevs.append(std_dataByLevs[int(inl)])

                dataByLevs.clear()
                mean_dataByLevs.clear()
                std_dataByLevs.clear()


                inl = inl + 1
                
            print()
            print(separator)
            print()

        print(' Making Graphics...')

        if OuterLoop == None:
            x_axis      = np.arange(0, len(InnerLoop1), 1)
            x_axis2     = np.arange(0, len(InnerLoop2), 1)
            
            mean_final  = np.ma.masked_array(np.array(list_meanByLevs), np.array(list_meanByLevs) == -99)
            std_final   = np.ma.masked_array(np.array(list_stdByLevs), np.array(list_stdByLevs) == -99)
            
            mean_final_l2  = np.ma.masked_array(np.array(list2_meanByLevs), np.array(list2_meanByLevs) == -99)
            std_final_l2   = np.ma.masked_array(np.array(list2_stdByLevs), np.array(list2_stdByLevs) == -99)
            nloop = nloop2

            #min_y = np.min([np.min(list_meanByLevs)-1.1*np.min(list_stdByLevs), np.min(list2_meanByLevs)-1.1*np.min(list2_stdByLevs)])
            #max_y = np.max([np.max(list_meanByLevs)+1.1*np.max(list_stdByLevs), np.max(list2_meanByLevs)+1.1*np.max(list2_stdByLevs)])
            
            instrument_title = '1º and 2º outer loops' + '  |  ' + ' Max inner loop = ' + str(nloop)
            Label_leg = '1º outer loop'
            
        else:
            x_axis      = np.arange(0, len(InnerLoop1), 1)
            
            mean_final  = np.ma.masked_array(np.array(list_meanByLevs), np.array(list_meanByLevs) == -99)
            std_final   = np.ma.masked_array(np.array(list_stdByLevs), np.array(list_stdByLevs) == -99)
            nloop = nloop1

            #min_y = np.min(list_meanByLevs)-1.1*np.min(list_stdByLevs)
            #max_y = np.max(list_meanByLevs)+1.1*np.max(list_stdByLevs)
            
            instrument_title = str(OuterLoop) + 'º outer loop' + '  |  ' + ' Max inner loop = ' + str(nloop)
            Label_leg = str(OuterLoop) + 'º outer loop'

        date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
        
        plt.rcParams['errorbar.capsize'] = 5
        
        fig = plt.figure(figsize=(10, 6))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        plt.style.use('seaborn-v0_8-darkgrid')

        #ax1.set_yscale("log") #, nonposy ='clip')
        if cost_gradient == False:
            plt.annotate('Flag '+Flag, xy=(0.45, 1.015), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', 
                         color='gray', fontweight='bold', fontsize='12', horizontalalignment='left', verticalalignment='center')

        
        ax1.errorbar(x_axis, mean_final, color='navy', yerr=std_final, fmt='-o', label=Label_leg, ms=4, capsize=5, ecolor='salmon')
        ax1.set_xlim(min(x_axis)-1, max(x_axis)+1)
        if OuterLoop == None:
            ax1.errorbar(x_axis2, mean_final_l2, color='forestgreen', yerr=std_final_l2, fmt='--s', label='2º outer loop', ms=4, 
                         capsize=5, ecolor='slategrey')
            ax1.set_xlim(min(x_axis2)-1, max(x_axis2)+1)
        # Make the y-axis label, ticks and tick labels match the line color.
        #ax1.set_ylim(bottom = min_y, top = max_y) 
        ax1.set_xlabel('Inner loop')
        ax1.set_ylabel('Mean ('+Label+')', color='black', fontsize=12)
        ax1.tick_params('y', colors='black')
        ax1.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   fancybox=True, shadow=True, frameon=True, ncol=2, prop={"size": 10})
        ax1.grid(True)
        
        plt.title(instrument_title, loc='left', fontsize=9)
        plt.title(date_title, loc='right', fontsize=9)
        plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
        plt.tight_layout()
        plt.savefig('Mean'+'_'+'OuterL'+str(OuterLoop)+'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
        if Clean:
            plt.clf()

        print(' Done!')
        print()
        
               

        return

# -------------------------- Fort 207 (radiancia) ---------------------#
    def time_series_fort207(self, Type=None, Sat=None, it=None, dateIni=None, dateFin=None, nHour="06", mask=None, channel=None, Clean=None):
        
        '''
        The time_series_fort207 function plots a time series for fort.207 data in different channel of the contribution term. A Hovmoller diagram is return if channel=None or channel is a list.

        '''
        if Clean == None:
            Clean = True

        delta = nHour
        
        idx = pd.IndexSlice

        separator = " ============================================================================================================="

        print()
        print(separator)
        
        varName = Type
        tab = 1
            
        zchans_all = []
        if varName == 'amsua':
            zchans_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] #list of all channels of the amsua sensor
        if varName == 'hirs4':
            zchans_all = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] #list of all channels of the hirs/4 sensor
        
        
        if type(channel) == list:
            zchan = channel
            chanList = 1
            zchans_def = zchan
        elif channel == None:
            zchan = zchans_all       #list of all flags 
            chanList = 0
            zchans_def = zchan
        else:
            zchan = channel
            chanList = 0 
            zchans_def = zchans_all  #list of all flags 

            
        print(zchan,chanList)
        print('')
        print('mask = ',mask)
        print('')
        print('Stage it: ',it)
        print('')
        print('Table: ',tab)
        print('')

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        levs_tmp, DayHour_tmp = [], []
        info_check = {}
        f = 0
        
        while (date <= datef):
            
            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
            # Try: For issues reading the file (file not found), 
            # in the except statement an error message is printed and continues for other dates
            try:
                dataDict = self[f][tab].query(mask).loc[idx[:], idx['nchan']]
                
                if (channel == None or chanList == 1):
                    levs_tmp = zchans_def[::-1]
                    print(date.strftime(' Preparing data for: channels' + "%Y-%m-%d:%H"))
                    print(' channels: ', sorted(levs_tmp), end='\n')
                    print("")
                    f = f + 1
                else:
                    if (channel != None and chanList != 1):
                        levs_tmp.extend([zchan])
                        print(date.strftime(' Preparing data for: ' + "%Y-%m-%d:%H"), ' - channel: ', zchan , end='\n')
                        f = f + 1
                    else:
                        print(date.strftime(setcolor.WARNING + ' Preparing data for: ' + "%Y-%m-%d:%H"), ' - No information on this date ' + setcolor.ENDC, end='\n')
                
                del(dataDict)
                
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> time_series_fort207 ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
                print("")
                f = f + 1
            
            date = date + timedelta(hours=int(delta))
            
        if(len(DayHour_tmp) > 4):
            DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
        else:
            DayHour = DayHour_tmp
        
        zlevs = [z if z in zchans_def else "" for z in sorted(set(levs_tmp+zchans_def))]

        print()
        print(separator)
        print()

        date = datei
        levs = sorted(list(set(levs_tmp)))
        levs_tmp.clear()
        del(levs_tmp[:])
        
        print('channels = ',levs)
        
        list_dataObsU, list_dataObsT = [], []
        list_dataVarCH, list_dataPnltCH, list_dataSqrt, list_dataSTD = [], [], [], []
        list_dataBiasBC, list_dataBiasAC = [], []
        
        f = 0
        while (date <= datef):

            print(date.strftime(' Calculating for ' + "%Y-%m-%d:%H"))
            datefmt = date.strftime("%Y%m%d%H")
            
            maski = "(it=="+str(it)+")"
            try:
                
                dataDict = self[f][tab].query(mask)
                dataObsU, value_dataObsU, dataObsT, value_dataObsT = {}, {}, {}, {}
                dataVarCH, value_dataVarCH = {}, {}
                dataPnltCH, value_dataPnltCH = {}, {}
                dataSqrt, value_dataSqrt = {}, {}
                dataSTD, value_dataSTD = {}, {}
                dataBiasBC, value_dataBiasBC, dataBiasAC, value_dataBiasAC = {}, {}, {}, {}
                [dataObsU.update({int(lvl): []}) for lvl in levs]
                [dataObsT.update({int(lvl): []}) for lvl in levs]
                [dataVarCH.update({int(lvl): []}) for lvl in levs]
                [dataBiasBC.update({int(lvl): []}) for lvl in levs]
                [dataBiasAC.update({int(lvl): []}) for lvl in levs]
                [dataPnltCH.update({int(lvl): []}) for lvl in levs]
                [dataSqrt.update({int(lvl): []}) for lvl in levs]
                [dataSTD.update({int(lvl): []}) for lvl in levs]
                #print('Passou 1')
                if channel != None and chanList != 1: 
                    forplot = 'Channel ='+str(zchan)
                    forplotname = 'Channel_'+str(zchan)
                    [ dataObsU[int(zchan)].append(v) for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).nobsused) if int(p) == zchan ]
                    [ dataObsT[int(zchan)].append(v) for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).nobstossed) if int(p) == zchan ]
                    [ dataVarCH[int(zchan)].append(v) for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).varCH) if int(p) == zchan ]
                    [ dataBiasBC[int(zchan)].append(v) for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).biasBC) if int(p) == zchan ]
                    [ dataBiasAC[int(zchan)].append(v) for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).biasAC) if int(p) == zchan ]
                    [ dataPnltCH[int(zchan)].append(v) for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).penaltyCH) if int(p) == zchan ]
                    [ dataSqrt[int(zchan)].append(v) for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).sqrt) if int(p) == zchan ]
                    [ dataSTD[int(zchan)].append(v) for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).STD) if int(p) == zchan ]
                else:
                    for ll in range(len(levs)):
                        lv = levs[ll]
                        cutlevs0 = [ v for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).nobsused) if int(p) == lv ]
                        cutlevs1 = [ v for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).nobstossed) if int(p) == lv ]
                        cutlevs2 = [ v for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).varCH) if int(p) == lv ]
                        cutlevs3 = [ v for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).biasBC) if int(p) == lv ]
                        cutlevs4 = [ v for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).biasAC) if int(p) == lv ]
                        cutlevs5 = [ v for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).penaltyCH) if int(p) == lv ]
                        cutlevs6 = [ v for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).sqrt) if int(p) == lv ]
                        cutlevs7 = [ v for p,v in zip(dataDict.query(maski).nchan, dataDict.query(maski).STD) if int(p) == lv ]
                        forplotname = 'List_Channels'
                        [ dataObsU[lv].append(il) for il in cutlevs0 ]
                        [ dataObsT[lv].append(il) for il in cutlevs1 ]
                        [ dataVarCH[lv].append(il) for il in cutlevs2 ]
                        [ dataBiasBC[lv].append(il) for il in cutlevs3 ]
                        [ dataBiasAC[lv].append(il) for il in cutlevs4 ]
                        [ dataPnltCH[lv].append(il) for il in cutlevs5 ]
                        [ dataSqrt[lv].append(il) for il in cutlevs6 ]
                        [ dataSTD[lv].append(il) for il in cutlevs7 ]
                        cutlevs0.clear()
                        cutlevs1.clear()
                        cutlevs2.clear()
                        cutlevs3.clear()
                        cutlevs4.clear()
                        cutlevs5.clear()
                        cutlevs6.clear()
                        cutlevs7.clear()
                f = f + 1
                for lv in levs:
                    if len(dataObsU[lv]) != 0 or len(dataObsT[lv]) != 0:
                        value_dataObsU.update({int(lv): dataObsU[lv][0]})
                        value_dataObsT.update({int(lv): dataObsT[lv][0]})
                    else:
                        value_dataObsU.update({int(lv): -99})
                        value_dataObsT.update({int(lv): -99})
                        
                    if len(dataVarCH[lv]) != 0 or len(dataPnltCH[lv]) != 0:
                        value_dataVarCH.update({int(lv): dataVarCH[lv][0]})
                        value_dataPnltCH.update({int(lv): dataPnltCH[lv][0]})
                    else:
                        value_dataVarCH.update({int(lv): -99})
                        value_dataPnltCH.update({int(lv): -99})
                        
                    if len(dataBiasBC[lv]) != 0 or len(dataBiasAC[lv]) != 0:
                        value_dataBiasBC.update({int(lv): dataBiasBC[lv][0]})
                        value_dataBiasAC.update({int(lv): dataBiasAC[lv][0]})
                    else:
                        value_dataBiasBC.update({int(lv): -99})
                        value_dataBiasAC.update({int(lv): -99})
                        
                    if len(dataSqrt[lv]) != 0 or len(dataSTD[lv]) != 0:
                        value_dataSqrt.update({int(lv): dataSqrt[lv][0]})
                        value_dataSTD.update({int(lv): dataSTD[lv][0]})
                    else:
                        value_dataSqrt.update({int(lv): -99})
                        value_dataSTD.update({int(lv): -99})
                        
            
            except:
                dataObsU, value_dataObsU, dataObsT, value_dataObsT = {}, {}, {}, {}
                dataVarCH, value_dataVarCH = {}, {}
                dataPnltCH, value_dataPnltCH = {}, {}
                dataSqrt, value_dataSqrt = {}, {}
                dataSTD, value_dataSTD = {}, {}
                dataBiasBC, value_dataBiasBC, dataBiasAC, value_dataBiasAC = {}, {}, {}, {}
                f = f + 1 # Estava faltando: sem essa atualização o dataDict do próximo UTC não é concatenado corretamente
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)

                for lv in levs:
                    value_dataObsU.update({int(lv): -99})
                    value_dataObsT.update({int(lv): -99})
                    value_dataVarCH.update({int(lv): -99})
                    value_dataBiasBC.update({int(lv): -99})
                    value_dataBiasAC.update({int(lv): -99})
                    value_dataPnltCH.update({int(lv): -99})
                    value_dataSqrt.update({int(lv): -99})
                    value_dataSTD.update({int(lv): -99})
            
            if channel == None or chanList == 1:
                list_dataObsU.append(list(reversed(value_dataObsU.values())))
                list_dataObsT.append(list(reversed(value_dataObsT.values())))
                list_dataVarCH.append(list(reversed(value_dataVarCH.values())))
                list_dataBiasBC.append(list(reversed(value_dataBiasBC.values())))
                list_dataBiasAC.append(list(reversed(value_dataBiasAC.values())))
                
                list_dataPnltCH.append(list(reversed(value_dataPnltCH.values())))
                list_dataSqrt.append(list(reversed(value_dataSqrt.values())))
                list_dataSTD.append(list(reversed(value_dataSTD.values())))
            else:
                list_dataObsU.append(value_dataObsU[int(zchan)])
                list_dataObsT.append(value_dataObsT[int(zchan)])
                list_dataVarCH.append(value_dataVarCH[int(zchan)])
                list_dataBiasBC.append(value_dataBiasBC[int(zchan)])
                list_dataBiasAC.append(value_dataBiasAC[int(zchan)])
                
                list_dataPnltCH.append(value_dataPnltCH[int(zchan)])
                list_dataSqrt.append(value_dataSqrt[int(zchan)])
                list_dataSTD.append(value_dataSTD[int(zchan)])
            
            dataObsU.clear()
            dataObsT.clear()
            dataVarCH.clear()
            dataBiasBC.clear()
            dataBiasAC.clear()
            value_dataObsU.clear()
            value_dataObsT.clear()
            value_dataVarCH.clear()
            value_dataBiasBC.clear()
            value_dataBiasAC.clear()
            
            dataPnltCH.clear()
            dataSqrt.clear()
            dataSTD.clear()

            date_finale = date
            date = date + timedelta(hours=int(delta))

        
        print()
        print(separator)
        print()

        print(' Making Graphics...')

        y_axis      = np.arange(0, len(zlevs), 1)
        x_axis      = np.arange(0, len(DayHour), 1)

        dataObsU_final  = np.ma.masked_array(np.array(list_dataObsU), np.array(list_dataObsU) == -99)
        dataObsT_final  = np.ma.masked_array(np.array(list_dataObsT), np.array(list_dataObsT) == -99)
        dataVarCH_final  = np.ma.masked_array(np.array(list_dataVarCH), np.array(list_dataVarCH) == -99)
        dataBiasBC_final  = np.ma.masked_array(np.array(list_dataBiasBC), np.array(list_dataBiasBC) == -99)
        dataBiasAC_final  = np.ma.masked_array(np.array(list_dataBiasAC), np.array(list_dataBiasAC) == -99)
        
        dataPnltCH_final  = np.ma.masked_array(np.array(list_dataPnltCH), np.array(list_dataPnltCH) == -99)
        dataSqrt_final  = np.ma.masked_array(np.array(list_dataSqrt), np.array(list_dataSqrt) == -99)
        dataSTD_final  = np.ma.masked_array(np.array(list_dataSTD), np.array(list_dataSTD) == -99)
        
        vmaxnobs = np.max(np.array([np.max(dataObsU_final), np.max(dataObsT_final)]))
        
        vminVarCH = 0.9*np.min(dataVarCH_final)
        vmaxVarCH = 1.1*np.max(dataVarCH_final)
        
        minBias = np.min(np.array([np.min(dataBiasBC_final), np.min(dataBiasAC_final)]))
        maxBias = np.max(np.array([np.max(dataBiasBC_final), np.max(dataBiasAC_final)]))
        
        limitBias = np.max([np.abs(minBias),np.abs(maxBias)])
        
        #vminBias = 0.9*minBias
        #vmaxBias = 1.1*maxBias
        
        vminBias = -limitBias
        vmaxBias = limitBias

        date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
        instrument_title = Type + ' | ' + Sat + '- Stage 0'+ str(it)

        # Figure with more than one Flag - default all Flags
        if channel == None or chanList == 1:
            fig = plt.figure(figsize=(12, 10))
            plt.rcParams['axes.facecolor'] = 'None'
            plt.rcParams['hatch.linewidth'] = 0.3

            plt.subplot(3, 2, 1)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(dataObsU_final.T), origin='lower', vmin=0.0, vmax=vmaxnobs, cmap='gist_heat_r', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Number of observations used', labelpad=60)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)
            
            plt.subplot(3, 2, 2)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(dataObsT_final.T), origin='lower', vmin=0.0, vmax=vmaxnobs, cmap='gist_heat_r', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Number of observations tossed', labelpad=60)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)
            
            plt.subplot(3, 2, 3)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(dataBiasBC_final.T), origin='lower', vmin=vminBias, vmax=vmaxBias, cmap='seismic', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Observation-guess before bias correction', labelpad=60)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)
            
            plt.subplot(3, 2, 4)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(dataBiasAC_final.T), origin='lower', vmin=vminBias, vmax=vmaxBias, cmap='seismic', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Observation-guess after bias correction', labelpad=60)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)
            
            plt.subplot(3, 2, 5)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(dataPnltCH_final.T), origin='lower', cmap='YlGnBu', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('Penalty channel', labelpad=60)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)
            
            plt.subplot(3, 2, 6)
            ax = plt.gca()
            ax.add_patch(mpl.patches.Rectangle((-1,-1),(len(DayHour)+1),(len(levs)+3), hatch='xxxxx', color='black', fill=False, snap=False, zorder=0))
            plt.imshow(np.flipud(dataSTD_final.T), origin='lower', cmap='Blues', aspect='auto', zorder=1,interpolation='none')
            plt.colorbar(orientation='horizontal', pad=0.18, shrink=1.0)
            plt.tight_layout()
            plt.title(instrument_title, loc='left', fontsize=10)
            plt.title(date_title, loc='right', fontsize=10)
            plt.ylabel('Channels')
            plt.xlabel('STD', labelpad=60)
            plt.yticks(y_axis, zlevs)
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax.set_xticks(major_ticks)
            

            plt.tight_layout()
            if chanList == 1:
                plt.savefig('hovmoller_'+ Type +'_'+ Sat +'_it'+ str(it) +'_'+ forplotname+'.png', bbox_inches='tight', dpi=100)
            else:
                plt.savefig('hovmoller_'+ Type +'_'+ Sat +'_it'+ str(it) +'_all.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()
                
            
        # Figure with only one flag
        else:

            fig = plt.figure(figsize=(8, 4))
            fig, ax1 = plt.subplots(figsize=(8, 4))
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.45, 1.025), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', color='slategray', fontweight='bold', fontsize='10',
            horizontalalignment='left', verticalalignment='center')
            
            ax1.plot(x_axis, list_dataBiasBC, "-o", c='navy', label="bias (before bias correction)") #c='steelblue'
            ax1.plot(x_axis, list_dataBiasAC, "-s", c='tomato', label="bias (after bias correction)")
            ax1.fill_between(x_axis, np.array(list_dataBiasAC) - np.array(list_dataSTD), np.array(list_dataBiasAC) + np.array(list_dataSTD), label='Std Dev',  facecolor='tomato', alpha=0.2, zorder=2)
            
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(1.1*vminBias, 1.1*vmaxBias)
            ax1.set_ylabel('Bias', color='black', fontsize=10)
            ax1.tick_params('y', colors='black')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            plt.axhline(y=np.mean(list_dataBiasBC),ls='dotted',c='steelblue')
            plt.axhline(y=np.mean(list_dataBiasAC),ls='dotted',c='tomato')
            
            plt.legend(numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   fancybox=True, shadow=True, frameon=True, ncol=3, prop={"size": 10})
            
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_bias_'+ Type +'_'+ Sat +'_it'+ str(it) +'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()
            
            
            fig = plt.figure(figsize=(7, 4))
            fig, ax1 = plt.subplots(figsize=(7, 4))
            plt.style.use('seaborn-v0_8-ticks')

            plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
            plt.annotate(forplot, xy=(0.45, 1.025), xytext=(0,0), xycoords='axes fraction', textcoords='offset points', color='slategray', fontweight='bold', fontsize='10',
            horizontalalignment='left', verticalalignment='center')

            #ax1.plot(x_axis, list_dataBiasBC, "b-", label="bias (before bias correction)")
            p1 = ax1.plot(x_axis, list_dataObsU, "-", c='teal', label="nobs (used in GSI analysis)") #c='steelblue'
            label1 = "nobs (used in GSI analysis)"
            #ax1.plot(x_axis, list_dataObsU, "-", c='teal', label="nobs (used in GSI analysis)") #c='steelblue'
            #ax1.plot(x_axis, list_dataObsT, "--", c='red', label="nobs (tossed by gross check)")
            #ax1.plot(x_axis, np.array(list_dataObsU) - np.array(list_dataObsT), "-", c='red', label="nobs (used less tossed by gross check)")
            #ax1.fill_between(x_axis, np.array(list_dataBiasAC) - np.array(list_dataSTD), np.array(list_dataBiasAC) + np.array(list_dataSTD), label='Std Dev',  facecolor='tomato', alpha=0.2, zorder=2)
            #ax1.bar(x_axis, list_dataBiasAC, yerr=list_dataSTD, ms=4, capsize=5, ecolor='tomato')
            ax1.set_xlabel('Date (DayHour)', fontsize=10)
            # Make the y-axis label, ticks and tick labels match the line color.
            ax1.set_ylim(0.0, 1.1*vmaxnobs)
            ax1.set_ylabel('nobs - used', color='teal', fontsize=10)
            ax1.tick_params('y', colors='teal')
            plt.xticks(x_axis, DayHour)
            major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            ax1.set_xticks(major_ticks)
            #plt.axhline(y=np.mean(list_dataBiasBC),ls='dotted',c='steelblue')
            #plt.axhline(y=np.mean(list_dataBiasAC),ls='dotted',c='tomato')
            
            
            ax2 = ax1.twinx()
            p2 = ax2.plot(x_axis, list_dataObsT, "--", c='red', label="nobs (tossed by gross check)")
            label2 = "nobs (tossed by gross check)"
            #ax2.plot(x_axis, std_finala, "r-", label="Std. Deviation ("+omflaga+")")
            #ax2.plot(x_axis, std_finala, "rs", label="Std. Deviation ("+omflaga+")")
            ax2.set_ylim(0.0, 1.1*np.max(dataObsT_final))
            ax2.set_ylabel('nobs - tossed', color='r', fontsize=10)
            ax2.tick_params('y', colors='r')
            #ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            #plt.axhline(y=np.mean(std_finala),ls='dotted',c='red')

            ps = p1 + p2
            labs = [l.get_label() for l in ps]
            
            plt.legend(ps, labs, numpoints=1, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
                   fancybox=True, shadow=True, frameon=True, ncol=2, prop={"size": 10})
            
            #plt.xticks(x_axis, DayHour)
            #major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
            #ax3.set_xticks(major_ticks)
            plt.title(instrument_title, loc='left', fontsize=9)
            plt.title(date_title, loc='right', fontsize=9)
            plt.subplots_adjust(left=None, bottom=None, right=0.80, top=None)
            plt.tight_layout()
            plt.savefig('time_series_nobs_'+ Type +'_'+ Sat +'_it'+ str(it) +'_'+forplotname+'.png', bbox_inches='tight', dpi=100)
            if Clean:
                plt.clf()
                
            
        # Cleaning up
        if Clean:
            plt.close('all')

        print(' Done!')
        print()
        
               

        return




#EOC
#-----------------------------------------------------------------------------#

