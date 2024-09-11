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



# Avaliação idq radiancia
    def statcount_idqc(self, varName=None, varType=None, dateIni=None, dateFin=None, nHour="06", channel=None, figTS=False, figMap=False, **kwargs):

        '''
        The StatCount_idqc function plots a time series of the idqc flags. 

        Example:

        varName = 'uv'           # Variable
        varType = 224            # Source Type
        dateIni = 2013010100     # Inicial Date
        dateFin = 2013010900     # Final Date
        nHour = "06"             # Time Interval
        channel = 10             # Radiance channel number
        figTS = True             # Creates the time series plot
        figMap = False           # Creates the spatial plot for each time        
        
        ! Table: classification idqc flags
        !
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   idqc |   flag                                          |   idqc |   flag                                          |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   0    | boa observação                                  |   7    | rejeitar devido a 'nuvem > limite' para canal   |
        !    |        |                                                 |        | na rotina qc                                    |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   1    | rejeitar devido ao flag no radinfo              |   8    | rejeitar devido a estimativa imprecisa de emis- |
        !    |        |                                                 |        | sividade/temperatura da superfície na rotina qc |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   2    | falha no CRTM                                   |   9    | rejeitar devido as observações estarem fora do  |
        !    |        |                                                 |        | intervalo na rotina de controle de qualidade    |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   3    | Rejeitar devido a falha grosseira               |   10   | rejeitar devido a recuperação física de tempera-|
        !    |        | de verificação                                  |        | tura de superfície muito grande na rotina qc    |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   4    | rejeitar devido à verificação                   |   50   | rejeitar porque 'factch6 > limite' na sub-rotina|
        !    |        | de intercâmbio                                  |        | qc_amsua                                        |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   5    | rejeitar devido a não usar sobre esta superfície|   51   | rejeitar porque 'factch4 > limite' na sub-rotina|
        !    |        | na rotina de controle de qualidade              |        | qc_amsua                                        |
        !    +--------+-------------------------------------------------+--------+-------------------------------------------------+
        !    |   6    | rejeitar devido a verificação bruta na rotina   |
        !    |        | de controle de qualidade específica             |
        !    +--------+-------------------------------------------------+
        '''


        varInfo = getVarInfo(varType, varName, 'instrument')
        if varInfo is not None:
            instrument_title = str(varName) + '-' + str(varType) + '  |  ' + varInfo
        else:
            instrument_title = str(varName) + '-' + str(varType) + '  |  ' + 'Unknown instrument'

        datei = datetime.strptime(str(dateIni), "%Y%m%d%H")
        datef = datetime.strptime(str(dateFin), "%Y%m%d%H")
        date  = datei

        flags, aux0 = [], []
        nf = {}
        
        DayHour_tmp = []
        
        if varName == 'amsua':
            #flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51]
            aux0 = list(map(round,self[0].obsInfo[varName].query("nchan=="+str(channel)).loc[varType].idqc))
            flags = pd.unique(aux0)
            flags.sort()
        elif varName == 'hirs4':
            #flags = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51]
            aux0 = list(map(round,self[0].obsInfo[varName].query("nchan=="+str(channel)).loc[varType].idqc))
            flags = pd.unique(aux0)
            flags.sort()
        else:
            print(setcolor.WARNING + "    >>> Flags for sensor " + varName + " not set <<< " + setcolor.ENDC)
                    
        [nf.update({int(i): []}) for i in flags]
        
        f = 0
        while (date <= datef):

            datefmt = date.strftime("%Y%m%d%H")
            DayHour_tmp.append(date.strftime("%d%H"))
            
#             list_flags, name_list = {}, {}
#             [ list_flags.update({int(i): []}) for i in flags ]
#             [ name_list.update({int(i): []}) for i in flags ]
#             print('list_flags = ', list_flags)
#             print('name_list = ', name_list)
            
            
            # try: For issues reading the file (file not found)
            # in the except statement an error message is printed and continues for other dates
            try:
                
                soma_flags = 0
                setColor = 0 
                legend_labels = []
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
                    
                    #---------------- cria lista da serie temp da qntdd de dados da flag ------------#
                    #nf[int(fl)].append(len(aux1))
                    #print('nf = ', nf)                    
                    
                    #---------------- soma qntdd de dados rejeitados (idqc!=0) ------------#
                    if (fl != 0.0):
                        soma_flags = soma_flags + len(aux1)
                    
                    #---------------- Comandos de plote ------------#
                    if ( len(aux1) > 0 ):
                        label = "flag "+str(fl)+" ["+str(len(aux1))+"]"
                        #color = getColor(minVal=0, maxVal=len(flags)-1, 
                        #                 value=fl,hex=True,cmapName='Paired')
                        color = getColor(minVal=0, maxVal=len(flags), 
                                         value=ll,hex=True,cmapName='tab20')
                        
                        
                        legend_labels.append(mpatches.Patch(color=color, label=label) )
                        ax = aux1.plot(ax=ax,legend=True, marker="o", color=color, **kwargs)
                        setColor += 1
                        plt.legend(handles=legend_labels, numpoints=1, loc='center left', bbox_to_anchor=(1.0, 0.5), 
                                   fancybox=True, shadow=False, frameon=False, ncol=1, prop={"size": 10})
                    
                #---------------- FIM for --> flags inteiras ------------#
                    
                for ll in range(len(flags)):
                    fl = flags[ll]
                    print('ll = ', ll,'flag = ', fl)
                    aux1 = []
                     
                    #---------------- teste flegs decimais com arredondamento ------------#
                    if (fl <= -1):
                        inf = fl - 1
                        exp  = "(nchan=="+str(channel)+") & (idqc>"+str(inf)+" & idqc<"+str(fl)+")"
                        
                        #---------------- lista com os dados da flag ------------#
                        aux1 = self[f].obsInfo[varName].loc[varType].query(exp)
                        
                        #---------------- soma qntdd de dados rejeitados (idqc!=0) ------------#
                        if (fl != 0.0):
                            soma_flags = soma_flags + len(aux1)
                            
                        #---------------- Comandos de plote ------------#
                        if ( len(aux1) > 0 ):
                            label = "flag btw ("+str(inf)+" "+str(fl)+") ["+str(len(aux1))+"]"
                            #color = getColor(minVal=0, maxVal=len(flags)-1, 
                            #                 value=fl,hex=True,cmapName='Paired')
                            color = getColor(minVal=0, maxVal=len(flags), 
                                             value=ll,hex=True,cmapName='tab20')
                        
                            legend_labels.append(mpatches.Patch(color=color, label=label) )
                            ax = aux1.plot(ax=ax,legend=True, marker="o", color=color, **kwargs)
                            setColor += 1
                            plt.legend(handles=legend_labels, numpoints=1, loc='center left', bbox_to_anchor=(1.0, 0.5), 
                                       fancybox=True, shadow=False, frameon=False, ncol=1, prop={"size": 10})
                        
                    elif (fl >= 1):
                        sup = fl + 1
                        exp  = "(nchan=="+str(channel)+") & (idqc>"+str(fl)+" & idqc<"+str(sup)+")"
                        
                        #---------------- lista com os dados da flag ------------#
                        aux1 = self[f].obsInfo[varName].loc[varType].query(exp)
                        
                        #---------------- soma qntdd de dados rejeitados (idqc!=0) ------------#
                        if (fl != 0.0):
                            soma_flags = soma_flags + len(aux1)
                            
                        #---------------- Comandos de plote ------------#
                        if ( len(aux1) > 0 ):
                            label = "flag btw ("+str(fl)+" "+str(sup)+") ["+str(len(aux1))+"]"
                            #color = getColor(minVal=0, maxVal=len(flags)-1, 
                            #                 value=fl,hex=True,cmapName='Paired')
                            color = getColor(minVal=0, maxVal=len(flags), 
                                             value=ll,hex=True,cmapName='tab20')
                        
                            legend_labels.append(mpatches.Patch(color=color, label=label) )
                            ax = aux1.plot(ax=ax,legend=True, marker="o", color=color, **kwargs)
                            setColor += 1
                            plt.legend(handles=legend_labels, numpoints=1, loc='center left', bbox_to_anchor=(1.0, 0.5), 
                                       fancybox=True, shadow=False, frameon=False, ncol=1, prop={"size": 10})
                        
                    else:
                        sup = fl + 1
                        inf = fl - 1
                        exp  = "(nchan=="+str(channel)+") & (idqc>"+str(inf)+" & idqc<"+str(sup)+")"
                        
                        #---------------- lista com os dados da flag ------------#
                        aux1 = self[f].obsInfo[varName].loc[varType].query(exp)
                        
                        #---------------- soma qntdd de dados rejeitados (idqc!=0) ------------#
                        if (fl != 0.0):
                            soma_flags = soma_flags + len(aux1)
                            
                        #---------------- Comandos de plote ------------#
                        if ( len(aux1) > 0 ):
                            label = "flag btw ("+str(inf)+" "+str(sup)+") ["+str(len(aux1))+"]"
                            #color = getColor(minVal=0, maxVal=len(flags)-1, 
                            #                 value=fl,hex=True,cmapName='Paired')
                            color = getColor(minVal=0, maxVal=len(flags), 
                                             value=ll,hex=True,cmapName='tab20')
                        
                            legend_labels.append(mpatches.Patch(color=color, label=label) )
                            ax = aux1.plot(ax=ax,legend=True, marker="o", color=color, **kwargs)
                            setColor += 1
                            plt.legend(handles=legend_labels, numpoints=1, loc='center left', bbox_to_anchor=(1.0, 0.5), 
                                       fancybox=True, shadow=False, frameon=False, ncol=1, prop={"size": 10})
                    
                #---------------- FIM for --> flags decimas - intervalos ------------#
                    
                
                #---------------- comandos finais do plote e salva figura ------------#
                forplot = 'Channel ='+str(channel)+' | Rejected total ='+str(soma_flags)
                
                date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
                plt.title(date_title, loc='right', fontsize=10)
                plt.title(instrument_title, loc='left', fontsize=9)
                plt.annotate(forplot, xy=(0.45, 1.015), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
                             color='gray', fontweight='bold', fontsize='10', horizontalalignment='left', verticalalignment='center')
                
                plt.tight_layout()
                plt.savefig('Flags-idqc_'+str(varName) + '-' + str(varType)+'_'+ 'CH' + str(channel) + '_' +datefmt+'.png', 
                            bbox_inches='tight', dpi=100)
                    
                
                
#                 exp    = "(nchan=="+str(channel)+") & (idqc==0.0)"
#                 flag_0 = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp    = "(nchan=="+str(channel)+") & (idqc==1.0)"
#                 flag_1 = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp    = "(nchan=="+str(channel)+") & (idqc==2.0)"
#                 flag_2 = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp    = "(nchan=="+str(channel)+") & (idqc==3.0)"
#                 flag_3 = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp    = "(nchan=="+str(channel)+") & (idqc==4.0)"
#                 flag_4 = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp    = "(nchan=="+str(channel)+") & (idqc==5.0)"
#                 flag_5 = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp    = "(nchan=="+str(channel)+") & (idqc==6.0)"
#                 flag_6 = self[f].obsInfo[varName].loc[varType].query(exp)
                
#                 exp     = "(nchan=="+str(channel)+") & (idqc==7.0)"
#                 flag_7  = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp     = "(nchan=="+str(channel)+") & (idqc==8.0)"
#                 flag_8  = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp     = "(nchan=="+str(channel)+") & (idqc==9.0)"
#                 flag_9  = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp     = "(nchan=="+str(channel)+") & (idqc==10.0)"
#                 flag_10 = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp     = "(nchan=="+str(channel)+") & (idqc==50.0)"
#                 flag_50 = self[f].obsInfo[varName].loc[varType].query(exp)
#                 exp     = "(nchan=="+str(channel)+") & (idqc==51.0)"
#                 flag_51 = self[f].obsInfo[varName].loc[varType].query(exp)
                
#                 soma_flags = 0.00000
#                 soma_flags = len(flag_0) + len(flag_1) + len(flag_2) + len(flag_3) + len(flag_4) + len(flag_5) + len(flag_6) + len(flag_7) + len(flag_8) + len(flag_9) + len(flag_10) + len(flag_50) + len(flag_51)
                
#                 nf_0.append(len(flag_0))
#                 nf_1.append(len(flag_1))
#                 nf_2.append(len(flag_2))
#                 nf_3.append(len(flag_3))
#                 nf_4.append(len(flag_4))
#                 nf_5.append(len(flag_5))
#                 nf_6.append(len(flag_6))
                
#                 nf_7.append(len(flag_7))
#                 nf_8.append(len(flag_8))
#                 nf_9.append(len(flag_9))
#                 nf_10.append(len(flag_10))
#                 nf_50.append(len(flag_50))
#                 nf_51.append(len(flag_51))
                    
#                 forplot = 'Channel ='+str(channel)+' | Total flags ='+str(soma_flags)
                
#                 # Radiance plots
#                 if (figMap):
#                     # Case: assimilated and rejected
#                     #if ((len(assim)) != 0 or (len(rejei)) != 0):
#                     df_list = [flag_0, flag_1, flag_2, flag_3, flag_4, flag_5, flag_6, flag_7, flag_8, flag_9, flag_10, flag_50, flag_51]    
#                     name_list = ["flag 0 ["+str(len(flag_0))+"]","flag 1 ["+str(len(flag_1))+"]","flag 2 ["+str(len(flag_2))+"]",
#                                  "flag 3 ["+str(len(flag_3))+"]","flag 4 ["+str(len(flag_4))+"]","flag 5 ["+str(len(flag_5))+"]",
#                                  "flag 6 ["+str(len(flag_6))+"]","flag 7 ["+str(len(flag_7))+"]","flag 8 ["+str(len(flag_8))+"]",
#                                  "flag 9 ["+str(len(flag_9))+"]","flag 10 ["+str(len(flag_10))+"]","flag 50 ["+str(len(flag_50))+"]",
#                                  "flag 51 ["+str(len(flag_51))+"]"]
# #                     marker_list = ["o","o","o","o","o","o","o","o","o","o","o","o","o"]    
#                     color_list = ["green","navy","teal","deepskyblue","cyan","springgreen","lime","greenyellow","yellow","orange",
#                                  "salmon","hotpink","red"]
                    
#                     setColor = 0 
#                     legend_labels = []
                    
#                     fig = plt.figure(figsize=(12, 6))
#                     ax  = fig.add_subplot(1, 1, 1)
#                     ax = geoMap(area=None,ax=ax)
# #                     for dfi,namedf,mk,cl in zip(df_list,name_list,marker_list,color_list):
#                     for dfi,namedf,cl in zip(df_list,name_list,color_list):
#                         df    = dfi
#                         legend_labels.append(mpatches.Patch(color=cl, label=namedf) )
# #                         ax = df.plot(ax=ax,legend=True, marker=mk, color=cl, **kwargs)
#                         ax = df.plot(ax=ax,legend=True, marker="o", color=cl, **kwargs)
#                         setColor += 1
#                         plt.legend(handles=legend_labels, numpoints=1, loc='center left', bbox_to_anchor=(1.0, 0.5), 
#                                    fancybox=True, shadow=False, frameon=False, ncol=1, prop={"size": 10})
                        
#                     date_title = str(date.strftime("%d%b%Y - %H%M")) + ' GMT'
#                     plt.title(date_title, loc='right', fontsize=10)
#                     plt.title(instrument_title, loc='left', fontsize=9)
#                     plt.annotate(forplot, xy=(0.45, 1.015), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
#                                  color='gray', fontweight='bold', fontsize='10', horizontalalignment='left', verticalalignment='center')
                    
#                     plt.tight_layout()
#                     plt.savefig('Flags-idqc_'+str(varName) + '-' + str(varType)+'_'+ 'CH' + str(channel) + '_' +datefmt+'.png', 
#                                 bbox_inches='tight', dpi=100)
# #                     else:
# #                        print("channel ",channel," not assimilated or rejected on the date -->",date.strftime("%Y-%m-%d:%H"))
                    
            except:
                print("++++++++++++++++++++++++++ ERROR: file reading --> STATCOUNT ++++++++++++++++++++++++++")
                print(setcolor.WARNING + "    >>> No information on this date (" + str(date.strftime("%Y-%m-%d:%H")) +") <<< " + setcolor.ENDC)
#                 if(channel == None):
#                     assi.append(None)
#                     moni.append(None)
#                     reje.append(None)
#                 else:
#                     assi.append(None)
#                     moniAssi.append(None)
#                     moniReje.append(None)
#                     reje.append(None)
                    
            f = f + 1
            date = date + timedelta(hours=int(nHour))
            date_finale = date
            


#         if (figTS):
#             if(channel == None):   # Conventional
#                 if(len(DayHour_tmp) > 4):
#                     DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
#                 else:
#                     DayHour = DayHour_tmp
                
#                 x_axis      = np.arange(0, len(DayHour), 1)
#                 date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
            
#                 fig = plt.figure(figsize=(6, 4))
#                 fig, ax1 = plt.subplots(1, 1)
#                 plt.style.use('seaborn-v0_8-ticks')

#                 plt.axhline(y=0.0,ls='solid',c='#d3d3d3')

#                 ax1.plot(x_axis, assi, "o", label="Assimilated \n["+str(sum(assi))+"]", color='green')
#                 ax1.plot(x_axis, moni, "o", label="Monitored \n["+str(sum(moni))+"]", color='blue')
#                 ax1.plot(x_axis, reje, "o", label="Rejected \n["+str(sum(reje))+"]", color='red')
#                 ax1.legend(fancybox=True, frameon=True, shadow=True, loc="upper center",ncol=3)
#                 ax1.set_xlabel('Date (DayHour)', fontsize=10)
#                 plt.title(date_title, loc='right', fontsize=10)
#                 plt.title(instrument_title, loc='left', fontsize=9)
                
#                 ax1.set_ylim(np.round(-0.05*np.max([assi,moni,reje])), np.round(1.25*np.max([assi,moni,reje])))
#                 ax1.set_ylabel('Total Observations', color='black', fontsize=10)
#                 ax1.tick_params('y', colors='black')
#                 plt.xticks(x_axis, DayHour)
#                 major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
#                 ax1.set_xticks(major_ticks)
#                 plt.axhline(y=np.mean(assi),ls='dotted',c='lightgray')
#                 plt.axhline(y=np.mean(moni),ls='dotted',c='lightgray')
#                 plt.axhline(y=np.mean(reje),ls='dotted',c='lightgray')
#                 plt.tight_layout()
#                 plt.savefig('time_series_'+str(varName) + '-' + str(varType)+'_TotalObs.png', bbox_inches='tight', dpi=100)
                
#             else:   # Radiance
#                 if(len(DayHour_tmp) > 4):
#                     DayHour = [hr if (ix % int(len(DayHour_tmp) / 4)) == 0 else '' for ix, hr in enumerate(DayHour_tmp)]
#                 else:
#                     DayHour = DayHour_tmp
                
#                 x_axis      = np.arange(0, len(DayHour), 1)
#                 date_title = str(datei.strftime("%d%b")) + '-' + str(date_finale.strftime("%d%b")) + ' ' + str(date_finale.strftime("%Y"))
            
#                 fig = plt.figure(figsize=(6, 4))
#                 fig, ax1 = plt.subplots(1, 1)
#                 plt.style.use('seaborn-v0_8-ticks')

#                 plt.axhline(y=0.0,ls='solid',c='#d3d3d3')
                
#                 # List with value None: is removed to calculate sum, max and min
#                 # The lists below are only used to define the scale of the axes and the total sum of assi/rejei/monit data
#                 assif     = [x for x in assi if x != None]
#                 moniAssif = [x for x in moniAssi if x != None]
#                 moniRejef = [x for x in moniReje if x != None]
#                 rejef     = [x for x in reje if x != None]

#                 ax1.plot(x_axis, assi, "o", label="Assimilated \n["+str(sum(assif))+"]", color='green')
#                 ax1.plot(x_axis, moniAssi, "o", label="Monitored-Assim \n["+str(sum(moniAssif))+"]", color='teal')
#                 ax1.plot(x_axis, moniReje, "o", label="Monitored-Rejei \n["+str(sum(moniRejef))+"]", color='purple')
#                 ax1.plot(x_axis, reje, "o", label="Rejected \n["+str(sum(rejef))+"]", color='red')
#                 ax1.legend(fancybox=True, frameon=True, shadow=True, loc="best",ncol=1)
#                 ax1.set_xlabel('Date (DayHour)', fontsize=10)
#                 plt.title(date_title, loc='right', fontsize=10)
#                 plt.title(instrument_title, loc='left', fontsize=9)
#                 plt.annotate(forplot, xy=(0.0, 0.965), xytext=(0, 0), xycoords='axes fraction', textcoords='offset points', 
#                              color='lightgray', fontweight='bold', fontsize='12', horizontalalignment='left', verticalalignment='center')
                
#                 ax1.set_ylim(np.round(-0.05*np.max([assif,moniAssif,moniRejef,rejef])),
#                              np.round(1.25*np.max([assif,moniAssif,moniRejef,rejef])))
#                 ax1.set_ylabel('Total Observations', color='black', fontsize=10)
#                 ax1.tick_params('y', colors='black')
#                 plt.xticks(x_axis, DayHour)
#                 major_ticks = [ DayHour.index(dh) for dh in filter(None,DayHour) ]
#                 ax1.set_xticks(major_ticks)
#                 plt.axhline(y=np.mean(assif),ls='dotted',c='lightgray')
#                 plt.axhline(y=np.mean(moniAssif),ls='dotted',c='lightgray')
#                 plt.axhline(y=np.mean(moniRejef),ls='dotted',c='lightgray')
#                 plt.axhline(y=np.mean(rejef),ls='dotted',c='lightgray')
#                 plt.tight_layout()
#                 plt.savefig('time_series_'+str(varName) + '-' + str(varType) +'_'+ 'CH' + str(channel) + '_'+'_TotalObs.png',
#                             bbox_inches='tight', dpi=100)

                
                
                

#EOC
#-----------------------------------------------------------------------------#

