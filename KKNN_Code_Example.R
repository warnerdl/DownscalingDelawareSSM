#set random seed 
set.seed(23)

#path to working folder, 
#this folder should contain unzipped versions of the test data folders that were provided with this code.
wpath = 'path to working folder'

#load libraries
library(raster)
library(lubridate)
library(dplyr)
library(rgdal)
library(raster)
library(kknn)
library(foreach)
library(doParallel)
library(RStoolbox)
library(caret)

######################################################################
#load a file with your independent SSM in situ observations
vwc=read.csv(paste0(wpath,'/deos_daily_vwc.csv'),header=T)
vwc$Time=date(vwc$Time)
vwc$Month=month(vwc$Time)
vwc$Day=day(vwc$Time)
vwc$Year=year(vwc$Time)
vwc$Week=week(vwc$Time)
vwc$Date=as.Date(paste0(vwc$Year,'-',vwc$Month,'-',vwc$Day))

#get low resolution (ESACCI) soil moisture files
esapath=paste0(wpath,'/ESAGrids')
lis=list.files(esapath,full.names=TRUE,pattern='*.tif$')
refRaster<-extent(raster(lis[1])) #establish a reference extent

########################################################################################
#Now we will go through the ESA rester names and make a dataframe with their file name and date
#NOTE: In lines 39-45 you will need to adjust the index value to get the right digits for date and year
date<-character() ##
##
#This will pull dates out of the filenames ##
for(i in 1:length(lis)){##
  y = substr(lis[i],nchar(lis)-11,nchar(lis)-8)##
  m = substr(lis[i],nchar(lis)-7,nchar(lis)-6)##
  d = substr(lis[i],nchar(lis)-5,nchar(lis)-4)##
  date[i] = paste0(y,'-',m,'-',d)##
}##

lis=as.data.frame(lis)##
names(lis)[1]<-'FileName' #column of file names
lis$Day<-day(as.Date(date)) #column of day
lis$Week<-week(as.Date(date)) #column of week
lis$Month<-month(as.Date(date)) #column of month
lis$Year<-year(as.Date(date)) #column of year
lis$Date<-as.Date(date)##
cutoff=as.Date('2018-01-01') #establish cut off date, should be 3 days before first date of interest, can adjust to be date ranges if needed
lis<-lis[lis$Date>=cutoff,]

lis$DOY= yday(lis$Date) #get day of year

##################################################################################################
# Now we will load our covariate layers already saved as a raster stack in a .rds, or you can load from a folder and stack.

#x<-readRDS(file='covariate file.rds')
x<-stack(list.files(paste0(wpath,'/CovariateGrids'), full.names = T, pattern = '*sdat$'))
#pcr = rasterPCA(x,) #use only if a principal components reduction is being used.

# Now get list of interpolated meteorological layers.
deospgrds=list.files(paste0(wpath,'/PrecipGrids'), full.names = T, pattern='*.tif$')
deostgrds=list.files(paste0(wpath,'/TempGrids'), full.names = T,pattern='*.tif$')
metgrds = cbind.data.frame(deospgrds, deostgrds, stringsAsFactors = F)
names(metgrds) = c('Precip','Temp')
metgrds$Year = substr(metgrds$Precip, nchar(metgrds$Precip) - 11, nchar(metgrds$Precip) - 8)
metgrds$Month = substr(metgrds$Precip, nchar(metgrds$Precip) - 7, nchar(metgrds$Precip) - 6)
metgrds$Day = substr(metgrds$Precip, nchar(metgrds$Precip) - 5, nchar(metgrds$Precip) - 4)
metgrds$Date = as.Date(paste0(metgrds$Year, '-', metgrds$Month, '-', metgrds$Day))

#Get daylength grid. This was generated using the 'geosphere' package and the centroid of our study area
dlengthpath = paste0(wpath,'/DayLength')
dll = list.files(dlengthpath, pattern = '*.sdat$', full.names = T)

###################################################################################################
#Get the XY coordinates of the validation SSM sites

sitepts=as.data.frame(read.csv(paste0(wpath,'/NetworkPoints/deos_met.csv'),header=T))
sitesxy=sitepts[,c(13,12)] #adjust index as needed to get xy columns
stations=as.data.frame(as.character(sitepts[,1])) #get column of station names/IDs
sitesdf=SpatialPointsDataFrame(coords=sitesxy,data=stations,proj4string=CRS('+init=epsg:4326')) #project as SPDF into whatever CRS the data are in
names(sitesdf)='Station'
sitesdf=spTransform(sitesdf,proj4string(x)) #reproject to CRS of your covariate and ESA grids

##########################################################################################################

###########################################
#MODEL TRAINING AND TUNING
##########################################

##########################################################################################################
#set bootstrap params and register a cluster for where we can use it
modoutpath = paste0(wpath,'/ModelOut')
dir.create(modoutpath)
registerDoParallel(cores=2)
nreps = 2 #number of bootstrap runs
ndays = 10 #number of random dates to pull training data  ####

#set kknn tuning params
#Set range for maximum k nearest neighbors
kmaxr=5:10 ###
#Set range of kernel functions
kernels=c("triangular","epanechnikov")#,"gaussian","optimal")

#Set C.V.parameters and tuning ranges for K and kernel
fitControl<-trainControl(method="repeatedcv",repeats=2,number=5,allowParallel=TRUE,savePredictions='final',verboseIter=TRUE)
traingrid_kknn<-expand.grid(kmax=kmaxr,distance=2,kernel=kernels)

#Establish empty objects
vimp<-data.frame() #This will be to collect our variable importance outputs
modelperformance<-data.frame() #This will be to collect RMSE, MAE, and R2 of model run
lev<-lis$Date #yyyy-mm-dd format

#Train and tune nreps KKNN models. This will take a long time.
for(j in 1:nreps) { #####
  #Randomly select ndays dates (must exclude first three dates to get 3-day antecedent Precipitation -- GET 3 Previous Days in all data inputs)
  lev3<-lev[3:length(lev)]
  rnd_dates=as.data.frame(sample(lev3,ndays,replace=FALSE)) ######
  names(rnd_dates)<-"Date"
  rnd_dates=merge(rnd_dates,lis[,6:7],by="Date") #Merge dates and corresponding ESACCI grid filenames
  
  ####for each random date, select 250 random points, get corresponding precip and temperature grids, and stack them into a dataframe
  #Run in parallel for faster process
  rndpts=foreach(i=1:nrow(rnd_dates),.packages=c('raster','kknn','caret'),.combine=rbind)%dopar%{
    d=rnd_dates$Date[i]
    d1 = d - 1
    d2 = d - 2
    
    p3=stack(metgrds$Precip[which(metgrds$Date %in% c(d,d1,d2))]) #Stack the precip and temp grids for antecedent values
    t2=stack(metgrds$Temp[which(metgrds$Date %in% c(d,d1))]) ####** 
    
    api3=sum(p3)
    ati2=mean(t2)
    
    esagrd=raster(as.character(lis$FileName[lis$Date==d])) #get ESACCI grid
    esagrd=projectRaster(esagrd,x,method='ngb') #resample and project ESACCI grid to match covariate stack
    
    daylen = raster(dll[which(grepl(d,dll))]) #get day length grid
    
    xall=stack(c(x,api3,ati2,esagrd,daylen)) #stack everything 
    names(xall)[(nlayers(x) + 1):(nlayers(x) + 4)]<-c('API3','ATI2','ESA_CCI_SM','DayLength') #Make sure when this runs all the variable names are correct
    rndpts = as.data.frame(sampleRandom(xall,250,na.rm=TRUE))
  }
  ###############
  #kknn model training
  
  kn_cv<-train(ESA_CCI_SM~.,
  data=rndpts,
  method="kknn",
  tuneGrid=traingrid_kknn,metric="RMSE",
  preProcess=c('scale','center'),
  maximize=FALSE,
  trControl=fitControl,
  importance=TRUE
  )
  
  #Extract variable importance for each iteration
  var_imp = as.data.frame(varImp(kn_cv)$importance)
  var_imp$Round = j
  var_imp$Covariate = row.names(var_imp)
  vimp <- rbind(vimp, var_imp)
  
  #Get best parameters
  best_k = kn_cv$bestTune$kmax
  best_kernel = kn_cv$bestTune$kernel
  
  #get MAE, RMSE, and Rsquared
  bestout = kn_cv$results[kn_cv$results$kmax == best_k & kn_cv$results$kernel == best_kernel,] ##
  bestout$Round = j
  modelperformance = rbind(modelperformance, bestout)
  
  saveRDS(kn_cv, file = paste0(modoutpath,'/kknn_out_rnd_',j,'.rds')) #save train object to .rds
}    

write.table(vimp,paste0(modoutpath,'/Var_Imp.txt'), quote = F, sep = '\t', row.names = F) #save variable importance
write.table(modelperformance,paste0(modoutpath,'/Model_Performance.txt'), quote = F, sep = '\t', row.names = F) #save model performances
##################################################################################################################################

##############
#************** NOTE: Look at your variable importance outputs... are there any weak variables to exclude? 
#************** If so, exclude them, and re-run the models. If not, proceed with making predictions
#************** 
##############

###################################################################################################################################

##################################################################################################################################

############################
#MAKING MODEL PREDICTIONS

##############################

#########################################################################################################################

#Make functions to calculate grid cell means and SD from ensemble
f1<-function(x)calc(x,mean,na.rm=TRUE)
f2<-function(x)calc(x,sd,na.rm=TRUE)

#Make list of days you want to predict...
pred_dates = as.data.frame(lev3) ###
names(pred_dates)<-"Date"
pred_dates=merge(pred_dates,lis[,6:7],by="Date")

#Make path for output grids
predpath=paste0(wpath, '/DownscaledGrids')
dir.create(predpath)

#Make list of KKNN saved models
modlist = list.files(modoutpath,full.names = T, pattern = '*.rds$')

#Looped parallel process for making predictions. This can take a long time. 
foreach(i=1:nrow(pred_dates),.packages=c('raster','kknn','caret'))%dopar%{
  d=pred_dates$Date[i]
  if(file.exists(paste0(predpath,'/KKNN_SSM_',d,'_Mean.tif')) == TRUE) next #Set this line so that it skips grids that have already been predicted ##
  
  d1 = d - 1
  d2 = d - 2
  
  p3=stack(metgrds$Precip[which(metgrds$Date %in% c(d,d1,d2))]) #Stack the precip and temp grids for antecedent values
  t2=stack(metgrds$Temp[which(metgrds$Date %in% c(d,d1))]) ####** 
  
  api3=sum(p3) #total 72 hr precipitation
  ati2=mean(t2) #mean 48 hr temperature
  daylen = raster(dll[[which(grepl(d,dll) == T)]]) #day length
  
  #Stack all covariates for this day
  xall_p=stack(c(x,api3,ati2,daylen))
  names(xall_p)[(nlayers(x) + 1):(nlayers(x) + 3)]<-c('API3','ATI2','DayLength') #make layer names == variable ames from model object
  xall_psp<-as(xall_p,'SpatialPixelsDataFrame') #convert to SPDF
  
  ##Next loop makes predictions for all 10 models and stacks them
  pstack = stack()
  for(k in 1:length(modlist)){
    mod = readRDS(modlist[k])
    pred=predict(xall_p,mod)
    pstack <- stack(pstack,pred)
  }
  
  predmean = f1(pstack) #ensemble means -> Final SSM predictions
  sdp = f2(pstack) #ensemble SD -> SD of ensemble distribution
  
  writeRaster(pred,paste0(predpath,'/KKNN_SSM_',d,'_Mean.tif'),overwrite=T) #write to file ##
  writeRaster(pred,paste0(predpath,'/KKNN_SSM_',d,'_SD.tif'),overwrite=T) #write to file ##
}

##############################################################################
# These grids are now in your output folder and ready for downstream analysis. 
##############################################################################

##############################################################################################
# Scale grids based on daily network observed min and max
# 
#
################################################################################################

#Function for scaling raster and writing output
scalr = function(pras, omin, omax){
  pmin = cellStats(pras, min)
  pmax = cellStats(pras, max)
  p01 = (pras - pmin)/(pmax - pmin)
  pscaled = omin + (p01*(omax - omin))
  return(pscaled)
}

#Get downscaled predicted grids (mean of ensemble)
predgrds=list.files(predpath,pattern='*.tif$')
predgrds_m=predgrds[which(grepl("Mean",predgrds)==T)]

#Extract predicted VS observed network values
deos_df_list=foreach(i=1:length(predgrds_m),.packages=c('raster','kknn','caret'),.combine=rbind)%dopar%{

  #Get network values for each date in sequence
  d=pred_dates$Date[i]
  deosvwc=data.frame(names(vwc)[2:29],t(vwc[vwc$Date==d,2:29])) #this was only necessary because of how our data was formatted
  names(deosvwc)=c('Station','Observed')
  obs=sitesdf
  
  #Get predicted grid and ESACCI grid for each date in sequence
  predgrd=raster(paste0(predpath,'/',predgrds_m[grepl(d,predgrds_m)]))
  esagrd=raster(as.character(lis$FileName[lis$Date==d]))
  esagrd=projectRaster(esagrd,x,method='ngb')
  
  #Extract downscaled predictions and ESACCI values
  obs@data$KKNN=extract(predgrd,obs)
  obs@data$ESACCI=extract(esagrd,obs)
  
  #Extract observed values 
  obs=merge(obs,deosvwc,by='Station') 
  ##

  # Now get min max values and rescale rasters
  obs_min = min(obs$Observed, na.rm = T)#
  obs_max = max(obs$Observed, na.rm = T)#

  #Scale and write ESA grid (e.g. ESAsc)
  esasc = scalr(esagrd, obs_min, obs_max)
  writeRaster(esasc, paste0(predpath,'/ESA_scaled_',d,'.tif'), format = "GTiff")
  
  #Scale and write downscaled grid (e.g. KKNNsc)
  knnsc = scalr(predgrd, obs_min, obs_max)
  writeRaster(knnsc, paste0(predpath,'/KKNN_scaled_',d,'.tif'), format = "GTiff")
  
  #Extract rescaled values
  obs@data$KKNN_scaled=extract(knnsc,obs)
  obs@data$ESACCI_scaled=extract(esasc,obs)
  
  #Pass date
  obs@data$Date = d
  as.data.frame(obs)
}

#Write table of raw and scaled datasets
write.table(deos_df_list,paste0(predpath,'/DEOS_DAILY_PREDS_OBS.txt'),sep='\t',quote=F)



