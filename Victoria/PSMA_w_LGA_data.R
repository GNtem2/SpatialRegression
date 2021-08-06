library(simstudy)
library(leaflet)
library(SpatialEpi)
library(spdep)
library(spatialreg)
library(ggplot2)
library(tmap)
library(sf)
library(dplyr)
library(MASS)

library(tidyverse)
library(units)
library(tmaptools)
library (mapview)
library(PSMA)
set.seed(1234)
df<-read.csv(".../Postcode_LGA_w_Data.csv")
df<-subset(df,Addresses==1)

gdt<- st_read(".../POA_2016_AUST.shp")
gdt$POA_CODE16<-as.integer(gdt$POA_CODE16)
df$Code<-as.integer(df$Code)
gdt<-left_join(df, gdt,by=c("POA_CODE_2016"="POA_CODE16"))
gdt<-st_as_sf(gdt)



x=1
LGA1 <- defData(varname = "Obese", dist = "binary", formula = df[x,]$Obese)
LGA1 <- defData(LGA1,varname = "Smoking", dist = "binary", formula = df[x,]$Smoking)
LGA1 <- defData(LGA1,varname = "Diabetes", dist = "binary", formula = df[x,]$Diabetes)
LGA1 <- defData(LGA1,varname = "HeartDisease", dist = "binary", formula = df[x,]$HeartDisease)
LGA1 <- defData(LGA1,varname = "MedianIncome", dist = "normal", formula = df[x,]$MedianIncome,variance=(df[x,]$MedianIncome*0.7)**2)
LGA1 <- defData(LGA1,varname = "LGA_id", formula = df[x,]$Code)
LGA1 <- defData(LGA1,varname = "PC_id", formula = df[x,]$POA_CODE_2016)
dtstudy <- genData(19, LGA1)
x=x+1


for (logoar in x:(length(df$LGA))) {
  LGA1 <- defData(varname = "Obese", dist = "binary", formula = df[x,]$Obese)
  LGA1 <- defData(LGA1,varname = "Smoking", dist = "binary", formula = df[x,]$Smoking)
  LGA1 <- defData(LGA1,varname = "Diabetes", dist = "binary", formula = df[x,]$Diabetes)
  LGA1 <- defData(LGA1,varname = "HeartDisease", dist = "binary", formula = df[x,]$HeartDisease)
  LGA1 <- defData(LGA1,varname = "MedianIncome", dist = "normal", formula = df[x,]$MedianIncome,variance=(df[x,]$MedianIncome*0.7)**2)
  LGA1 <- defData(LGA1,varname = "LGA_id", formula = df[x,]$Code)
  LGA1 <- defData(LGA1,varname = "PC_id", formula = df[x,]$POA_CODE_2016)
  dtstudy1 <- genData(19, LGA1)
  dtstudy<-rbind(dtstudy,dtstudy1)
  x=x+1
}

dtstudy$MedianIncome[dtstudy$MedianIncome<0] <- 0




dtstudy$Obese<-as.factor(dtstudy$Obese)

View(dtstudy)
count<-dtstudy %>% count(PC_id)

z=1
for (i in 2:length(dtstudy$PC_id)){
  z<-rbind(z,i)
}
dtstudy$index<-z[,1]
View(dtstudy)
library(ggplot2)

ggplot(data=dtstudy,aes(x=Obese,y=MedianIncome,color=as.factor(Diabetes),shape=as.factor(Smoking)))+geom_boxplot()+geom_jitter()

addressesPerPostcode <- 19



samplePCode <- function(pcode, number) {
  d <- fetch_postcodes(pcode)
  return(d[, .SD[sample(.N, min(number, .N))], by=.(POSTCODE)])
}

randomaddresses<-map(count$PC_id[1],
                      samplePCode,
                      number=count$n[1]) %>%
  bind_rows() %>%
  sf::st_as_sf(coords = c("LONGITUDE", "LATITUDE"))


for (i in 2:length(count$PC_id)){
  randomaddresses1<-map(count$PC_id[i],
                        samplePCode,
                        number=count$n[i]) %>%
    bind_rows() %>%
    sf::st_as_sf(coords = c("LONGITUDE", "LATITUDE"))
  randomaddresses<-rbind(randomaddresses,randomaddresses1)
}

y=1
for (i in 2:length(randomaddresses$POSTCODE)){
  y<-rbind(y,i)
}
randomaddresses$index<-y[,1]
View(randomaddresses)
dtstudy<-left_join(dtstudy, randomaddresses,by=c("index"="index"))
