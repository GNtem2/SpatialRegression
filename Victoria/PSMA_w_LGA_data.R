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
df<-read.csv("./Postcode_LGA_w_Data.csv")
df<-subset(df,Addresses==1)

gdt<- st_read("/home/richardb/Projects/GeospatialStroke/ABSData/Boundaries/POA_2016_AUST.shp")
gdt$POA_CODE16<-as.integer(gdt$POA_CODE16)
df$Code<-as.integer(df$Code)
gdt<-left_join(df, gdt,by=c("POA_CODE_2016"="POA_CODE16"))
gdt<-st_as_sf(gdt)



oneLGA <- function(idx, df, n=19) {
  LGA1 <- defData(varname = "Obese", dist = "binary", formula = df[idx,]$Obese)
  LGA1 <- defData(LGA1,varname = "Smoking", dist = "binary", formula = df[idx,]$Smoking)
  LGA1 <- defData(LGA1,varname = "Diabetes", dist = "binary", formula = df[idx,]$Diabetes)
  LGA1 <- defData(LGA1,varname = "HeartDisease", dist = "binary", formula = df[idx,]$HeartDisease)
  LGA1 <- defData(LGA1,varname = "MedianIncome", dist = "normal", formula = df[idx,]$MedianIncome,variance=(df[idx,]$MedianIncome*0.7)**2)
  LGA1 <- defData(LGA1,varname = "LGA_id", formula = df[idx,]$Code)
  LGA1 <- defData(LGA1,varname = "PC_id", formula = df[idx,]$POA_CODE_2016)
  dtstudy <- genData(n, LGA1)
  return(dtstudy)
}

dtstudy <- map(1:nrow(df), ~oneLGA(.x, df, n=19))
dtstudy <- bind_rows(dtstudy)

dtstudy$MedianIncome[dtstudy$MedianIncome<0] <- 0




dtstudy$Obese<-as.factor(dtstudy$Obese)

View(dtstudy)
dtcount<-dtstudy %>% count(PC_id)

dtstudy <- mutate(dtstudy, index = row_number())
# or
# dtstudy$index <- 1:nrow(dtstudy)
library(ggplot2)

ggplot(data=dtstudy,aes(x=Obese,y=MedianIncome,color=as.factor(Diabetes),shape=as.factor(Smoking)))+geom_boxplot()+geom_jitter()

addressesPerPostcode <- 19



samplePCode <- function(pcode, number) {
  d <- fetch_postcodes(pcode)
  return(d[, .SD[sample(.N, min(number, .N))], by=.(POSTCODE)])
}

randomaddresses <- NULL

randomaddresses <- map2_df(dtcount$PC_id, dtcount$n, ~samplePCode(.x, .y))
randomaddresses <- sf::st_as_sf(randomaddresses, coords = c("LONGITUDE", "LATITUDE"))
randomaddresses <- mutate(randomaddresses, index = row_number())

# No real need to use joins here - could cbind.
dtstudy<-left_join(dtstudy, randomaddresses,by=c("index"="index"))
