# SpatialRegression



## Spatial Regression with INLA

This is data published in Jama 29/4/2020 on COVD-19 in New York. The New York borough shapefiles were obtained from New York Open Data at https://data.cityofnewyork.us/City-Government/Borough-Boundaries/tqmj-j8zm. For those wishing to evaluate other datasets, there’s lung cancer data in SpatialEpi library, lip cancer, leukemia in DClusterm library. 

Key aspect of spatial regression is that neighbouring regions are similar and distant regions are less so. It uses the polyn2nb in spdep library to create the neighbourhood weight. This section uses Bayesian modeling for regression with fitting of the model by  INLA.
https://www.r-bloggers.com/spatial-data-analysis-with-inla/. The Rmd file is contained within the NewYork folder.

Firstly, we estimate the calculate the raw rate of disease (cases/total population at risk). The expected number of cases is the rate multiply by the population in each suburb. SMR is the number of cases divided by the expected number of cases.

## Spatial Regression with rstan

The Rmd file is contained within the NewYork folder along with the rstan files. This analysis is still in development as I ran into problems with rstan and trying to reinstall rstan. The rstan analysis takes a lot longer to perform than with INLA.

## Spatio-temporal regression with INLA

## Git Bash
echo "# SpatialRegression" >> README.md
git init
git add README.md
git commit -m "first commit"
git remote add origin https://github.com/GNtem2/SpatialRegression.git
git push -u origin master