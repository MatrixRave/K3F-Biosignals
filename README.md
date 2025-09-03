# K3F-Biosignals
This repository contains code created during a project at Hochschule Kaiserslautern. 

By running the docker compose file InfluxDB and Grafana are automatically created and linked. 
Basic Dashboards are provided via provisioning so users can diretly look into the data they record. 

All components are standalone and can be run without the need of running the other components. 
Please note that for running the shimmer part of this project the following device is needed: 
https://www.shimmersensing.com/product/shimmer3-gsr-unit/

Importing telemetry data live into the database is currently not supported, because a custom version of 
the Telemetrick extension for Assetto Corsa needs to be created. Since there is no out of the box mqtt
support there is currently a custom version in development which will support live export of telemetry data
from Assetto. 
Until this is done, a CSV export can be used to get the data into the database. Therefore this project 
provides an import option which inserts the data from the csv file into the database. 