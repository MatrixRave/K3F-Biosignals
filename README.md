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
provides an import option which inserts the data from the csv file into the database. Therefore in the Influx web view 
it is currently necessary to create a telemetry bucket that the data is written into. The influx docker image does not support
the creation of multiple buckets when the container is created in the docker file but a solution using a bash script and an entry point 
is currently being worked on. 
To import telemetry data into the database, the import_csv file can be used. Make sure the files you import provide a timestamp in the filename
when the recording started. The script will the automatically generate the timestamps according to the provided time in the data. Currently this
script has only been tested for csv-files recorded with Telemetrick, but it should run fine with other files using the same approach. 

To use the gazeTracking module a camera is required to capture the live image that gets analyzed. It is recommmended 
to use a camera with a resolution of at least 1080p and at least 30 frames per second. 

To ensure that data is written into the database make sure the container is running when running the gazeTracking module. 
This of course has also to be ensured for all other modules. 

Altough it is possible to livestream data from the shimmer sensor unit directly into the database using a device running on macOS, 
it is the developers recomendation to use a windows device. This is because the connection between the devices is more stable and 
it is easier to pair them. Writing data from the sensor units sd-card  into the data base is possible on both platforms by running the 
shimmer_prerecorded file in the shimmer3 folder of this project. 