#!/bin/sh
set -e
influx bucket create -n telemetry -r 150d
influx bucket create -n shimmer -r 150d