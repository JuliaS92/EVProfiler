#!/bin/bash
source /opt/pythonenvs/holoviz/bin/activate
cd /opt/gitrepos/EVProfiler
panel serve Predictor.py --port 8080 --allow-websocket-origin 141.5.106.250:8080 --allow-websocket-origin evprofiler.bornerlab.org >> /var/log/evprofiler/panel.log
