# EVProfiler
Holoviz app for neighborhood analysis of extracellular vesicle profiling data

## Installation
This is a holoviz app that can either be used as Jupyter Notebook or can be serverd using bokeh/panel.

1. Clone this repository
2. Create a new python environment and activate it  
```bash
python3 -m venv evprofiler\
source evprofiler\bin\activate
```
3. Install the python requirements  
```bash
pip install -r requirements.txt
```
4. In case this fails try installing wheels separately first.
5. Check that panel is working in python:
```python
import panel as pn
pn.extension()  
exit()
```
6. Either launch a Jupyter notebook server and use the notebook or serve the app using panel:  
```bash
panel serve Predictor.py --port 8080 --allow-websocket-origin 0.0.0.0:8080 &
```
Note: Adjust the allowed websocket and port as needed. For a guide how to configure a reverse proxy for bokeh look
[here](https://docs.bokeh.org/en/latest/docs/user_guide/server.html#basic-reverse-proxy-setup).

## Background information
This app is part of a scientific paper on extracellular vesicle characterization, which is currently under revision.  
A current version of the app is hosted at bornerlab.org:8080.  
For details on the mathematical and experimental background of the analysis please refer to the paper or the abouts section of the app.
