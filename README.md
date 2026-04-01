
# RC Cantilever Shoring Wall Design - Streamlit Prototype

This version is focused on the **engineering design logic** visible in your screenshots rather than copying the static worksheet appearance.

## Included
- Input parameters taken from the sample image
- Force calculations using the visible equations
- IBC-style limit checks shown in the image
- ACI-style shear check
- Simplified circular RC axial-flexural interaction diagram
- Embedment estimate
- CSV export

## Important
This is still a prototype and should be checked against:
- your exact office design sheet
- final code interpretation
- project-specific geotechnical assumptions

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```
