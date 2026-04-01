
# AI for SE - Shoring / Sheet Pile Wall Design Prototype

This is a Streamlit prototype inspired by the sheet pile / shoring design sheet you shared.

## What it includes
- Input panel for soil, surcharge, pile geometry, and reinforcement
- Summary page with wall sketch and section view
- Analysis page with force calculations
- Capacity page with a simplified axial-flexural interaction diagram
- Shear check and embedment estimate
- CSV export of key results

## Important note
This is a planning/prototype tool for engineering review. It is **not** a sealed design package and the formulas should be checked against your exact office design procedure, current code interpretation, and project conditions.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
