# 🚗 ABS Soft Computing Case Study Simulator

An interactive Python and Streamlit-based educational simulator demonstrating the application of soft-computing techniques in an **Anti-lock Braking System (ABS)**. 

This project explores how computational intelligence—specifically **Takagi–Sugeno Fuzzy Logic**, **Artificial Neural Networks (ANN)**, **Particle Swarm Optimization (PSO)**, and **Sliding Mode Control (SMC)** concepts—can be integrated to regulate wheel slip during emergency braking under dynamic road friction conditions.

---

## 📌 Project Overview

During hard braking, wheels may lock up, drastically reducing tire grip and driver steering control. Standard ABS algorithms must quickly adapt to unpredictable surface transitions (e.g., dry asphalt to wet ice). 

This repository provides an interactive web application that simulates a **quarter-car ABS dynamics model** subjected to sudden road friction drop-offs, comparing a hybrid soft-computing control strategy against a traditional Proportional–Integral (PI) control baseline.

### Key Soft-Computing Components Incorporated:
1. **Fuzzy Logic (Takagi–Sugeno Inference):** Estimates real-time tire-road friction coefficients based on current vehicle velocity, slip error, and surface cues.
2. **Artificial Neural Network (ANN):** Approximates a nonlinear control law mapping using an MLP Regressor (`scikit-learn`), replacing discontinuous control switching to drastically reduce **chattering**.
3. **Particle Swarm Optimization (PSO):** Executes a global parameter search to determine an optimal auxiliary robust controller gain .
4. **Sliding Mode Control (SMC) & Feed-Forward:** Forms the primary mathematical trajectory-tracking control law regulating wheel slip ratio .

---

## 🚀 Interactive Application Features

* **Configurable Simulation Parameters:** Adjust initial vehicle speed (30–140 km/h), desired slip ratio , surface friction bounds, transition step timing, and maximum brake torque limits.
* **Real-time Controller Comparison:** Toggle between the **Hybrid Soft-Computing Controller** and the **Friction Feed-Forward + PI Baseline**.
* **Comprehensive 6-Panel Visualization:**
  - Wheel-Slip Tracking 
  - Applied Braking Torque 
  - Vehicle Chassis Velocity 
  - Wheel Rotational Angular Velocity 
  - Moving Slip Tracking Error 
  - Simulated vs. Fuzzy Estimated Road Friction Coefficient 
* **Performance Metrics Logging:** View quantitative results including stopping distance, stopping time, mean/max absolute slip error, and peak/RMS braking torque.
* **Image Export:** Built-in high-DPI graph plot export button for reports and presentations.

---
