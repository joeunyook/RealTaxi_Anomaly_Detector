# RealTaxi Anomaly Detector

## Overview

This project builds a real-time anomaly detection system for NYC taxi activity data.
The goal is to detect sudden drops, spikes, or unusual behavior in city-level taxi demand.

## Dataset

We use the NYC Taxi dataset from the Numenta Anomaly Benchmark 
The data is a 30-minute aggregated time-series of passenger counts with labeled anomalies.

## Objective
Our model generates an anomaly score at each time step and flags abnormal events.
Performance is evaluated using AUC-ROC and detection delay.

## Approach
We compare traditional methods (LOF, Isolation Forest, KNN), incremental models, ensemble methods, and deep temporal models (CNN, GRU, LSTM, Transformer) with our custom built anomaly detector.
