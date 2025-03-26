# README - Strava Commute Project 🚴‍♂️

## Overview

The Strava Commute Project is an innovative initiative designed to automate the extraction, analysis, and visualization of cycling commute activities. By leveraging machine learning and data automation, this project provides valuable insights into commuting patterns, carbon savings, and cyclist behaviors. The ultimate goal is to enhance urban mobility planning, support sustainable transportation, and optimize commute tracking for cyclists worldwide.

## Why This Matters 🚀

- **For Employers & Businesses**: Gain insights into employee commuting habits and promote sustainability initiatives.
- **For Cyclists**: Get personalized commute tracking, data-driven insights, and carbon-saving statistics.

## Project Components

<div align="center">
  <img src="flowchart b2w.png">
</div>

### 1. **ETL Process** ⚙️

A fully automated pipeline for data extraction, transformation, and storage:

- **Data Source**: Strava API
- **Transformation**: Data cleaning, aggregation, and enrichment
- **Storage**: Google Sheets for structured and accessible data
- **Automation**: GitHub Actions (current) with Google Cloud Functions (future expansion)

### 2. **Machine Learning Model** 🧠

Using AI to classify cycling commute activities:

- **Algorithms**:
  - Random Forest (_Python deployment, currently in use_)
  - Decision Tree (JavaScript deployment for web integration)
  - TensorFlow (planned as dataset grows)
- **Training Dataset**: Currently <10k records, aiming for scalable expansion

### 3. **Visualization & Insights** 📊

Transforming raw data into actionable insights through interactive dashboards:

- **Tools**: Tableau & Looker Studio
- **Dashboard Features**:
  - 🚴‍♂️ Commute activity trends
  - 🌍 Carbon savings and sustainability impact
  - 📈 Commuter behavior insights

## End-to-End Pipeline 🔄
This section provides a step-by-step breakdown of how the Strava Commute Project processes data from start to finish:

1. **Data Extraction** 🛠️
   - Fetches cycling activity data via the Strava API from a club.
   - Retrieves details such as activity type, timestamp, and user metadata (excluding location data).
   - Data extraction is done for every 5 minute.

2. **Transformation & Processing** 🔍
   - Cleans and preprocesses raw data by handling missing values and structuring records.
   - Aggregates commute-related activities based on predefined filtering criteria.
   - Enhances dataset with computed fields, such as estimated carbon savings.

3. **Machine Learning Classification** 🤖
   - Runs classification models (Random Forest/Decision Tree) to predict if an activity is a commute.
   - Stores results in Google Sheets for further validation and analysis.

4. **Storage & Automation** 📂
   - Saves processed data in Google Sheets for easy access and integration.
   - Automates the entire workflow using GitHub Actions (current) and plans migration to Google Cloud Functions for better scalability.

5. **Data Visualization & Reporting** 📈
   - Uses Tableau and Looker Studio to generate interactive dashboards.
   - Provides insights into cycling behaviors, carbon impact, and commute trends.

## Technologies & Tools 🔧

- **Programming Languages**: Python, JavaScript
- **Data Storage**: Google Sheets (scalable to BigQuery in the future)
- **Automation & Scheduling**: GitHub Actions, Google Cloud Functions
- **Data Processing**: Python
- **Visualization**: Tableau & Looker Studio

## Future Roadmap 🌍

- **Enhance ML model accuracy** to improve commute classification
- **Integrate real-time analytics** for live tracking and reporting
- **Expand to other transportation modes**, including walking and e-scooters

## Get Involved 🤝

We welcome collaborations, contributions, and partnerships!

- **Employers**: Promote eco-friendly commuting and track employee activity.
- **Developers**: Help improve the automation and ML models.

### Want to contribute?

Feel free to fork this repository, submit pull requests, or reach out for collaborations.

---

This README will be continuously updated as the project evolves. Stay tuned! 🚴‍♀️

