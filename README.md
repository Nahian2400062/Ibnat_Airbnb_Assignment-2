# Ibnat_Airbnb_Assignment-2
# Ibnat Airbnb Assignment - Analysis & Modeling

## Overview

This repository contains an end-to-end analysis project on Airbnb data. The analysis covers data exploration, feature engineering, model building, evaluation, and interpretability using various regression and machine learning methods. The project is primarily implemented in R and includes an R Markdown document, supporting scripts, detailed visualizations, and a discussion of results.

## Repository Structure

- **Ibnat_Airbnb.Rmd**  
  The main R Markdown file with the full analysis workflow—from data cleaning and exploration to modeling and evaluation.

- **Ibnat_Airbnb.html**  
  The HTML report generated from the R Markdown file. It provides an interactive narrative of the analysis.

- **amsterdam_airbnb.R**  
  An R script focused on analyzing Amsterdam-specific Airbnb data.

- **Airbnb_Result_Discussion.pdf**  
  A PDF document detailing the discussion of the results, including insights and conclusions drawn from the analysis.

- **Images and Plots**  
  These files include key visualizations generated during the analysis:  
  - `lasso_cv_plot.png`: LASSO regression cross-validation performance.  
  - `ridge_cv_plot.png`: Ridge regression cross-validation performance.  
  - `model_comparison_rmse.png`: Comparison of RMSE values for different models.  
  - `model_rmse_plot.png`: Detailed RMSE plot for model evaluation.  
  - `rf_feature_importance.png` & `rf_feature_importance_top10.png`: Feature importance plots from Random Forest models.  
  - `shap_global_top20.png` & `shap_xgb_explanation.png`: SHAP value plots to help interpret model predictions.  
  - `xgb_feature_importance_hd.png`, `xgb_feature_importance_top10.png`, & `xgb_shap_summary.png`: Feature importance and SHAP summary plots for the XGBoost model.

- **LICENSE**  
  The project is released under the MIT License.


## Data and Variables

The analysis uses an Airbnb dataset containing various features that describe each listing. While the exact dataset may vary, typical variables include:

- **ID**: Unique identifier for each listing.
- **Name**: Title or name of the Airbnb listing.
- **Host_ID**: Identifier for the host managing the listing.
- **Neighbourhood/Location**: Geographic area where the listing is located.
- **Latitude & Longitude**: Geographic coordinates.
- **Room_Type**: Type of accommodation (e.g., entire home, private room, shared room).
- **Price**: Nightly price of the listing.
- **Minimum_Nights**: Minimum required booking duration.
- **Number_of_Reviews**: Total count of reviews received.
- **Last_Review**: Date of the most recent review.
- **Reviews_Per_Month**: Average number of reviews per month.
- **Calculated_Host_Listings_Count**: Total number of listings the host manages.
- **Availability_365**: Number of days the listing is available in a year.

*Note: Specific variable names and additional features might be present. Refer to the code comments and data exploration sections in the R Markdown file for a detailed description of the dataset used.*

## Handling Missing Data

Handling missing data is an important part of the analysis. The following strategies were implemented:

- **Exploratory Analysis**: A preliminary investigation was conducted to understand the extent of missing data in each variable.
- **Imputation**: For numerical features with minor missingness, imputation using the median or mean was applied.
- **Exclusion/Flagging**: Variables or observations with a high percentage of missing values were either removed or flagged for special consideration.
- **Documentation**: All methods and decisions regarding missing data are documented within the analysis narrative in the R Markdown file.

## Installation and Usage

### Prerequisites

- **R**: Version 4.0 or later is recommended.
- **RStudio**: For a user-friendly development environment.
- **Required R Packages**: (Install using `install.packages()` in R)
  - `tidyverse`
  - `caret`
  - `randomForest`
  - `xgboost`
  - `glmnet`
  - `shapley` (or similar for SHAP analysis)
  - *Additional packages as required by the scripts.*

### Running the Analysis

1. **R Markdown Analysis**:  
   Open `Ibnat_Airbnb.Rmd` in RStudio and click the “Knit” button to generate the HTML report (`Ibnat_Airbnb.html`).

2. **Script Execution**:  
   Run the `amsterdam_airbnb.R` script in your R environment to execute the analysis specific to Amsterdam Airbnb data.

3. **Reviewing Results**:  
   - View the generated plots (located in the repository) for visual insights into model performance and feature importance.  
   - Read through `Airbnb_Result_Discussion.pdf` for an in-depth discussion of the outcomes and conclusions.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for further details.

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgements

Thank you to all contributors and resources that have supported the development of this project.
