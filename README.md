# Solar-Prediction

During this study, a prototype of the effective use of a small solar power plant was created. The main idea of it was to turn on certain electrical devices when the overflow of the generated energy is forecasted. Devices should not be necessary at the moment but their operation should be beneficial, for example - charging the electrical vehicle or other electronic devices. When a solar power plant is generating more energy than the household can utilize, overflow energy is being sent to the main grid of the electricity provider for storage. When a household is experiencing an energy shortage, the same energy is being bought back at the higher price which means some part of the generated energy is being wasted during this process. During this study, the data was collected from solar inverters, historical weather API tool, and smart consumed energy meters. The study describes collected data preparation and feature engineering process that was later used to create various models. Two main parts of the created prototype are different forecasting models - one that forecasts the energy generation of a solar power plants, the other that forecasts energy consumption of a household. Both of the model types were tested with Facebook Prophet and different neural network architectures - Feed-forward, Long Short-Term Memory, and Gated Recurrent Unit networks. In addition, a baseline model was developed for forecasting energy generation. 

## Setup environment
    conda env create -f environment.yaml
    conda activate Solar-Prediction

## File structure
    .
    ├── Data-Files
    │   ├── consumption_data.csv         # Collected consumption data of 8 households. Only the households 1 and 5 were used.
    │   ├── solar_5k_generation.csv      # Collected generation data of Solar 5K solar power plant
    │   ├── solar_30k_generation.csv     # Collected generation data of Solar 30K solar power plant     
    │   └── weather_data.csv             # Historical weather data collected from an API
    │
    ├── Model-Creation                    
    │   ├── config.py                    # Configuration file with global variables
    │   ├── data_preparation.py          # Weather data prepration function and functions to prepare data for using in different models 
    │   ├── feature_encodings.py         # Feature engineering functions
    │   ├── main.py                      # File that should be ran to perform model evaluation
    │   ├── models_baseline.py           # Calculating prediction values and testing baseline model
    │   ├── models_nn.py                 # Training and testing different Neural-Network based models
    │   ├── models_prophet.py            # Facebook Prophet model training and testing
    │   └── scaler.py                    # Custom scaler to perform scaling on a dataframe, with intension to later be used in C++ environment
    │
    ├── Model-Results                    
    |   ├── Plots                        # Comparison plots of baseline, prophet and most accurate Neural-Network based models
    │   ├── baseline_results.csv         # Results acquired with baseline models
    │   ├── nn_results.csv               # Results acquired with Neural-Network based models
    │   └── prophet_results.csv          # Results acquired with Facebook Prophet models
    │
    ├── environment.yaml                 # Conda environment setup file
    └── README.md
