import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from loanprediction.config.core import config
from loanprediction.processing.features import EmploymentTypeImputer
from loanprediction.processing.features import Mapper
from loanprediction.processing.features import OutlierHandler,WeekdayOneHotEncoder

loanprediction_pipe = Pipeline([

    ######### Imputation ###########
    ('employmentTypeImputer_imputation', EmploymentTypeImputer(variable = config.model_config_.employmentType_var)),
    
    ######### Mapper ###########
    ('map_hasMortgage', Mapper(variable = config.model_config_.hasMortgage_var, mappings = config.model_config_.hasMortgage_mappings)),
    ('map_hasDependents', Mapper(variable = config.model_config_.hasDependents_var, mappings = config.model_config_.hasDependents_mappings)),
    ('map_hasCoSigner', Mapper(variable = config.model_config_.hasCoSigner_var, mappings = config.model_config_.hasCoSigner_mappings)),
    
    
    ######## Handle outliers ########
    ('handle_outliers_age', OutlierHandler(variable = config.model_config_.age_var)),
    ('handle_outliers_income', OutlierHandler(variable = config.model_config_.income_var)),
    ('handle_outliers_loanAmount', OutlierHandler(variable = config.model_config_.loanAmount_var)),
    ('handle_outliers_creditScore', OutlierHandler(variable = config.model_config_.creditScore_var)),
    ('handle_outliers_monthsEmployed', OutlierHandler(variable = config.model_config_.monthsEmployed_var)),
    ('handle_outliers_numCreditLines', OutlierHandler(variable = config.model_config_.numCreditLines_var)),
    ('handle_outliers_interestRate', OutlierHandler(variable = config.model_config_.interestRate_var)),
    ('handle_outliers_loanTerm', OutlierHandler(variable = config.model_config_.loanTerm_var)),
    ('handle_outliers_dTIRatio', OutlierHandler(variable = config.model_config_.dTIRatio_var)),
    ######## One-hot encoding ########
    ('encode_education', WeekdayOneHotEncoder(variable = config.model_config_.education_var)),

    ('encode_employmentType', WeekdayOneHotEncoder(variable = config.model_config_.employmentType_var)),
    ('encode_maritalStatus', WeekdayOneHotEncoder(variable = config.model_config_.maritalStatus_var)),
    ('encode_loanPurpose', WeekdayOneHotEncoder(variable = config.model_config_.loanPurpose_var)),
    #('encode_employmentType', WeekdayOneHotEncoder(variable = config.model_config_.employmentType_var)),
    #('encode_employmentType', WeekdayOneHotEncoder(variable = config.model_config_.employmentType_var)),
    #('encode_employmentType', WeekdayOneHotEncoder(variable = config.model_config_.employmentType_var)),
    #('encode_employmentType', WeekdayOneHotEncoder(variable = config.model_config_.employmentType_var)),
    #('encode_employmentType', WeekdayOneHotEncoder(variable = config.model_config_.employmentType_var)),

    # Scale features
    ('scaler', StandardScaler()),
    
    # Regressor
    ('model_rf', RandomForestClassifier(n_estimators = config.model_config_.n_estimators, 
                                       max_depth = config.model_config_.max_depth,
                                       class_weight='balanced',
                                      random_state = config.model_config_.random_state))
    
    ])
