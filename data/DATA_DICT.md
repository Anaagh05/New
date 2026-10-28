# Customer Churn — Data Dictionary

## Raw Features

| Column              | Type    | Description                                      |
|---------------------|---------|--------------------------------------------------|
| customerID          | string  | Unique customer identifier (dropped)            |
| gender              | string  | Male / Female                                    |
| SeniorCitizen       | int     | 1 = Senior Citizen                               |
| Partner             | string  | Whether customer has a partner                   |
| Dependents          | string  | Whether customer has dependents                  |
| tenure              | int     | Months as customer                               |
| PhoneService        | string  | Phone service subscription                       |
| MultipleLines       | string  | Multiple phone lines                             |
| InternetService     | string  | DSL / Fiber optic / No                           |
| OnlineSecurity      | string  | Online security add-on                           |
| OnlineBackup        | string  | Online backup add-on                             |
| DeviceProtection    | string  | Device protection add-on                         |
| TechSupport         | string  | Tech support add-on                              |
| StreamingTV         | string  | Streaming TV add-on                              |
| StreamingMovies     | string  | Streaming movies add-on                          |
| Contract            | string  | Month-to-month / One year / Two year             |
| PaperlessBilling    | string  | Paperless billing enabled                        |
| PaymentMethod       | string  | Payment method                                   |
| MonthlyCharges      | float   | Monthly charge amount                            |
| TotalCharges        | float   | Total charge (may have blanks → imputed)         |
| Churn               | string  | **Target** — Yes / No                            |

## Notes
- `TotalCharges` whitespace entries are imputed with median before scaling
- All categoricals label-encoded; numerics standard-scaled
- Train/test split: 80/20 stratified on `Churn`
