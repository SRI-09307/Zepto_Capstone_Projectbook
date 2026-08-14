from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                             recall_score, f1_score, roc_curve, roc_auc_score,
                             mean_absolute_error, mean_squared_error, r2_score)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

# ONE raw-data load, with committed CSV as the offline fallback.
local_csv = ROOT / 'titanic.csv'
try:
    df = sns.load_dataset('titanic')
    df.to_csv(local_csv, index=False)
except Exception:
    df = pd.read_csv(local_csv)

# Standardize column names for the Seaborn version and remove duplicate rows.
df.columns = [c.lower() for c in df.columns]
# The committed fallback has 12 columns; seaborn has extra derived columns.
keep = ['survived','pclass','sex','age','sibsp','parch','fare','embarked']
df = df[keep].copy()

# ---------------- Part A: profiling and cleaning ----------------
profile = pd.DataFrame({
    'dtype': df.dtypes.astype(str),
    'missing_count': df.isna().sum(),
    'missing_pct': (df.isna().mean()*100).round(2)
})
profile.to_csv(OUT / 'missing_values_report.csv')

with open(OUT / 'profile.txt','w') as f:
    f.write('SHAPE\n' + str(df.shape) + '\n\nINFO\n')
    df.info(buf=f)
    f.write('\n\nDESCRIBE\n' + df.describe(include='all').to_string())

# Missing-value rule: <5% drop rows; 5-30% impute; >30% drop column.
clean = df.copy()
missing_notes = []
for col in clean.columns:
    pct = clean[col].isna().mean()*100
    if pct == 0:
        continue
    if pct < 5:
        before = len(clean)
        clean = clean.dropna(subset=[col])
        missing_notes.append(f'{col}: {pct:.2f}% missing -> dropped affected rows (<5%).')
    elif pct <= 30:
        if pd.api.types.is_numeric_dtype(clean[col]):
            fill = clean[col].median()
            clean[col] = clean[col].fillna(fill)
            missing_notes.append(f'{col}: {pct:.2f}% missing -> median imputation (5-30%).')
        else:
            fill = clean[col].mode()[0]
            clean[col] = clean[col].fillna(fill)
            missing_notes.append(f'{col}: {pct:.2f}% missing -> mode imputation (5-30%).')
    else:
        clean = clean.drop(columns=[col])
        missing_notes.append(f'{col}: {pct:.2f}% missing -> column dropped (>30%) because imputation would be unreliable.')
clean = clean.drop_duplicates().reset_index(drop=True)
with open(OUT / 'cleaning_notes.txt','w') as f:
    f.write('\n'.join(missing_notes))
    f.write(f'\nFinal cleaned shape: {clean.shape}\n')
clean.to_csv(OUT / 'titanic_cleaned.csv', index=False)

# Outliers and skewness.
out_rows=[]
for col in ['age','fare']:
    q1,q3=clean[col].quantile([.25,.75]); iqr=q3-q1
    lo,hi=q1-1.5*iqr,q3+1.5*iqr
    out_rows.append({'column':col,'outlier_count':int(((clean[col]<lo)|(clean[col]>hi)).sum()),'skewness':clean[col].skew()})
pd.DataFrame(out_rows).to_csv(OUT/'outlier_report.csv',index=False)

# Histograms and boxplots.
for col in ['age','fare']:
    plt.figure(figsize=(7,4)); plt.hist(clean[col].dropna(), bins=20); plt.title(f'{col.title()} distribution'); plt.tight_layout(); plt.savefig(OUT/f'{col}_histogram.png',dpi=160); plt.close()
    plt.figure(figsize=(7,3)); plt.boxplot(clean[col].dropna(), vert=False); plt.title(f'{col.title()} box plot'); plt.tight_layout(); plt.savefig(OUT/f'{col}_boxplot.png',dpi=160); plt.close()

# Bivariate survival summaries.
for cols,name in [(['sex'],'survival_by_sex'),(['pclass'],'survival_by_pclass'),(['sex','pclass'],'survival_by_sex_pclass')]:
    tab=clean.groupby(cols,dropna=False)['survived'].agg(['count','mean']).reset_index()
    tab['survival_rate_pct']=(tab['mean']*100).round(2)
    tab.to_csv(OUT/f'{name}.csv',index=False)

# Correlation matrix: exactly six requested columns.
corr_cols=['survived','pclass','age','sibsp','parch','fare']
corr=clean[corr_cols].corr()
corr.to_csv(OUT/'correlation_matrix.csv')
plt.figure(figsize=(7,5));
try:
    sns.heatmap(corr,annot=True,fmt='.2f')
except Exception:
    plt.imshow(corr,aspect='auto'); plt.colorbar(); plt.xticks(range(len(corr_cols)),corr_cols,rotation=45); plt.yticks(range(len(corr_cols)),corr_cols)
plt.title('Titanic correlation heatmap'); plt.tight_layout(); plt.savefig(OUT/'correlation_heatmap.png',dpi=180); plt.close()

pairs=[]
for i,a in enumerate(corr_cols):
    for b in corr_cols[i+1:]: pairs.append((a,b,corr.loc[a,b]))
pairs=sorted(pairs,key=lambda x:abs(x[2]),reverse=True)[:2]
with open(OUT/'correlation_interpretation.txt','w') as f:
    for a,b,v in pairs: f.write(f'{a} and {b}: correlation={v:.3f}. Absolute magnitude ranks among the two strongest off-diagonal relationships.\n')

# Four distinct multivariate charts with interpretations.
tab=clean.groupby(['sex','pclass'])['survived'].mean().unstack(); plt.figure(figsize=(7,5)); tab.plot(kind='bar',ax=plt.gca()); plt.ylabel('Survival rate'); plt.title('Survival by sex and passenger class'); plt.tight_layout(); plt.savefig(OUT/'chart1_survival_sex_class.png',dpi=180); plt.close()
plt.figure(figsize=(7,5));
for val in [0,1]: plt.hist(clean.loc[clean.survived==val,'age'],bins=20,alpha=.6,label=f'survived={val}')
plt.legend(); plt.title('Age distribution by survival'); plt.tight_layout(); plt.savefig(OUT/'chart2_age_survival.png',dpi=180); plt.close()
plt.figure(figsize=(7,5)); clean.boxplot(column='fare',by=['pclass','survived'],ax=plt.gca()); plt.suptitle(''); plt.title('Fare by class and survival'); plt.tight_layout(); plt.savefig(OUT/'chart3_fare_class_survival.png',dpi=180); plt.close()
plt.figure(figsize=(7,5));
for sex in clean.sex.unique():
    sub=clean[clean.sex==sex]; plt.scatter(sub.age,sub.fare,c=sub.survived,alpha=.7,label=sex)
plt.xlabel('Age'); plt.ylabel('Fare'); plt.title('Age vs fare by survival and sex'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'chart4_age_fare.png',dpi=180); plt.close()
with open(OUT/'chart_interpretations.txt','w') as f:
    f.write('1. Survival is substantially higher among females, with passenger class also separating outcomes.\n')
    f.write('2. Survivors are concentrated at younger ages, although survival occurs across the age range.\n')
    f.write('3. Higher fares are associated with higher-class passengers and generally better survival outcomes.\n')
    f.write('4. Age and fare together show that survival patterns vary by both demographic and ticket characteristics.\n')

# Exploratory standardization sanity check (not used in modeling).
std_check=clean[['age','fare']].copy(); before=std_check.agg(['mean','std'])
std_check=(std_check-std_check.mean())/std_check.std(ddof=1); after=std_check.agg(['mean','std'])
with open(OUT/'standardization_check.txt','w') as f:
    f.write('BEFORE\n'+before.to_string()+'\n\nAFTER\n'+after.to_string())

# ---------------- Part B: predictive modeling ----------------
features=['pclass','sex','age','sibsp','parch','fare','embarked']
X=clean[features]; y=clean['survived']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
num=['pclass','age','sibsp','parch','fare']; cat=['sex','embarked']
preprocessor=ColumnTransformer([
    ('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),num),
    ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)
])
models={
    'Logistic Regression': LogisticRegression(max_iter=1000,random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=5,random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200,random_state=42)
}
rows=[]; fitted={}
for name,est in models.items():
    pipe=Pipeline([('preprocessor',preprocessor),('model',est)])
    pipe.fit(X_train,y_train); fitted[name]=pipe
    pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
    rows.append({'model':name,'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred),'recall':recall_score(y_test,pred),'f1':f1_score(y_test,pred),'auc':roc_auc_score(y_test,prob)})
    pd.DataFrame(confusion_matrix(y_test,pred),index=['actual_0','actual_1'],columns=['pred_0','pred_1']).to_csv(OUT/f'{name.lower().replace(" ","_")}_confusion_matrix.csv')

# Decision tree visualization.
dt=fitted['Decision Tree']; names=dt.named_steps['preprocessor'].get_feature_names_out()
plt.figure(figsize=(18,9)); plot_tree(dt.named_steps['model'],feature_names=names,class_names=['Not survived','Survived'],filled=True,max_depth=3,fontsize=7); plt.tight_layout(); plt.savefig(OUT/'decision_tree.png',dpi=160); plt.close()

# ROC curves.
plt.figure(figsize=(7,5))
for name,pipe in fitted.items():
    prob=pipe.predict_proba(X_test)[:,1]; fpr,tpr,_=roc_curve(y_test,prob); auc=roc_auc_score(y_test,prob); plt.plot(fpr,tpr,label=f'{name} (AUC={auc:.3f})')
plt.plot([0,1],[0,1],'k--'); plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.title('ROC comparison'); plt.legend(); plt.tight_layout(); plt.savefig(OUT/'roc_comparison.png',dpi=180); plt.close()

comparison=pd.DataFrame(rows).sort_values('f1',ascending=False)
comparison.to_csv(OUT/'model_comparison.csv',index=False)

# Imbalance comparison using one classifier (Random Forest).
imb=[]
base=Pipeline([('preprocessor',preprocessor),('model',RandomForestClassifier(n_estimators=200,random_state=42))]); base.fit(X_train,y_train)
for label,pipe in [('baseline',base)]: pass
balanced=Pipeline([('preprocessor',preprocessor),('model',RandomForestClassifier(n_estimators=200,class_weight='balanced',random_state=42))]); balanced.fit(X_train,y_train)
smote=ImbPipeline([('preprocessor',preprocessor),('smote',SMOTE(random_state=42)),('model',RandomForestClassifier(n_estimators=200,random_state=42))]); smote.fit(X_train,y_train)
for label,pipe in [('baseline',base),('class_weight_balanced',balanced),('smote_training_only',smote)]:
    pred=pipe.predict(X_test); prob=pipe.predict_proba(X_test)[:,1]
    imb.append({'strategy':label,'precision':precision_score(y_test,pred),'recall':recall_score(y_test,pred),'f1':f1_score(y_test,pred),'auc':roc_auc_score(y_test,prob)})
pd.DataFrame(imb).to_csv(OUT/'imbalance_comparison.csv',index=False)
with open(OUT/'class_balance.txt','w') as f: f.write('Overall class counts:\n'+y.value_counts().sort_index().to_string()+'\n\nProportions:\n'+y.value_counts(normalize=True).sort_index().to_string())

# Hyperparameter tuning with OOB scoring enabled.
rf_pipe=Pipeline([('preprocessor',preprocessor),('model',RandomForestClassifier(oob_score=True,random_state=42,n_jobs=-1))])
param_grid={'model__n_estimators':[100,200],'model__max_depth':[None,5,10],'model__max_features':['sqrt','log2']}
grid=GridSearchCV(rf_pipe,param_grid,cv=5,scoring='f1',n_jobs=-1); grid.fit(X_train,y_train)
best=grid.best_estimator_; oob=best.named_steps['model'].oob_score_
pd.DataFrame(grid.cv_results_).sort_values('rank_test_score').head(10).to_csv(OUT/'gridsearch_results_top10.csv',index=False)
with open(OUT/'hyperparameter_tuning.txt','w') as f: f.write(f'Best parameters: {grid.best_params_}\nBest CV F1: {grid.best_score_:.4f}\nOOB score: {oob:.4f}\n')

# Regression side-task: predict fare from the other available features.
reg_features=['survived','pclass','sex','age','sibsp','parch','embarked']
RX=clean[reg_features]; ry=clean['fare']; RX_train,RX_test,ry_train,ry_test=train_test_split(RX,ry,test_size=.2,random_state=42)
rpre=ColumnTransformer([('num',Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())]),['survived','pclass','age','sibsp','parch']),('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),['sex','embarked'])])
reg=Pipeline([('preprocessor',rpre),('model',LinearRegression())]); reg.fit(RX_train,ry_train); rpred=reg.predict(RX_test)
r2=r2_score(ry_test,rpred); adj=1-(1-r2)*(len(ry_test)-1)/(len(ry_test)-RX_test.shape[1]-1)
with open(OUT/'regression_results.txt','w') as f: f.write(f'MAE: {mean_absolute_error(ry_test,rpred):.4f}\nRMSE: {np.sqrt(mean_squared_error(ry_test,rpred)):.4f}\nR2: {r2:.4f}\nAdjusted R2: {adj:.4f}\n')
res=ry_test-rpred
plt.figure(figsize=(7,4)); plt.scatter(rpred,res); plt.axhline(0,ls='--'); plt.xlabel('Predicted fare'); plt.ylabel('Residual'); plt.title('Fare regression residual plot'); plt.tight_layout(); plt.savefig(OUT/'regression_residuals.png',dpi=180); plt.close()

# Save complete fitted preprocessing + estimator pipeline.
joblib.dump(best, ROOT/'titanic_final_pipeline.joblib')

best_name=comparison.iloc[0]['model']
with open(OUT/'final_recommendation.txt','w') as f:
    f.write(f'Recommended classifier by held-out F1: {best_name}.\n')
    f.write('Use the tuned Random Forest artifact for deployment because it includes preprocessing and the final estimator in one reloadable pipeline.\n')

print('Titanic analytics completed successfully.')
print(comparison.to_string(index=False))
print('Best RF parameters:', grid.best_params_, 'OOB:', round(oob,4))
