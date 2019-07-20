import pandas as pd
predict_full = pd.read_csv('../../data/output/lgb_test_stacking_full_v1.csv')
predict_full['label'] = predict_full[['0','1','2','3','4','5']].idxmax(axis=1)
predict_full['label'] = predict_full['label'].astype('int') + 1
predict_full.rename(columns={'uId': 'id'}, inplace=True)
predict_full = predict_full[['id','label']]
predict_full.to_csv('../../data/output/submission_full.csv', index=False, encoding='utf-8')

predict_na = pd.read_csv('../../data/output/lgb_model_score_stacking_full_v1.csv')
predict_na['label'] = predict_na[['0','1','2','3','4','5']].idxmax(axis=1)
predict_na['label'] = predict_na['label'].astype('int') + 1
predict_na.rename(columns={'uId': 'id'}, inplace=True)
predict_na = predict_na[['id','label']]
predict_na.to_csv('../../data/output/submission_na.csv', index=False, encoding='utf-8')

predict_full = pd.read_csv('../../data/output/submission_full.csv')
predict_2 = pd.read_csv('../../data/output/submission_na.csv')
usage_test = pd.read_csv('../../data/processed_data/age_test_na.csv')

usage_test.rename(columns={'uId': 'id'}, inplace=True)
predict = pd.merge(predict_full,usage_test,how='inner',on='id')
predict = pd.concat([predict_2, predict])
predict = predict.sort_values(by=('id'))
predict.to_csv('../../data/submit/submission.csv', index=False, encoding='utf-8')