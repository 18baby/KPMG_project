# 사용함수 총정리

# 모든 입력값이 null인 행 제거
def x_null_drop(df):
    select_column = ['부처', '법령명', '조번호', '항번호', '호번호', '조문제목', '조문']
    delete_row_idx = list(df[df[select_column].isnull().all(axis = 1)].index)
    delete_row_idx.sort(reverse = True)
    for i in delete_row_idx:
        df = df.drop([i],axis = 0)
    return df

# 사무판단이 0인데 대분류가 분류 되어있는 경우
def x_wrong_drop(df):
    delete_row_idx = list(df[(df['사무판단']==0) & (df['사무유형(대분류)']!=0)].index)
    delete_row_idx.sort(reverse = True)
    for i in delete_row_idx:
        df = df.drop([i],axis = 0)
    return df

# 사무판단 없는 행 비사무 처리
def no_work_check(df):
    df.loc[df['사무판단'].isna(), '사무판단'] = 0
    df['사무판단'] = df['사무판단'].astype(int)
    return df

# 대분류 라벨 생성
def make_large_type(df):
    df.loc[df['사무유형(대분류)']=='국가', '사무유형(대분류)'] = 1
    df.loc[df['사무유형(대분류)']=='지방', '사무유형(대분류)'] = 2
    df.loc[df['사무유형(대분류)']=='공동', '사무유형(대분류)'] = 3
    df['사무유형(대분류)'] = df['사무유형(대분류)'].astype(int)
    return df

# rule-base 제거 가능 열 생성
def rule_based(df):
    df['rule_based'] = 1

    # 조문이 결측치인 행
    df.loc[df['조문'].isnull(), 'rule_based'] = 0

    # '^제.*\)$' 표현 0으로
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'^제.*\)$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'제\d+조$')), 'rule_based'] = 0

    # '^제(\d+)(장|절)' 표현 0으로
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'^제(\d+)(장|절|편)')), 'rule_based'] = 0

    # 조문제목 == '목적'|'정의' 0으로
    df.loc[((df['조문제목'].notnull()) & ((df['조문제목'] == '목적') | (df['조문제목'] == '정의'))), 'rule_based'] = 0

    # 삭제된 조문 0으로
    df.loc[((df['조문'].str.contains("삭제 <"))|(df['조문'].str.contains("삭제<"))), 'rule_based'] = 0

    # 조문이 '목적', '명칭'뿐인 것
    df.loc[(df['조문'].notnull()) & ((df['조문'].str.match(r"\d+\. 목적")) | (df['조문'] == '목적')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & ((df['조문'].str.match(r"\d+\. .*명칭")) | (df['조문'] == '명칭')), 'rule_based'] = 0

    # 조문에서 사무판단 무조건 0인 표현들
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*있는 경우$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*없게 된 경우$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*거짓이나 그 밖의 부정한 방법으로 지정을 받은 경우')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*의사를 밝히는 경우')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*인정되는 경우')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*친족이었던 경우')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*대리인이었던 경우')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\..*인증을 받은 경우')), 'rule_based'] = 0

    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*후견인$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*소재지')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*정관')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*계획서')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*서류')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*상호$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*증명서')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*경매')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*등록증')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*주소$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*절차$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*출연금')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*정도$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*대표자')), 'rule_based'] = 0


    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*공고의 방법')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*대통령령으로 정하는 사항')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*되지 아니한 자')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'\d+\. .*그 밖에 필요한 사항')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*협회는 법인으로 한다')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*이사회.*사항$')), 'rule_based'] = 0
    df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*회계.*사항$')), 'rule_based'] = 0

    ## new_수행주체 열도 필요한 경우
    #df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*이 포함되어야 한다')) & (df['new_수행주체'].isnull()), 'rule_based'] = 0
    #df.loc[(df['조문'].notnull()) & (df['조문'].str.match(r'.*가 포함되어야 한다')) & (df['new_수행주체'].isnull()), 'rule_based'] = 0

    return df

# 2차 사무유형 판단 결과 데이터 프에임 생성
def make_result_df(pd, y_true, y_pred, probs0, probs1, probs2, probs3):
    result_df = pd.DataFrame({
        '사무유형(대분류)' : y_true,
        '사무유형(대분류) 예측' : y_pred,
        '비사무 확률' : round(probs0, 5),
        '국가 확률' : round(probs1, 5),
        '지방 확률' : round(probs2, 5),
        '공동 확률' : round(probs3, 5),
        })
    return result_df

# 조문에서 수행주체 뽑기 -> (list, 문자열) 두개의 형태로
def new_subject_make(df, subject_list, tqdm):
    df['수행주체(리스트)'] = ''  # 초기 열 설정
    df['수행주체(문자열)'] = ''  # 초기 열 설정
    
    for i in tqdm(df.index, desc="Processing rows"):
        text = df.loc[i, '조문']
        new_subject_list = [subject for subject in subject_list if subject in text]
        df.at[i, '수행주체(리스트)'] = new_subject_list
        df.loc[i, '수행주체(문자열)'] = ' '.join(new_subject_list)

    return df

# 수행주체 기준 새로운 열 생성
def score_subject(df,subject_dic,tqdm):
    df['score_subject_nan'] = 0
    df['score_subject_n'] = 0
    df['score_subject_r'] = 0
    df['score_subject_p'] = 0
    df['score_subject_len'] = 0
    for ii in tqdm(df.index, desc="Processing rows"):
        s = df.loc[ii, '수행주체(리스트)']
        if len(s) != 0:
            for jj in s:
                df.at[ii, 'score_subject_nan'] = df.at[ii, 'score_subject_nan'] + subject_dic[jj][0]
                df.at[ii, 'score_subject_n'] = df.at[ii, 'score_subject_n'] + subject_dic[jj][1]
                df.at[ii, 'score_subject_r'] = df.at[ii, 'score_subject_r'] + subject_dic[jj][2]
                df.at[ii, 'score_subject_p'] = df.at[ii, 'score_subject_p'] + subject_dic[jj][3]
        df.at[ii, 'score_subject_len'] = len(s)
    return df

# 앙상블 결과 저장 ('사무유형(대분류) 예측' 열 추가)
def make_ensemble_pred(df):
    cols = ['비사무 확률', '국가 확률', '지방 확률', '공동 확률']
    df['사무유형(대분류) 예측'] = df[cols].idxmax(axis=1)
    for i in range(len(cols)):
        df.loc[df['사무유형(대분류) 예측']==cols[i], '사무유형(대분류) 예측'] = i
    
    return df

# 앙상블 결과에 확실, 주의 구분('need_to_check' 열 추가)
def make_need_to_check(e_result, rf_result, encoder_result):
    good_df = (e_result[(rf_result['사무유형(대분류) 예측'] == encoder_result['사무유형(대분류) 예측'])])
    bad_df = (e_result[(rf_result['사무유형(대분류) 예측'] != encoder_result['사무유형(대분류) 예측'])])
    e_result['need_to_check'] = 0
    e_result.loc[bad_df.index,'need_to_check'] = 1
    
    return e_result, good_df, bad_df

# [확실, 주의] 정확도 확인 함수
def check_result(df):
    jF = df.loc[(df['need_to_check']==1)&(df['사무유형(대분류)']!=df['사무유형(대분류) 예측']), :]
    jT = df.loc[(df['need_to_check']==1)&(df['사무유형(대분류)']==df['사무유형(대분류) 예측']), :]
    hF = df.loc[(df['need_to_check']==0)&(df['사무유형(대분류)']!=df['사무유형(대분류) 예측']), :]
    hT = df.loc[(df['need_to_check']==0)&(df['사무유형(대분류)']==df['사무유형(대분류) 예측']), :]
    
    print(f'주의 사무 개수: {len(jF) + len(jT)}')
    print(f'주의에서 틀린 개수: {len(jF)}')
    print(f'주의에서 맞은 개수: {len(jT)}')
    print('------------------')
    print(f'확실 사무 개수: {len(hF) + len(hT)}')
    print(f'확실에서 틀린 개수: {len(hF)}')
    print(f'확실에서 맞은 개수: {len(hT)}')
    
    h_ratio = len(hT)/(len(hF)+len(hT))
    j_ratio = len(jT)/(len(jF)+len(jT))
    
    print("============")
    print(f"확실 정확도: {h_ratio}")
    print(f"애매 정확도: {j_ratio}")
    
    return h_ratio, j_ratio  # 확실 사무중 맞은 것의 개수

# 2차 필터까지 진행한 결과 전체 데이터에 저장
def make_final_df(sub_df, sencond_filtered_df):
    # 열추가
    sub_df['수행주체'] = sencond_filtered_df['수행주체(리스트)']
    sub_df['rule_based'] = sencond_filtered_df['rule_based']
    sub_df['사무유형(대분류) 결과'] = sencond_filtered_df['사무유형(대분류) 예측']
    sub_df['비사무 확률'] = sencond_filtered_df['비사무 확률']
    sub_df['국가 확률'] = sencond_filtered_df['국가 확률']
    sub_df['지방 확률'] = sencond_filtered_df['지방 확률']
    sub_df['공동 확률'] = sencond_filtered_df['공동 확률']
    sub_df['need_to_check'] = sencond_filtered_df['need_to_check']
    
    # 열의 빈값 처리
    sub_df.loc[sub_df['사무유형(대분류) 결과'].isna(), '사무유형(대분류) 결과'] = 0
    sub_df.loc[sub_df['rule_based'].isna(), 'rule_based'] = 0
    sub_df.loc[sub_df['need_to_check'].isna(), 'need_to_check'] = 0
    sub_df.loc[sub_df['수행주체'].isna(), '수행주체'] = ''
    sub_df.loc[sub_df['비사무 확률'].isna(), '비사무 확률'] = 0
    sub_df.loc[sub_df['국가 확률'].isna(), '국가 확률'] = 0
    sub_df.loc[sub_df['지방 확률'].isna(), '지방 확률'] = 0
    sub_df.loc[sub_df['공동 확률'].isna(), '공동 확률'] = 0
    
    # 자료형 정리
    sub_df['사무유형(대분류) 결과'] = sub_df['사무유형(대분류) 결과'].astype(int)
    sub_df['need_to_check'] = sub_df['need_to_check'].astype(int)
    
    return sub_df