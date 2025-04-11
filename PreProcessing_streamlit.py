import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings

warnings.filterwarnings("ignore")  # sklearn thread 관련 경고 무시

st.title("📊 센서 데이터 전처리 프로그램")

uploaded_file = st.file_uploader("CSV 파일을 업로드하세요", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 원본 데이터 미리보기", df.head())

    st.write("---")
    st.subheader("🛠️ 수행할 전처리 작업 선택")

    # 사용자 선택
    apply_datetime = st.checkbox("1. 시계열 열을 datetime으로 변환")
    apply_corr_missing = st.checkbox("2. 상관관계+결측치 기반 열 제거 및 평균 대체")
    apply_passfail_encoding = st.checkbox("3. Pass_Fail 열 인코딩 (-1→0, 1→1)")
    apply_smote = st.checkbox("4. SMOTE로 데이터 균형 맞춤")
    apply_scaling = st.checkbox("5. 정규화 (MinMaxScaler)")

    time_column = None
    if apply_datetime:
        time_column = st.selectbox("datetime으로 변환할 열 선택", df.columns)

    if st.button("🚀 전처리 시작"):
        with st.spinner("전처리 중입니다..."):

            # 1. datetime 변환
            if apply_datetime and time_column:
                df[time_column] = pd.to_datetime(df[time_column])

            # 2. 상관관계 + 결측치 처리
            if apply_corr_missing:
                if 'Pass_Fail' in df.columns:
                    sensor_columns = [col for col in df.columns if 'Sensor' in col and col != 'Pass_Fail']
                    correlations = df[sensor_columns + ['Pass_Fail']].corr()['Pass_Fail'].abs()
                    columns_to_drop = [col for col in sensor_columns if (correlations[col] < 0.05) and (df[col].isna().mean() > 0.1)]
                    df.drop(columns=columns_to_drop, axis=1, inplace=True)
                    df.fillna(df.mean(), inplace=True)

            # 3. Pass_Fail 인코딩
            if apply_passfail_encoding:
                if 'Pass_Fail' in df.columns:
                    st.write("Pass_Fail 변환 전 클래스 분포:")
                    st.write(df['Pass_Fail'].value_counts())
                    df['Pass_Fail'] = df['Pass_Fail'].map({-1: 0, 1: 1})
                    st.write("변환 후 클래스 분포:")
                    st.write(df['Pass_Fail'].value_counts())

            # 4. SMOTE
            if apply_smote:
                try:
                    if 'Pass_Fail' in df.columns:
                        X = df.drop('Pass_Fail', axis=1).select_dtypes(include=[np.number])
                        y = df['Pass_Fail'].astype(int)

                        counter = Counter(y)
                        min_class = min(counter.values())

                        if min_class > 1:
                            smote = SMOTE(random_state=42, k_neighbors=min(min_class - 1, 5))
                            X_resampled, y_resampled = smote.fit_resample(X, y)
                            df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='Pass_Fail')], axis=1)
                        else:
                            st.warning(f"SMOTE를 적용할 수 없습니다. 소수 클래스 데이터 수가 너무 적습니다 (min class: {min_class})")
                except Exception as e:
                    st.error(f"SMOTE 적용 중 오류 발생: {str(e)}")

            # 5. 정규화
            if apply_scaling:
                sensor_cols = [col for col in df.columns if 'Sensor' in col and col != 'Pass_Fail']
                scaler = MinMaxScaler()
                df[sensor_cols] = scaler.fit_transform(df[sensor_cols])

        st.success("전처리 완료!")
        st.write("### 전처리된 데이터 미리보기", df.head())

        # CSV 다운로드
        csv = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 전처리 결과 다운로드", data=csv, file_name='preprocessed_data.csv', mime='text/csv')
