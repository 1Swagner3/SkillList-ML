def clean_data(df):
    class_counts = df['Label'].value_counts()
    to_keep = class_counts[class_counts >= 2].index
    df_cleaned = df[df['Label'].isin(to_keep)]
    return df_cleaned